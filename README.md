# DeepSeek V4 Flash HIP Kernels (gfx936)

14 个 HIP kernel，覆盖 DeepSeek V4 Flash W8A8 推理中 Triton/TileLang 路径的 elementwise/index/attention 辅助算子。目标：Hygon DCU gfx936，CUDA graph 兼容，集成进 sglang 引擎。

## Kernel 列表

| # | Kernel | 文件 | 正确性 | 性能 vs SOTA |
|---|--------|------|--------|------|
| 1 | per_token_quant_int8 | k01 | bit-exact | 8.2x |
| 2 | per_token_group_quant_int8 | k02 | bit-exact | 14.8x |
| 3 | rmsnorm_self | k03 | maxdiff<1e-5 | 6.3x |
| 4 | fused_rope (interleaved) | k04 | maxdiff=0 | 38.7x |
| 5 | silu_and_mul | k05 | maxdiff<1e-5 | 10.6x |
| 6 | silu_mul_masked_quant | k06 | masked=0 | 22.3x |
| 7 | hc_split_sinkhorn | k07 | d<1e-4 | 61.4x |
| 8 | act_quant_fp8 (NSA) | k08 | scale=0, e4m3 1-ulp | 30.2x |
| 9 | merge_attn_states | k09 | d<1e-2 | 8.9x |
| 10 | topk_transform_512 | k10 | set match | 15.78x (radix) |
| 11 | mhc_pre | k11 | maxdiff<1e-4 | 17.5x |
| 12 | mhc_post | k12 | maxdiff<0.03 | 2.2x |
| 13 | swa_prefill_indices | k13 | set match | — |
| 14 | grouped_gemm_int8 | k14 | maxdiff<0.5 | 0.77x (tile; 引擎用 vendor marlin) |

## 目录结构

```
src/                      # 14 个独立 kernel 文件 + common.hip + launchers.hip
  common.hip              # helpers + f32_to_e4m3 (bit-exact)
  k01..k14_*.hip          # 每 kernel + launch wrapper
  launchers.hip           # include 入口
build.sh                  # hipcc -O3 --offload-arch=gfx936 → libdsv4_all_hip.so
dsv4_all_hip_kernels.hip  # 单文件版（source of truth）
verify_kernels_v2.py      # 精度验证 (25/27)
test_graph_safe.py        # CUDA graph 兼容 (15/15)
tests/                    # pytest 套件 (28 passed)
bench_vs_sota.py          # vs sglang SOTA 性能
e2e_op_chain.py           # 引擎 op 链 e2e (2.28x)
bench_server.py           # 8-GPU server A/B benchmark
hip_dsv4_integration.py   # sglang monkey-patch (11 patch 生效)
sitecustomize.py serve_8gpu.sh  # 启动 + 自动 patch
REPORT.md PERF_ANALYSIS.md pytest.ini  # 文档
archive/                  # 历史实验脚本
```

## 构建

```bash
# 容器内 (gfx936, DTK hipcc)
bash build.sh              # 编译 src/ → libdsv4_all_hip.so
# 或
hipcc -O3 --offload-arch=gfx936 -shared -fPIC -o libdsv4_all_hip.so dsv4_all_hip_kernels.hip
```

## 验证

```bash
python verify_kernels_v2.py    # 精度
python test_graph_safe.py      # CUDA graph capture/replay
python bench_all.py            # 性能
```

## 引擎集成

11 个 patch 覆盖所有 kernel（`hip_dsv4_integration.py`，`sys.meta_path` import hook + buffer pool）：

```bash
export SGLANG_USE_HIP_DSV4=1
export SGLANG_HIP_PTQ=1 SGLANG_HIP_PTGQ=1 SGLANG_HIP_SILU=1 SGLANG_HIP_SILU_QUANT=1 \
       SGLANG_HIP_RMSNORM=1 SGLANG_HIP_NSA_QUANT=1 SGLANG_HIP_MHC=1 \
       SGLANG_HIP_TOPK=1 SGLANG_HIP_MERGE=1 SGLANG_HIP_ROPE=1 SGLANG_HIP_SWA=1
export SGLANG_APPLY_CONFIG_BACKUP=none
export PYTHONPATH=/workspace/hip_kernels:/workspace/sglang/python
# sitecustomize.py 自动 import hip_dsv4_integration，patch 在 sglang 模块加载时同步生效
SGLANG_APPLY_CONFIG_BACKUP=none python3 -m sglang.launch_server \
  --model-path <DeepSeek-V4-Flash-Channel-INT8-w8a8> --tp 8 \
  --quantization slimquant_marlin --moe-a2a-backend none
```

| 开关 | Patch 目标 |
|------|-----------|
| SGLANG_HIP_PTQ | `lmslim.per_token_quant_int8` |
| SGLANG_HIP_PTGQ | `lmslim.per_token_group_quant_int8` |
| SGLANG_HIP_SILU | `sglang SiluAndMul.forward_cuda` |
| SGLANG_HIP_SILU_QUANT | `lmslim.hip_silu_mul_masked_quant` (registered) |
| SGLANG_HIP_RMSNORM | `sglang jit_kernel.rmsnorm_self` |
| SGLANG_HIP_NSA_QUANT | `nsa.tilelang_kernel.act_quant` |
| SGLANG_HIP_MHC | `mhc.hc_split_sinkhorn` / `mhc_post_torch` |
| SGLANG_HIP_TOPK | `indexer.topk_transform_512_pytorch_vectorized` |
| SGLANG_HIP_MERGE | `vllm triton_merge_attn_states` |
| SGLANG_HIP_ROPE | `deepseek_v4_rope.apply_rotary_emb` |
| SGLANG_HIP_SWA | `jit_kernel.tilelang_make_swa_prefill_indices` |

详见 [REPORT.md](REPORT.md) 和 [PERF_ANALYSIS.md](PERF_ANALYSIS.md)。
