# DeepSeek V4 Flash — 14 HIP Kernel 报告

**目标**: 验证 `dsv4_all_hip_kernels.hip` 8-18 号 kernel 正确性，对比 **sglang 实际使用的 triton/tilelang/jit_kernel/lightop/lmslim SOTA** 测性能，确保 CUDA graph 兼容，集成进引擎观察端到端性能。

**环境**: 容器 `baoming_test` (sglang-deepseek-v4-dev-zkjh), Hygon DCU gfx936×8, DTK hipcc, sglang 源码, 权重 `/home_aclsylqidf/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8` (275G, 43层, 256 expert, slimquant_marlin W8A8)。

## 1. Kernel 修复（全部已修）

| # | Kernel | 问题→修复 |
|---|--------|------|
| 8 | per_token_group_quant_int8 | cross-wavefront reduction + 修正 `v/scale` 量化 → bit-exact |
| 9 | fused_rope | split-half → interleaved (real,imag 相邻), warp-per-head |
| 10 | silu_mul_masked_quant | masked 行写 0（graph-safe） |
| 12 | mhc_post | 空 stub → 完整 einsum，修正 a 转置索引 |
| 13 | hc_split_sinkhorn | 空 stub → 完整 20轮 Sinkhorn |
| 14 | act_quant_fp8 | 存 bf16 → fp8_e4m3 输出，手写 e4m3 round-to-nearest-even (bit-exact) |
| 15 | topk_transform_512 | 缺 page 变换 → 补全；long path radix select |
| 16 | swa_prefill_indices | 签名全错 → 完整重写 [num_q_tokens,128] indices |
| 17 | merge_attn_states | 空 stub → 完整 LSE merge |
| 18 | grouped_gemm | 源码缺失 → 新增 int8 tiled GEMM |

## 2. 正确性验证

- **pytest 套件** (`tests/test_kernels.py`, 14 个 Test 类, 参数化多 shape): **28 passed, 1 skipped**
- **CUDA graph** (`test_graph_safe.py`): **15/15 PASS** (capture+replay maxdiff=0)
- bit-exact: ptq/ptgq/fused_rope/silu；其余在 FP8/int8/bf16 量化噪声内

## 3. 性能：vs sglang 实际 SOTA (`bench_vs_sota.py`)

| Kernel | SOTA 对比对象 | HIP speedup |
|--------|--------------|-------------|
| per_token_quant_int8 | lmslim/triton | **7.95x** |
| per_token_group_quant_int8 | lmslim/triton | **2.44-14.5x** |
| rmsnorm_self | lightop/DCU (gemma_rmsnorm) | 1.01-1.03x |
| fused_rope | sglang/jit_kernel (tvm_ffi) | **12.2x** |
| fused_rope | sglang/triton (apply_rotary_emb_triton) | **21.5x** |
| silu_and_mul | torch/ref | 6.6-10.5x |
| merge_attn_states | vllm/triton | **2.24x** |
| topk_transform_512 | sglang/jit_kernel (ASM) | 0.14x (vendor ASM 极致) |
| hc_split_sinkhorn | sglang/tilelang | 0.27-0.86x |
| act_quant_fp8 | lightop/DCU (vendor) | 0.23-0.70x |
| grouped_gemm_int8 | torch/ref | 0.69x |

**真超 SOTA**: fused_rope (12-21x)、merge_attn (2.24x)、ptq/ptgq (2-14x)、rmsnorm (持平)。
**慢于 vendor ASM SOTA**: topk/sinkhorn/act_quant（vendor 极致优化，HIP kernel 不替换引擎 vendor 路径）。

## 4. 引擎集成 (`hip_dsv4_integration.py`)

monkey-patch 进 sglang，`sys.meta_path` import hook 同步 patch（fork TP worker 前）。**6 patch 全生效**：nsa_quant/mhc/topk/merge/rope/swa。用 **buffer pool**（`_buf`）避免 graph capture 时分配 tensor 导致 VM fault。

## 5. 端到端 8-GPU server 性能 (`bench_server.py`)

tp=8, cuda graph ON, slimquant_marlin W8A8, `--moe-a2a-backend none` (DeepEP/hcoll 缺失), cuda_graph_max_bs=256。同 config A/B 对比：

| in | out | baseline | HIP-on | 加速 |
|----|----|----|----|----|
| 128 | 64 | 3924ms | 3464ms | **1.13x** |
| 512 | 64 | 3925ms | 3459ms | **1.13x** |
| 4096 | 32 | 2327ms | 2076ms | **1.12x** |
| 4096 | 8 | 930ms | 851ms | **1.09x** |

**HIP patch 端到端加速 ~12%**，无 crash（buffer-pool 修复后）。

op-chain (`e2e_op_chain.py`, lightop GEMM + HIP elementwise): MLP step **2.28x**。

## 6. 目录结构

```
src/{common.hip, k01..k14_*.hip, launchers.hip}  # 拆分，build.sh 编译
dsv4_all_hip_kernels.hip                          # 单文件版
tests/{conftest.py, test_kernels.py}              # pytest 套件 (28 pass)
verify_kernels_v2.py  test_graph_safe.py          # 验证
bench_vs_sota.py                                  # vs sglang SOTA 性能
e2e_op_chain.py  bench_server.py  hip_dsv4_integration.py  sitecustomize.py  serve_8gpu.sh
README.md REPORT.md pytest.ini
archive/                                          # 历史脚本
```

## 7. 结论

- 14 kernel 全部修复，pytest 28 pass + graph 15/15
- vs sglang 实际 SOTA: fused_rope 12-21x、merge 2.24x、ptq/ptgq 2-14x 真超；rmsnorm 持平
- 8-GPU server e2e 加速 12%（cuda graph + slimquant_marlin，公平 A/B）
- 6 引擎 patch 全生效，buffer-pool 保证 graph-safe
