# bw1000-dsv4-kernels

DeepSeek V4 Flash W8A8 推理 kernel 优化，针对海光 DCU BW (gfx936)。

## 仓库导航

```
├── README.md                          ← 本文件
├── DSV4_OPERATOR_ANALYSIS.md          ← SGLang 推理 DSV4 时每个算子的实际实现分析
├── INTEGRATION_GUIDE.md               ← 如何将 HIP kernel 集成到 SGLang 引擎
│
├── hip_kernels/                       ← 所有 HIP kernel 源码和编译库
│   ├── dsv4_native_ext.cpp            ← PyTorch C++ native extension（7 个 graph-safe kernel）
│   ├── dsv4_all_hip_kernels.hip       ← 全部 14 个 HIP kernel（含 per_token_group_quant 等）
│   ├── w8a8_gemm_v5.hip              ← W8A8 INT8 GEMM（sdot4 + shared memory tiling）
│   ├── w8a8_gemm_intmm_fused.hip     ← W8A8 GEMM fused scale（_int_mm + HIP scale epilogue）
│   ├── hip_graph_safe.py             ← .pth 注入模块（graph-safe 集成 per_token_quant）
│   ├── libdsv4_ops_hip.so            ← 编译好的 HIP kernel 共享库
│
├── traces/                            ← 所有验证和性能 trace（JSON）
│   ├── final_correct_status.json     ← 7 个 native ext kernel 最终状态
│   ├── bench_serving_ab.json         ← SGLang 官方 bench_serving A/B 结果
│   ├── bench_serving_prefill_sweep.json ← 不同输入长度 prefill 性能
│   ├── fused_intmm_results.json      ← _int_mm + HIP fused scale GEMM 结果
│   ├── definitive_verification.json   ← bit-exact + 引擎保真度验证
│   ├── breakthrough_attempts.json     ← 融合 MoE 链 6.75× 突破
│   ├── overhead_reduction.json       ← native ext vs ctypes 开销分析
│   ├── fair_comparison.json          ← 公平性能对比（带宽指标）
│   ├── gemm_all_approaches.json     ← 所有 GEMM 后端对比
│   ├── accuracy_and_metrics.json     ← 推理精度 + TTFT/TPOT 指标
│
├── dsv4_ops_unit_tests/              ← mmt-at/dsv4_ops_unit_tests baseline 测试集
│   ├── tests/                        ← 单元测试（per_token_quant, flash_mla, MoE GEMM 等）
│   ├── utils/model_config.py         ← DeepSeek V4 Flash 模型配置
│   └── conftest.py                   ← pytest 配置
```

## 可用 Kernel 列表

### Graph-safe Native Extension（`dsv4_native_ext.cpp`，7 个 kernel）

| # | Kernel | 正确性 | 加速比 | Graph | 已集成 | 使用方式 |
|---|--------|--------|-------|-------|--------|---------|
| 1 | `per_token_quant_int8` | ✅ bit-exact | **5.8×** | ✅ | ✅ | `.pth` → patch lmslim |
| 2 | `rmsnorm` | ✅ maxdiff<0.012 | **4.7×** | ✅ | ❌ | 需 patch lightop |
| 3 | `silu_and_mul` | ✅ maxdiff=0 | **7.8×** | ✅ | ❌ | 需改 model forward |
| 4 | `silu_mul_quant` | ✅ maxdiff=1 | **13.0×** | ✅ | ❌ | 需改 model forward |
| 5 | `add_rmsnorm_quant` | ✅ maxdiff=1 | **4.8×** | ✅ | ❌ | 需改 model forward |
| 6 | `w8a8_gemm` | ✅ diff=0 | 0.80× | ❌ | ❌ | _int_mm+HIP scale, M≥32 |
| 7 | `flash_mla_decode` | ✅ no NaN | 0.22× | ✅ | ❌ | 简化版, SOTA 已是 ASM |

### 其他 HIP Kernel（`dsv4_all_hip_kernels.hip`，14 个 kernel）

| # | Kernel | 正确性 | 说明 |
|---|--------|--------|------|
| 8 | `per_token_group_quant_int8` | ✅ bit-exact | 修复 cross-wavefront reduction |
| 9 | `fused_rope` | ✅ 编译通过 | 需精度验证 |
| 10 | `silu_mul_masked_quant` | ✅ 编译通过 | EP MoE 路径 |
| 11 | `mhc_pre` | ✅ maxdiff=0 | MHC sigmoid+mix |
| 12 | `mhc_post` | ⚠️ stub | 需完整实现 |
| 13 | `hc_split_sinkhorn` | ⚠️ stub | 需完整实现 |
| 14 | `act_quant_fp8` | ✅ scale_diff=0 | NSA FP8 量化 |
| 15 | `topk_transform_512` | ✅ correct | TopK 变换 |
| 16 | `swa_prefill_indices` | ✅ correct | SWA 索引 |
| 17 | `merge_attn_states` | ✅ 编译通过 | Attention 合并 |
| 18 | `grouped_gemm` | ⚠️ indexing bug | MoE grouped GEMM |

## 已集成的端到端效果

per_token_quant_int8 通过 `.pth` 注入集成到 SGLang 引擎（CUDA graph 兼容）：

| 指标 | 原始 (Triton) | HIP (native ext) | 改善 |
|------|-------------|-----------------|------|
| Mean TTFT | 439.69ms | 349.52ms | **-20.5%** |
| Mean TPOT | 117.32ms | 95.22ms | **-18.8%** |
| P99 TPOT | 202.67ms | 126.87ms | **-37.4%** |
| 并发吞吐 | 72.0 tok/s | 102.9 tok/s | **+43%** |

（SGLang 官方 bench_serving, input=128, output=32, 16 prompts, concurrency=8）

## 集成方式

```bash
# 1. 编译 native extension
hipcc -std=c++17 -O3 -fPIC -shared \
  $(python3 -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I'+p for p in include_paths()]))") \
  -I$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") \
  -I/opt/dtk/hip/include \
  -DTORCH_API_INCLUDE_EXTENSION_H -D__HIP_PLATFORM_AMD__ \
  -DTORCH_EXTENSION_NAME=dsv4_native_ext \
  -o dsv4_native_ext.cpython-310-x86_64-linux-gnu.so \
  dsv4_native_ext.cpp \
  -L$(python3 -c "import torch; print(torch.__file__.rsplit('/',1)[0])")/lib \
  -ltorch -ltorch_python -lc10 \
  -lpython3.10 -Wl,-rpath,$(python3 -c "import torch; print(torch.__file__.rsplit('/',1)[0])")/lib

# 2. 安装 .pth 注入
cp hip_graph_safe.py /usr/local/lib/python3.10/dist-packages/
echo 'import hip_graph_safe; hip_graph_safe._apply_patch() if __import("os").environ.get("SGLANG_USE_HIP_QUANT","0")=="1" else None' \
  > /usr/local/lib/python3.10/dist-packages/zz_hip_graph_safe.pth

# 3. 启动 SGLang
export SGLANG_USE_HIP_QUANT=1
python3 -m sglang.launch_server --model-path $MODEL_PATH --tp 8 --quantization slimquant_marlin ...
```

## 验证方式

```bash
# 单 kernel 正确性 + 性能
python3 hip_kernels/final_correct_status.py

# 引擎 A/B benchmark
export HF_ENDPOINT=https://hf-mirror.com
python3 -m sglang.bench_serving --backend sglang-oai --port 30001 \
  --model $MODEL_PATH --tokenizer $MODEL_PATH \
  --dataset-name random --random-input-len 128 --random-output-len 32 \
  --num-prompts 16 --max-concurrency 8 --seed 42
```

## W8A8 GEMM 进化

| 版本 | 方法 | M=64 性能 | vs SOTA |
|------|------|----------|---------|
| v1-v5 | HIP C++ sdot4 | 2-5 TOPS | 0.03-0.05× |
| **FUSED** | **_int_mm + HIP scale** | **76 TOPS** | **0.80×** |
| SOTA | lightop ASM .co | 98 TOPS | 1.0× |

lightop ASM 反汇编发现：不使用 MFMA，用 `v_dot4_i32_i8`（与我们的 `__builtin_amdgcn_sdot4` 相同）。差距在寄存器分配和指令调度（3137 条手工排列指令 vs hipcc 自动生成）。
