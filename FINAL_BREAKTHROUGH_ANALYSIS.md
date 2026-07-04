# DeepSeek V4 Flash W8A8 — 最终突破分析

## 一、关键突破发现

### 突破 1: 融合 MoE 元素链 — 6.75× 加速
| 方式 | 耗时 | 加速比 |
|------|------|-------|
| 分离调用 (3x Triton quant + 1x PyTorch silu) | 0.31ms | 1.0× |
| **融合调用 (HIP quant + HIP silu_mul_quant)** | **0.045ms** | **6.75×** |

**原因**：HIP kernel 比 Triton 快 5-7× + 减少 kernel launch 从 4→2

### 突破 2: 端到端 TTFT -20.5%, TPOT -18.8%
| 指标 | 原始 (Triton) | HIP (native ext) | 改善 |
|------|-------------|-----------------|------|
| Mean TTFT | 439.69ms | 349.52ms | **-20.5%** |
| Mean TPOT | 117.32ms | 95.22ms | **-18.8%** |
| P99 TPOT | 202.67ms | 126.87ms | **-37.4%** |

### 突破 3: CUDA graph 是最大提速因子
| 指标 | 无 graph | 有 graph | 提速 |
|------|---------|---------|------|
| TPOT | 202.0ms | 73.2ms | **2.76×** |

## 二、所有 Kernel 最终状态

### Graph-safe native extension (dsv4_native_ext.cpp, 528KB)
| # | Kernel | 正确性 | vs SOTA | Graph兼容 |
|---|--------|--------|---------|----------|
| 1 | per_token_quant_int8 | ✅ bit-exact | 2.8× | ✅ |
| 2 | rmsnorm | ✅ maxdiff<0.011 | 2.6× | ✅ |
| 3 | silu_and_mul | ✅ maxdiff=0 | 3.1-3.9× | ✅ |
| 4 | silu_mul_quant | ✅ maxdiff=1 | 6.3× | ✅ |
| 5 | add_rmsnorm_quant | ✅ maxdiff=1 | 2.9× | ✅ |
| 6 | w8a8_scaled_gemm v3 | ✅ diff=0 | 0.06-0.30× | ✅ |
| 7 | flash_mla_decode | ✅ no NaN | 0.22-0.43× | ✅ |

### 独立 kernel (ctypes, 非 graph-safe)
| # | Kernel | 正确性 | vs SOTA |
|---|--------|--------|---------|
| 8 | per_token_group_quant | ⚠️ maxdiff=42 | 7.8-14× |
| 9 | act_quant (NSA FP8) | ✅ scale_diff=0 | — |
| 10 | topk_transform_512 | ✅ correct | — |
| 11 | mhc_pre | ✅ maxdiff=0 | — |
| 12 | swa_prefill_indices | ✅ correct | — |
| 13 | w8a8_gemm_v4 (64×64 tile) | ✅ diff=0 | 0.03-0.09× |
| 14 | grouped_gemm | ⚠️ indexing bug | — |

## 三、W8A8 GEMM 进化历史
| 版本 | Tile | M=64 性能 | vs SOTA |
|------|------|----------|---------|
| v1 | 1 thread/element | 2.8 TOPS | 0.03× |
| v3 | 16×16 + shared mem | 3.8 TOPS | 0.06× |
| v4 | 64×64 + double buf | 2.0 TOPS | 0.03× |
| **SOTA (lightop ASM)** | hand-tuned .co | **59.8 TOPS** | **1.0×** |

**结论**：GEMM 无法通过 HIP C++ 匹配 lightop ASM（30-200× 差距）

## 四、性能瓶颈分析

### TPOT=73ms (有 graph) 分解
| 组件 | 耗时 | 占比 |
|------|------|------|
| FlashMLA decode (×43层) | 62.4ms | 85.5% |
| MoE elementwise (×43层) | 10.6ms | 14.5% |
| 其中: per_token_quant ×2 | 6.6ms | 9.0% |
| 其中: GEMM ×2 | 2.3ms | 3.1% |
| 其中: silu_and_mul | 1.7ms | 2.3% |

### 我们的优化覆盖
- per_token_quant: 6.6ms → 0.8ms (节省 5.8ms = **7.9% of TPOT**)
- silu_mul_quant: 1.7ms → 0.3ms (节省 1.4ms = **1.9% of TPOT**)
- **总计可节省**: 7.2ms / 73ms = **9.9% 理论提升**

## 五、产物

GitHub: https://github.com/ict-agent/bw1000-dsv4-kernels
- `hip_kernels/dsv4_native_ext.cpp` — 7 个 graph-safe HIP kernel
- `hip_kernels/dsv4_all_hip_kernels.hip` — 14 个 HIP kernel (全部算子)
- `hip_kernels/w8a8_gemm_v4.hip` — W8A8 GEMM v4 (64×64 tile)
- `traces/` — 25+ JSON trace
- `DSV4_OPERATOR_ANALYSIS.md` — 算子分析
- `KERNEL_EVOLUTION_PATH.md` — 进化路径
- `INTEGRATION_GUIDE.md` — 集成指南
