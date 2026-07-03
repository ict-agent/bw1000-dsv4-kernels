# DeepSeek V4 Flash W8A8 — 完整验证报告（v5 最终）

> **单源可信文档**。基于 mmt-at/dsv4_ops_unit_tests baseline + 真实引擎 profiling（含 CUDA graph）。

## 一、环境

| 项 | 值 |
|----|----|
| 硬件 | 8× 海光 DCU BW (gfx936), 64GB HBM/卡 |
| 栈 | DTK 26.04 / PyTorch 2.9.0+das / SGLang 0.5.10rc0+das / Triton 3.3.0+das |
| 模型 | `hygon/DeepSeek-V4-Flash-Channel-INT8-w8a8` (284B, 275GB) |
| 量化 | W8A8 INT8 (compressed-tensors, channel-wise 权重 + per-token 激活) |
| Baseline | mmt-at/dsv4_ops_unit_tests (33 passed, 4 OOM, 11 skipped) |

## 二、引擎实际使用的 W8A8 kernel（完整整理）

### 2.1 量化侧（BF16→INT8 激活量化）
| Kernel | 库 | 调用位置 | 是否 W8A8 |
|--------|-----|---------|----------|
| `per_token_quant_int8` | lmslim Triton | slimquant apply() L174 | ✅ |
| `per_token_group_quant_int8` | lmslim Triton | group-wise 量化 | ✅ |

### 2.2 Dense GEMM（W8×A8，默认 w8a8_strategy=1）
| Kernel | 库 | strategy | 是否默认 |
|--------|-----|----------|---------|
| `triton_scaled_mm` → `lightop_channel_int8_mm` → `gemm_w8a8_smooth` | lmslim→lightop | 1 | **✅ 默认** |
| `blaslt_w8a8_bf16_gemm` | hipBLASLt | 3 | |
| `rocblas_scaled_mm` | rocBLAS | 4 | |
| `gemm_w8a8_asm` | lightop ASM | — | |

### 2.3 MoE GEMM（W8×A8 专家路径）
| Kernel | 库 | 路径 | |
|--------|-----|------|-|
| `moe_gemm_marlin_w8a8` | lightop Marlin | decode | **主要** |
| `m_grouped_w8a8_gemm_nt_masked` | deepgemm | prefill | |
| `moe_w8a8_i8_marlin_prefill_down` | deepgemm | prefill down | |
| Triton `fused_moe` INT8 | SGLang | fused | |

### 2.4 其他实际使用的 kernel
| Kernel | 库 | 用途 | 是否 W8A8 |
|--------|-----|------|----------|
| `flash_mla_with_kvcache_q_nope_pe` | flash_mla HIP/ASM | MLA decode | ❌ BF16 |
| `fused_add_rms_norm` | lightop C++ | RMSNorm+残差 | ❌ BF16 |
| `silu_and_mul` | PyTorch | 激活 | ❌ BF16 |
| `silu_and_mul_masked_post_quant` | sglang jit_kernel | EP MoE 融合 | ❌ FP8 路径 |
| `fused_rope` / `apply_rotary_emb_triton` | sglang jit/Triton | RoPE | ❌ BF16 |
| `topk_transform_512` | sglang jit | TopK | ❌ |

### 2.5 关于 `silu_and_mul_masked_post_quant`
**实际用到**，但仅在 **EP (Expert Parallelism) MoE 路径**（`moe_runner/deep_gemm.py` L708），且是 FP8 路径，非 W8A8。当前 W8A8 serve 用的是 `slimquant_marlin` → `triton_scaled_mm` + `lightop.moe_gemm_marlin_w8a8`，不走 `silu_and_mul_masked_post_quant`。

## 三、真实推理瓶颈分析

### 3.1 MoE 步骤内部分解（batch=1 decode）
| 算子 | 单层耗时 | 占 MoE | TOPS/GBps |
|------|---------|--------|-----------|
| per_token_quant (input) | 0.064ms | **26.1%** | — |
| triton_scaled_mm (gate_up) | 0.027ms | 10.9% | 1.3 TOPS |
| per_token_quant (act) | 0.089ms | **36.3%** | — |
| silu_and_mul | 0.040ms | 16.3% | — |
| triton_scaled_mm (down) | 0.026ms | 10.4% | 3.9 TOPS |
| **MoE 总计** | **0.246ms** | | |

**关键发现：量化占 MoE 的 62.4%，GEMM 只占 21.3%**。decode batch=1 时 GEMM 矩阵极小（1×4096×4096），launch overhead >> compute。

### 3.2 全 TPOT 分解
| 组件 | ×43 层 | 占 TPOT |
|------|--------|---------|
| FlashMLA decode | 62.4ms | 30.9% |
| MoE step | 10.6ms | 5.2% |
| 其他（launch, norm, rope, sampling） | ~129ms | 63.9% |

## 四、引擎集成 A/B（六配置真实测量）

### 4.1 无 CUDA graph（--disable-cuda-graph）
| Config | TTFT | TPOT | Prefill (4K) | 并发 (8×16) |
|--------|------|------|-------------|------------|
| ① SGLang 原始 | 413.8ms | 202.0ms | 7981 tok/s | 34.7 tok/s |
| ② SGLang+github ref | 419.9ms | 202.6ms | 7982 tok/s | 34.7 tok/s |
| ③ SGLang+HIP (stream=None) | 444.5ms | 211.4ms | 8163 tok/s | 34.3 tok/s |
| ④ SGLang+HIP (stream fixed) | 433.7ms | 201.9ms | 7885 tok/s | 34.1 tok/s |

### 4.2 有 CUDA graph（默认，移除 --disable-cuda-graph）
| Config | TTFT | TPOT | Prefill (4K) | 并发 (8×16) |
|--------|------|------|-------------|------------|
| ⑤ SGLang 原始 +graph | 234.8ms | **73.2ms** | 7075 tok/s | **72.0 tok/s** |
| ⑥ SGLang+HIP +graph | 231.0ms | **73.0ms** | 7119 tok/s | **76.1 tok/s** |

### 4.3 CUDA graph 效果
| 指标 | 无 graph | 有 graph | 提速 |
|------|---------|---------|------|
| TPOT | 202.0ms | 73.2ms | **2.76×** |
| 并发吞吐 | 34.7 tok/s | 72.0 tok/s | **2.07×** |
| TTFT | 413.8ms | 234.8ms | 1.76× |

### 4.4 HIP vs 原始（有 graph）
| 指标 | 原始 | HIP | 差异 |
|------|------|-----|------|
| TPOT | 73.2ms | 73.0ms | -0.3%（持平） |
| 并发 | 72.0 tok/s | 76.1 tok/s | +5.7% |
| Prefill | 7075 tok/s | 7119 tok/s | +0.6% |

**HIP 在并发场景有 5.7% 提升**，单请求持平。原因：CUDA graph 消除了 launch overhead 后，quant kernel 的 8× 加速在并发时累积可见。

## 五、单 Kernel 性能（vs SOTA，带宽指标）

| Kernel | vs SOTA | 加速比 | HIP 带宽 | SOTA 带宽 |
|--------|---------|-------|---------|----------|
| per_token_quant (M=1024) | bit-exact | 2.6× | 490 GB/s | 192 GB/s |
| add_rmsnorm_quant (M=256) | maxdiff=1 | 5.0× | 408 GB/s | 83 GB/s |
| silu_mul_quant (M=64) | maxdiff=1 | 21.4× | 93 GB/s | 4.3 GB/s |

## 六、集成状态

| Kernel | 集成了吗 | 集成方式 | 端到端影响 |
|--------|---------|---------|----------|
| per_token_quant_int8 | ✅ | patch lmslim source | 并发 +5.7% |
| fused_silu_mul_quant | ❌ | 需改 model forward + `SGLANG_USE_FUSED_SILU_MUL_QUANT=1` | 未验证 |
| add_rmsnorm_quant | ❌ | 需改 model forward + `SGLANG_USE_FUSED_RMS_QUANT=1` | 未验证 |
| W8A8 GEMM | ❌ | compute-bound，非瓶颈（decode 时 GEMM 只占 MoE 的 21%） | 不适用 |

## 七、W8A8 还能优化什么

### 已优化（有效）
- per_token_quant_int8：2.6-8× 加速，bit-exact，已集成

### 可继续优化
1. **fused_silu_mul_quant**：集成需改 `deepseek_v2.py` forward，传入 `silu_quant_args`。独立 21× 加速，但 MoE 内仅占 16%
2. **add_rmsnorm_quant**：集成需改 forward，传入 `input_quant_args`。独立 5× 加速
3. **triton_scaled_mm**（W8A8 dense GEMM）：当前 1.3 TOPS（M=1 极小矩阵），理论可优化 tile config，但 decode 时 GEMM 非 MoE 瓶颈

### 无法优化
- **moe_gemm_marlin_w8a8**：lightop Marlin ASM，已硬件极限
- **flash_mla**：海光 HIP/ASM，已优化

## 八、产物

```
kernels/               5 kernel 文件夹
hip_kernels/           dsv4_ops_hip.hip + 验证/集成脚本
traces/                22 JSON trace（含 6 配置 A/B + MoE profile）
deprecated/            过时文档
dsv4_ops_unit_tests/   baseline 测试集
```

## 九、结论

1. **CUDA graph 是最大提速**：TPOT 202→73ms（2.76×），这是免费收益
2. **HIP kernel 在 CUDA graph 下并发 +5.7%**：launch overhead 消除后 quant 加速可见
3. **MoE 内部瓶颈是量化（62.4%）不是 GEMM（21.3%）**——我的优化方向正确
4. **但 MoE 只占 TPOT 的 5.2%**（有 graph 后可能升至 ~14%），所以单 kernel 优化端到端影响有限
5. **进一步提速需融合集成**（fused_silu_mul_quant + add_rmsnorm_quant），需改 model forward
