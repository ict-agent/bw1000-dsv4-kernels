# bw1000-dsv4-kernels

DeepSeek V4 Flash W8A8 推理 kernel 优化，针对海光 DCU BW (gfx936)。

## 内容
- `kernels/` — 5 个 HIP C++ kernel（per_token_quant, silu_and_mul, silu_mul_quant, rmsnorm, add_rmsnorm_quant）
- `hip_kernels/` — 完整源码 + 验证/集成脚本
- `traces/` — 22 个 JSON trace（含 6 配置 A/B + MoE profile）
- `dsv4_ops_unit_tests/` — baseline 测试集
- `KERNEL_VERIFICATION_FINAL_REPORT.md` — 完整验证报告

## 关键结果
- CUDA graph: TPOT 202→73ms (2.76×)
- HIP kernel (graph+并发): 76.1 vs 72.0 tok/s (+5.7%)
- 单 kernel: per_token_quant 2.6-8×, silu_mul_quant 21×
- 正确性: bit-exact vs SOTA Triton
