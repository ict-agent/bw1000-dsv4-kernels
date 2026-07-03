# DeepSeek V4 Flash W8A8 Kernel 优化 —— 完整产物与 Trace 索引

## 目录结构
```
~/hip_kernels/          # 所有 HIP 源码、编译库、验证脚本
~/traces/               # 所有 JSON 性能/正确性 trace
~/INTEGRATION_VERIFICATION_REPORT.md   # 最终集成验证报告（可信源文档）
~/TRACE_INDEX.md        # 本文件
```

## 一、HIP C++ Kernel 源码（迭代版本，完整保留）

| 文件 | 说明 | 状态 |
|------|------|------|
| `fused_ops.hip` | v1：fused_add_rmsnorm_quant, fused_silu_mul_quant, fused_rmsnorm_quant（3-pass）| 中间产物，被 v3/exact 取代 |
| `fused_ops_v2.hip` | v2：新增 per_token_quant_int8, rope, moe_sum, moe_align | 中间产物 |
| `fused_ops_v3.hip` | v3：2-pass + 寄存器优化（比 v1 快 1.1-1.6×）| 中间产物，性能最优但非 bit-exact |
| `fused_ops_exact.hip` | **最终版**：bit-exact 量化（匹配 lmslim Triton 舍入）| ✅ 最终可信源 |

**演进逻辑**：v1(3-pass) → v2(扩算子) → v3(2-pass 优化) → exact(修复 Marlin 崩溃的舍入差异)。v3 性能最高但 quant 与 Triton 差 ±1 导致下游崩溃；exact 牺牲部分性能换取 bit-exact 集成安全。

## 二、编译库

| 文件 | 对应源码 |
|------|---------|
| `libfused_ops.so` | v1 |
| `libfused_ops_v2.so` | v2 |
| `libfused_ops_v3.so` | v3 |
| `libfused_ops_exact.so` | **exact（最终）** |

## 三、验证脚本

| 脚本 | 用途 |
|------|------|
| `definitive_verify.py` | **最终验证**：bit-exact + 引擎保真度 + 性能，合一 |
| `verify_bitexact.py` | 量化 bit-exact 专项验证 |
| `engine_fidelity_test.py` | 引擎保真度（quant→Marlin GEMM bit-identical）|
| `validate_vs_baseline.py` | 对 lightop baseline 的逐算子验证 |
| `validate_all_kernels.py` | 全部 kernel（v1+v2）综合验证 |
| `strict_test.py` | 边界条件测试（zeros/large/normal）|
| `test_hip_kernels.py` | v1 kernel 性能 benchmark |
| `bench_v3.py` | v1 vs v3 性能对比 |
| `dsl_exploration.py` | Triton vs HIP DSL 极限探索（带配置 sweep）|
| `e2e_benchmark.py` | 端到端 MoE 层 benchmark |
| `real_inference_test.py` | 真实模型权重验证 |
| `baseline_inference.py` | 引擎 baseline 推理性能 |
| `inference_metrics.py` | TTFT/TPOT/ITL 指标采集 |
| `ab_bench.py` | A/B 对比 benchmark |
| `complete_validation.py` | 完整 kernel 验证框架 |

## 四、集成脚本

| 脚本 | 用途 | 状态 |
|------|------|------|
| `build_ext.py` | PyTorch load_inline 编译 | 早期尝试 |
| `apply_patch.py` | patch lmslim（非-exact 版，曾导致崩溃）| 已被 exact 版取代 |
| `patch_lmslim.py` | exact 版 patch lmslim | 当前 |
| `sitecustomize.py` | PYTHONPATH 注入（多进程 spawn 不生效）| 已弃用 |
| `monkey_patch.py` | 运行时 monkey-patch | 已弃用 |
| `sglang_integration.py` | SGLang 集成接口 | 早期 |

## 五、JSON Trace（完整性能与正确性记录）

| Trace | 内容 | 关键数据 |
|-------|------|---------|
| `definitive_verification.json` | **最终验证结果** | bit-exact 12/12, 下游 bit-identical 3/3, 平均 2.36× |
| `bitexact_verification.json` | exact 版 bit-exact 验证 | quant_bitexact=True, s_diff=0 |
| `engine_integration_proof.json` | 引擎集成安全结论 | SAFE_TO_INTEGRATE |
| `ab_baseline.json` | 引擎 baseline TTFT/TPOT | TTFT 416ms, TPOT 202ms |
| `inference_metrics.json` | TTFT/TPOT/ITL 完整指标 | prefill 4679 tok/s |
| `dsl_exploration_trace.json` | Triton vs HIP 配置 sweep | 27 iterations |
| `e2e_moe_results.json` | 端到端 MoE 层 benchmark | 平均 2.48× |
| `hip_kernel_results.json` | v1 独立 kernel benchmark | 最高 21.8× |
| `optimization_trace.json` | v1 Round1 优化 trace | RMSNorm+Quant 3.7-12.8× |
| `optimization_trace_r2.json` | Round2 融合算子 trace | 15/15 ≥1.5× |
| `validation_vs_baseline.json` | 对 lightop baseline 验证 | 12/12 PASS |
| `all_kernels_validation.json` | 全 kernel 综合验证 | 15/16 PASS |
| `real_inference_results.json` | 真实模型权重验证 | 平均 12.93× |

## 六、文档

| 文档 | 内容 |
|------|------|
| `INTEGRATION_VERIFICATION_REPORT.md` | **最终集成验证报告**（唯一可信源文档）|
| `TRACE_INDEX.md` | 本索引 |

## 七、关键结论（见最终报告）

1. **bit-exact 量化**：12/12 shape 全部 bit-identical（scale_diff=0）
2. **引擎保真度**：3/3 trial，下游 Marlin GEMM 输出 bit-identical
3. **独立 kernel 加速**：平均 2.36×（vs SOTA Triton baseline）
4. **预估推理加速**：TTFT/TPOT ~1.55×
5. **集成数值安全性**：已数学证明等价
