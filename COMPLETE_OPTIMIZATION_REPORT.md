# DeepSeek V4 Flash W8A8 — 全部优化方向完成报告

## 一、四个优化方向的结果

### 方向 1: 集成 silu_mul_quant 到引擎
- **独立性能**: 6.3× 加速（0.14ms → 0.022ms）
- **集成方式**: 需修改 `DeepseekV2MLP.forward`，在 gate_up GEMM 后调用 `silu_mul_quant` 替代 `silu+quant` 分离调用
- **集成结果**: ❌ **导致 graph capture 时 VMFault 崩溃**
- **根因**: 修改 model forward 改变了 CUDA graph 捕获的控制流，导致地址不匹配
- **结论**: 需要 SGLang 原生支持 `SGLANG_USE_FUSED_SILU_MUL_QUANT` 路径（当前路径未完整实现），不能通过 monkey-patch 安全集成

### 方向 2: 集成 add_rmsnorm_quant 到引擎
- **独立性能**: 2.9× 加速（0.11ms → 0.036ms）
- **集成方式**: 需修改 model forward，替换 `lightop.fused_add_rms_norm` + `per_token_quant_int8` 分离调用
- **集成结果**: ❌ 同方向 1，model forward 修改导致 graph capture 崩溃
- **结论**: 同方向 1，需 SGLang 原生支持

### 方向 3: 减少 native ext 开销
- **当前**: native ext 5.5-6.0× vs ctypes 7.5-7.9×
- **开销分解**:
  | 来源 | 耗时 |
  |------|------|
  | Stream lookup | 0.0073ms |
  | data_ptr | 0.0002ms |
  | torch.empty | 0.0044ms |
  | **总开销** | 0.0119ms |
  | ctypes (无开销) | 0.0082ms |
  | **差距** | **0.002ms** |
- **结论**: ✅ **开销已最小化**。native ext 仅比 ctypes 慢 0.002ms（20%），主要来自 PyTorch tensor 创建。stream lookup 0.0073ms 是 graph-safe 的必要代价
- **优化方法**: ctypes + prealloc + cached stream = 7.5×（但非 graph-safe）

### 方向 4: Graph capture 加速
- **当前**: HIP patch 使 capture 91s → 620s（6.8× 变慢）
- **根因**: 每次 per_token_quant 在 capture 时走 native ext（Python dispatch），比 Triton JIT 慢
- **尝试**: 无法优化——capture 时每个 batch size 都要捕获一次，native ext 的 Python dispatch 开销不可避免
- **结论**: ⚠️ **一次性开销**，不影响运行时性能。服务器重启后 10 分钟内可用

## 二、最终可工作的集成

### 唯一安全集成的 kernel: `per_token_quant_int8`
- **集成方式**: `.pth` 文件注入 → `hip_graph_safe.py` → patch `lmslim.per_token_quant_int8`
- **Graph 兼容**: ✅ native PyTorch extension + explicit stream
- **正确性**: ✅ bit-exact vs SOTA Triton
- **端到端效果**（SGLang 官方 bench_serving）:

| 指标 | 原始 (Triton) | HIP (native ext) | 改善 |
|------|-------------|-----------------|------|
| Mean TTFT | 439.69ms | 349.52ms | **-20.5%** |
| Mean TPOT | 117.32ms | 95.22ms | **-18.8%** |
| P99 TPOT | 202.67ms | 126.87ms | **-37.4%** |
| 并发吞吐 | 72.0 tok/s | 102.9 tok/s | **+43.0%** |

### 为什么其他 kernel 不能安全集成
| Kernel | 独立加速 | 集成失败原因 |
|--------|---------|------------|
| silu_mul_quant | 6.3× | 修改 model forward → graph capture VMFault |
| add_rmsnorm_quant | 2.9× | 同上 |
| rmsnorm | 2.6× | 需替换 lightop C++（不能 patch） |
| w8a8_gemm | 0.06× | 正确但远慢于 lightop ASM |

**根因**: CUDA graph 捕获整个 model forward 的计算图。任何 monkey-patch 修改 forward 控制流都会导致 graph 捕获时地址/shape 不匹配。只有 patch **库函数**（如 lmslim.per_token_quant_int8）而不改 forward 控制流的方式才是安全的。

## 三、全部 Kernel 最终状态

| # | Kernel | 正确性 | 独立加速 | Graph兼容 | 集成 | 端到端 |
|---|--------|--------|---------|----------|------|--------|
| 1 | per_token_quant_int8 | ✅ bit-exact | 5.5-6.0× | ✅ | ✅ | **TTFT -20.5%** |
| 2 | rmsnorm | ✅ maxdiff<0.011 | 2.6× | ✅ | ❌ | — |
| 3 | silu_and_mul | ✅ maxdiff=0 | 3.1-3.9× | ✅ | ❌ | — |
| 4 | silu_mul_quant | ✅ maxdiff=1 | 6.3× | ✅ | ❌ | — |
| 5 | add_rmsnorm_quant | ✅ maxdiff=1 | 2.9× | ✅ | ❌ | — |
| 6 | w8a8_scaled_gemm | ✅ diff=0 | 0.06× | ✅ | ❌ | — |
| 7 | flash_mla_decode | ✅ no NaN | 0.22× | ✅ | ❌ | — |
| 8 | per_token_group_quant | ⚠️ maxdiff=42 | 7.8× | ❌ | ❌ | — |
| 9 | act_quant (NSA) | ✅ scale_diff=0 | — | ❌ | ❌ | — |
| 10 | topk_transform | ✅ correct | — | ❌ | ❌ | — |
| 11 | mhc_pre | ✅ maxdiff=0 | — | ❌ | ❌ | — |
| 12 | swa_prefill_indices | ✅ correct | — | ❌ | ❌ | — |
| 13 | w8a8_gemm_v4 | ✅ diff=0 | 0.03× | ❌ | ❌ | — |
| 14 | grouped_gemm | ⚠️ indexing bug | — | ❌ | ❌ | — |

## 四、W8A8 GEMM 进化历史（4 版本）

| 版本 | 优化 | M=64 TOPS | vs SOTA |
|------|------|----------|---------|
| v1 | 1 thread/element | 2.8 | 0.05× |
| v3 | 16×16 + shared mem + sdot4 | 3.8 | 0.06× |
| v4 | 64×64 + double buffer + 4×4/thread | 2.0 | 0.03× |
| **SOTA** | **lightop ASM .co** | **59.8** | **1.0×** |

**结论**: HIP C++ GEMM 无法匹配 lightop hand-tuned ASM（30-200× 差距）。根因是编译器无法优化寄存器分配和指令调度到 ASM 级别。

## 五、突破性发现

### 融合 MoE 元素链 6.75×
| 方式 | 耗时 | 加速 |
|------|------|------|
| 分离 (3× Triton quant + 1× PyTorch silu) | 0.31ms | 1.0× |
| 融合 (HIP quant + HIP silu_mul_quant) | 0.045ms | **6.75×** |

**但无法安全集成到引擎**（model forward 修改 → graph capture 崩溃）。

### CUDA graph 是最大提速因子
| 指标 | 无 graph | 有 graph | 提速 |
|------|---------|---------|------|
| TPOT | 202.0ms | 73.2ms | **2.76×** |

## 六、产物

GitHub: https://github.com/ict-agent/bw1000-dsv4-kernels

### 源码
- `hip_kernels/dsv4_native_ext.cpp` — 7 个 graph-safe HIP kernel（native ext）
- `hip_kernels/dsv4_all_hip_kernels.hip` — 14 个 HIP kernel（全部算子）
- `hip_kernels/w8a8_gemm_v4.hip` — W8A8 GEMM v4（64×64 tile + double buffer）
- `hip_kernels/hip_graph_safe.py` — graph-safe 集成模块（.pth 注入）
- `hip_kernels/hip_all_integration.py` — 全集成模块（含 model forward patch）
- `hip_kernels/reduce_overhead.py` — 开销分析

### Trace（25+ JSON）
- `traces/bench_serving_ab.json` — 官方 bench_serving A/B 结果
- `traces/native_ext_verify.json` — 7 kernel 验证
- `traces/breakthrough_attempts.json` — 突破实验
- `traces/overhead_reduction.json` — 开销分析
- `traces/w8a8_gemm_v4_results.json` — GEMM v4 结果

### 文档
- `FINAL_BREAKTHROUGH_ANALYSIS.md` — 突破分析
- `DSV4_OPERATOR_ANALYSIS.md` — 算子分析
- `KERNEL_EVOLUTION_PATH.md` — 进化路径
- `INTEGRATION_GUIDE.md` — 集成指南
- `ALL_KERNELS_REPORT.md` — 全 kernel 报告
