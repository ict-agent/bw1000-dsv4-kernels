# WORKFLOW.md — 任务流程定义

每个 kernel 从需求到集成上线的完整流程。按顺序执行，每步有验证门槛。

## Phase 0: 引擎调用路径调研（必须先做）

**目的**：确认 wrapper 签名和 patch 目标对齐引擎实际调用。不调研直接写 wrapper = 必踩坑。

1. 派 Explore agent 调研该 op 在 sglang 引擎里的：
   - 精确 Python 签名（参数名/类型/默认值/返回值）
   - 实际调用 shape（decode bs=256 TP=8 + prefill）
   - wrapper 内部 Python 层创建了哪些 tensor（`torch.empty`/`.contiguous()`/`.item()`）
   - 哪些操作可移入 kernel
   - env 开关决定走哪个分支（如 SGLANG_OPT_USE_JIT_NORM / SGLANG_OPT_USE_TILELANG_MHC_*）
2. 对照 `/workspace/dsv4_ops_unit_tests/README.md` 的"真实形状验证"表
3. 确认 dtype（gfx936: is_fp8_fnuz()=False → e4m3，非 e5m2fnuz）

**产出**：签名+shape+graph-unsafe 操作清单+patch 目标函数路径。

## Phase 1: Layer 1 HIP kernel（功能正确性）

1. 在 `dsv4_all_hip_kernels.hip` 写/改 kernel
2. 所有 launch wrapper 带 `hipStream_t st`（graph-safe）
3. 用 raw ctypes 单独测正确性（`verify_kernels_v2.py` 模式）
4. 测 graph capture/replay（`test_graph_safe.py` 模式）
5. **门槛**：bit-exact 或量化噪声内 + graph 15/15

## Phase 2: Layer 2 hip_wrapper.py（sglang 对齐）

1. 按 Phase 0 调研的签名写 wrapper，**逐字对齐**
2. 返回新 tensor 的 → 用 `_buf(name="xxx")` 静态池（name tag 防 alias）
3. in-place / out-param 的 → 不用 _buf（caller 预分配）
4. 把 Python 层的多余操作移入 kernel（如 fused_rope 读 complex 不 copy、silu split kernel 不切片）
5. **门槛**：`tests/test_wrapper_engine_convention.py` 全 pass（严格模拟引擎调用约定+实际 shape）

## Phase 3: Layer 3 引擎集成（hip_dsv4_integration.py）

1. 按 Phase 0 的 patch 目标挂 wrapper（不是自己猜的目标）
2. 用 `sys.meta_path` finder 同步 patch（fork TP worker 前）
3. 确认 env 开关让 patch 被调用（如 rmsnorm 需 SGLANG_OPT_USE_JIT_NORM=1）
4. **门槛**：启动 server，patch log 全显示 "patched xxx"，server 不 crash

## Phase 4: 性能评测

1. **单 kernel**：`bench_wrapper_perf.py`（wrapper 层 vs SOTA，实际 shape）
2. **隔离 op-chain**：`e2e_op_chain.py`（MLP step）
3. **端到端**：`bench_server.py`（8-GPU A/B，同 config）
4. **profiling**：在 sglang 推理时 profile，确认单 kernel 加速比真实（见 SKILLS.md "Profiling"）
5. **门槛**：单 kernel ≥1x vs SOTA；若端到端无加速，确认瓶颈在 GEMM/attention（不是 elementwise）

## Phase 5: 文档 + push

1. 更新 REPORT.md / PERF_ANALYSIS.md / README.md
2. commit + push 到 `hip-kernels-v2` branch
3. 更新 AGENTS.md 的"已知调用路径"表和"陷阱"列表
