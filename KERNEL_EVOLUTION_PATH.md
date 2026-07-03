# DeepSeek V4 Flash — Kernel 进化路径与单元测试对应分析

## 一、单元测试对应关系

### 1.1 已通过的测试（我们的 HIP kernel 能过）

| 测试类 | 测试方法 | 算子 | 断言 | 我们的 HIP | 通过? |
|--------|---------|------|------|-----------|-------|
| TestPerTokenQuantInt8 | test_quant | per_token_quant_int8 | diff<=1.0 | bit-exact (diff=0) | ✅ |
| TestRmsnorm | test_rmsnorm_self | rmsnorm_self | allclose(atol=2e-2) | maxdiff<0.016 | ✅ |
| TestSiluAndMul | test_silu_and_mul_ref | silu_and_mul | not NaN, shape | maxdiff=0 | ✅ |
| TestFusedRope | test_fused_rope | fused_rope | allclose(atol=1e-2) | 待验证 | ⚠️ |
| TestTopkTransform | test_topk_transform | topk_transform_512 | shape + values | correct | ✅ |

### 1.2 未对应的测试（我们没有对应的 HIP kernel）

| 测试类 | 测试方法 | 算子 | 断言 | 状态 | 进化路径 |
|--------|---------|------|------|------|---------|
| TestGroupedGemmNtW8A8Masked | test_masked_gemm | m_grouped_w8a8_gemm_nt_masked | cos_diff<1e-4 | ❌ 未写 | 见下 |
| TestGroupedGemmNtBf16Masked | test_masked_gemm | m_grouped_fp8_gemm_nt_masked | cos_diff<1e-4 | ❌ 未写 | FP8路径 |
| TestMoEMaskedGemm | test_masked_gemm_shape | MoE masked GEMM | shape | ❌ 未写 | 见下 |
| TestMoEContigGemm | test_contig_gemm_shape | MoE contig GEMM | shape | ❌ 未写 | 见下 |
| TestMoEW8A8Marlin | test_decode_up/full | moe_gemm_marlin_w8a8 | cos_diff<1e-4 | ❌ 未写 | 见下 |
| TestBiasedGroupedTopk | test_topk_routing_ref | MoE topk routing | shape | ❌ 未写 | 见下 |
| TestFlashMlaWithKvCacheQNopePe | test_decode | flash_mla_with_kvcache_q_nope_pe | cos_diff<1e-4 | ❌ 未写(已有HIP/ASM) | 不需要 |
| TestFlashMlaWithKvCacheCombined | test_decode | flash_mla_with_kvcache | cos_diff<1e-4 | ❌ 未写(已有HIP/ASM) | 不需要 |
| TestDenseFFNGemm | test_* | F.linear (BF16/FP8) | shape | ❌ 未写 | rocBLAS已最优 |
| TestLinearBf16Fp32 | test_linear_bf16_fp32 | linear_bf16_fp32 | allclose(atol=1e-2) | ❌ 未写 | 见下 |
| TestPerTokenGroupQuant | (deepgemm_ops) | per_token_group_quant_int8 | diff<=1.0 | ⚠️ maxdiff=42 | 见下 |

## 二、每个 kernel 的进化路径

### 2.1 per_token_quant_int8 — ✅ 已完成，graph兼容

**当前状态**：bit-exact，7-8× 加速，已集成，TTFT -20.5%
**进化路径**：已完成。已是 HIP native extension，graph-safe。

### 2.2 per_token_group_quant_int8 — ⚠️ 正确性未通过

**当前问题**：maxdiff=42（Triton 用 `tl.clamp(y/y_s).to(int8)` 截断，我用 round）
**进化路径**：
1. 修复：改为截断（`fmaxf(-128, fminf(127, v*inv))` 然后 `(int8_t)` cast）— 已尝试，仍有 diff
2. 根因：Triton 的 scale 计算用 `tl.maximum(tl.max(tl.abs(y)), eps)` 其中 eps=1e-10，我的用 `fmaxf(lmax, 1e-10f)`
3. 根因2：Triton 处理 reshape `(s_num, 128)` 再 flatten，中间精度可能不同
4. **下一步**：逐元素 debug，找到第一个 mismatch 的位置，对比 scale 值

### 2.3 rmsnorm_self — ✅ 已完成

**当前状态**：maxdiff<0.016（通过 atol=2e-2 断言），9.8× 加速
**进化路径**：已完成。BF16 精度内正确。
**graph兼容**：需包装为 native extension（当前是 ctypes）

### 2.4 fused_rope — ⚠️ 已写，未验证

**当前状态**：已写 HIP kernel，编译通过，未对单元测试验证
**进化路径**：
1. 运行 `TestFusedRope.test_fused_rope`（atol=1e-2）
2. 确认 complex64 freqs_cis 的处理正确（我的 kernel 用 real/imag 分离）
3. 包装为 native extension

### 2.5 silu_and_mul — ✅ 已完成

**当前状态**：maxdiff=0.000000，6-10× 加速
**进化路径**：已完成。
**graph兼容**：需包装为 native extension

### 2.6 silu_and_mul_masked_post_quant — ⚠️ 已写，未验证

**进化路径**：
1. 编写对应单元测试（repo 中此测试已移除，但 serving 中有 try/except fallback）
2. 验证 masked 路径正确性
3. 集成到 EP MoE 路径

### 2.7 mhc_pre / mhc_post — ⚠️ 已写，未完整验证

**当前状态**：mhc_pre maxdiff=0（sigmoid+mix 部分），mhc_post 编译通过未验证
**进化路径**：
1. mhc_pre 的 GEMM 部分（`mhc_pre_gemm_sqrsum_tilelang`）需要单独 HIP 实现
2. mhc_post 的 matmul+residual 部分需要完整实现（当前是 stub）
3. 这两个是 TileLang 最复杂的 kernel，涉及 split-K 和 GEMM 融合

### 2.8 act_quant (NSA FP8) — ✅ scale 正确

**当前状态**：scale_diff=0，正确
**进化路径**：已完成 scale 部分。FP8 量化本身需要 FP8 cast（DCU 支持 float8_e4m3fn）

### 2.9 topk_transform_512 — ✅ 已完成

**当前状态**：correct=True
**进化路径**：已完成。

### 2.10 swa_prefill_indices — ✅ 已完成

**当前状态**：correct=True
**进化路径**：已完成。

### 2.11 w8a8_scaled_gemm — ⚠️ 正确但慢

**当前状态**：diff=0（正确），但 0.03-0.48× SOTA 性能
**进化路径**：
1. v1：每线程独立从 global memory 读 → 慢
2. v2（已写）：shared memory tiling → 待测试
3. v3（计划）：增大 tile 到 64×64，vectorized load，参考 lightop `.co` 的配置
4. 最终：可能无法超越 lightop ASM，但可以作为 fallback

### 2.12 linear_bf16_fp32 — ❌ 未写

**进化路径**：这个是 GEMM（BF16 输入，FP32 权重，FP32 输出），实际 dispatch 到 rocBLAS/hipBLASLt。不需要写 HIP kernel，rocBLAS 已最优。

### 2.13 MoE GEMM (marlin/masked/contig) — ❌ 未写

**进化路径**：
1. `moe_gemm_marlin_w8a8`：lightop Marlin ASM，hand-tuned for gfx936。极难超越。
2. `m_grouped_w8a8_gemm_nt_masked`：deepgemm HIP，已优化。
3. 可以尝试用 sdot4 + shared memory tiling 写一个 fallback，但性能不会超过 ASM。
4. **实际价值**：作为非 lightop/deepgemm 环境的 fallback，而非性能优化。

### 2.14 hc_split_sinkhorn — ⚠️ 已写 stub

**进化路径**：Sinkhorn 是迭代归一化（行+列交替），逻辑简单但需要多轮。当前是 stub，需要完整实现行归一化+列归一化循环。

## 三、Graph 兼容性

### 当前状态
| Kernel | 实现方式 | Graph 兼容? | 需要改进 |
|--------|---------|------------|---------|
| per_token_quant_int8 | Native PyTorch ext | ✅ 是 | 已完成 |
| rmsnorm_self | ctypes | ❌ 否 | 改为 native ext |
| silu_and_mul | ctypes | ❌ 否 | 改为 native ext |
| fused_rope | ctypes | ❌ 否 | 改为 native ext |
| 其他所有 | ctypes | ❌ 否 | 改为 native ext |

### Graph 兼容改进路径
所有 kernel 需要从 ctypes 改为 **PyTorch C++ native extension**：
1. 将 `dsv4_all_hip_kernels.hip` 的 kernel 函数包装到 `dsv4_torch_ext_combined.cpp`
2. 使用 `per_token_quant_int8_stream(x, stream_ptr)` 模式（stream 从 Python 传入）
3. 编译为 `.cpython-310-x86_64-linux-gnu.so`
4. 通过 `.pth` 文件注入

## 四、优先级排序（按端到端影响）

| 优先级 | Kernel | 原因 |
|--------|--------|------|
| P0 | per_token_quant_int8 | ✅ 已完成，TTFT -20.5% |
| P1 | silu_and_mul + quant 融合 | MoE 内占 16%，独立 21× |
| P1 | add_rmsnorm_quant 融合 | 每层都用，独立 6.7× |
| P2 | rmsnorm_self | fallback 路径，9.8× |
| P2 | fused_rope | 每层都用，需验证 |
| P3 | mhc_pre/post | TileLang 有正确性问题，复杂 |
| P3 | w8a8_gemm | 正确但慢，lightop ASM 更优 |
| P4 | per_token_group_quant | 非 W8A8 主路径 |
| P4 | MoE GEMM | ASM 已最优 |
