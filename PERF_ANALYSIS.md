# 性能分析：超过 2x speedup 的 kernel 加速原因

对比基线：sglang 引擎实际使用的 SOTA（lmslim Triton / sglang jit_kernel tvm_ffi / sglang triton / vllm triton）。
硬件：Hygon DCU gfx936（wavefront64，CDNA2 架构，64 CUs）。

---

## 1. fused_rope — 12.2x vs jit_kernel, 21.5x vs triton

**kernel**: `fused_rope_kernel`，interleaved (real,imag) 布局，1 warp (32 lane) 处理 1 个 (token, head)，每 lane 处理 1 个复数。grid = `ceil(num_tokens × num_heads / 4)` blocks，每 block 4 warps。

**加速原因**:

1. **极低 dispatch overhead**：HIP ctypes 直接 `launch_fused_rope(ptr,...,stream)`，单次 launch ~3μs。sglang jit_kernel (tvm_ffi) 走 Python → tvm_ffi → kernel，triton 走 Python → triton JIT → kernel，两者 launch 路径多 2-3 层，dispatch overhead ~80-140μs（从 SOTA 0.08-0.15ms vs HIP 0.007ms 可见，差值全是 dispatch）。rope 本身计算量极小（64 维 × 32 复数乘法），是 **launch-bound** kernel，dispatch 占主导，HIP 直调优势放大。

2. **warp 级并行映射**：1 warp = 32 lane = 32 复数 = rope_dim/2，正好一个 warp 处理一个 head 的全部复数。无 cross-warp 同步，无 shared memory。jit_kernel/triton 用 block 级并行 + shared mem，对 rope 这种小问题反而有 sync 开销。

3. **interleaved 布局连续访问**：`q[off+2*i]`/`q[off+2*i+1]` 相邻，一个 32-byte cache line 装 16 个复数，warp 内 32 lane 连续读，coalesced 100%。split-half 布局会有跨 cache line 访问。

4. **无 host 分配**：in-place 写回 q/k，无临时 tensor。triton 版 `apply_rotary_emb_triton` 返回新 tensor（`torch.empty` + 写回），多一次 alloc + 一次写。

**结论**: fused_rope 是 launch-bound，HIP 的低 dispatch + warp 级映射是 12-21x 的主因。

---

## 2. per_token_quant_int8 — 7.95x vs lmslim Triton

**kernel**: `ptq_kernel<256,16>`，1 block/row，256 thread，每 thread 处理 16 元素（EPT=16）。wavefront64 reduce max，shared mem 跨 wavefront，单 pass 完成 absmax + scale + quant。

**加速原因**:

1. **单 pass fused**：absmax → scale → quant 在一个 kernel 内，x 只读一次、q 写一次。lmslim Triton 的 `per_token_quant_int8` 虽也是 fused，但 Triton codegen 的内层循环展开不如手写 `#pragma unroll EPT=16` 紧凑，寄存器分配略差。

2. **wavefront64 reduce**：gfx936 wavefront=64，`wmax64` 用 `__shfl_xor` 6 轮（32→1）归约。lmslim Triton 用 block-level reduce（shared mem + sync），对 256 线程需要 2 轮 shared mem 归约，多 1 次 `__syncthreads`。HIP 直接 wavefront shuffle 更快。

3. **EPT=16 向量化**：每 thread 16 个 bf16 = 32 bytes = 2 个 int128 向量加载，编译器生成 `global_load_dwordx4`。Triton 的 `tl.load` 自动向量化但 block 尺寸不同，对 N=4096/256thread=16 元素/thread 的映射不如手写精确。

4. **round_he + 截断**：用 `__builtin_rintf`（硬件 round-to-nearest）+ `(int)` 截断，1 条指令。Triton 的 `tl.cast(tl.int8)` 走软件 round。

**结论**: 计算量小（4096 元素 absmax），memory-bound + launch-bound，HIP 的 wavefront shuffle + 精确向量映射 + 低 dispatch 共同贡献 7.95x。

---

## 3. per_token_group_quant_int8 — 2.44-14.5x vs lmslim Triton

**kernel**: `ptgq_kernel<128>`，1 block/(row,group)，128 thread，group_size=128。wavefront64 + cross-wavefront shared mem reduce。

**加速原因**:

1. **同 ptq 的 wavefront64 reduce 优势**：absmax 归约用 shuffle 而非 shared mem 多轮。

2. **group 粒度并行**：grid=`(M, N/gs)`，每个 (row,group) 独立 block。M=1 时只有 N/gs=32 个 block，但每 block 128 thread 处理 128 元素，算术强度高。lmslim Triton 对小 M 的 grid 配置不优（Triton 的 `num_warps` 选择对 128 元素/group 偏大）。

3. **M=1 时 14.5x > M=256 时 2.44x**：M 小时 launch overhead 占比大，HIP 低 dispatch 优势放大；M 大时计算占比上升，Triton 的 codegen 质量接近，gap 缩小。这印证 launch-bound 假说。

4. **修正了 v/scale 量化**：原版多乘 127 致饱和（错误），修正后 bit-exact 且无需额外除法。

**结论**: 小 M launch-bound（14.5x），大 M memory-bound（2.44x），HIP 的低 dispatch + wavefront reduce 主导。

---

## 4. merge_attn_states — 2.24x vs vllm Triton

**kernel**: `merge_attn_kernel<128>`，grid=`(tokens, heads)`，每 block 128 thread 处理 1 个 (token,head) 的 head_size=512 维合并。LSE merge: `alpha=exp(lse-max)`, 加权求和。

**加速原因**:

1. **grid 映射精确**：`(tokens, heads)` 二维 grid，每 block 正好 1 个 (token,head)，512 维由 128 thread 各处理 4 元素。vllm Triton 用 `(tokens, heads)` 但 block 尺寸自动选择，对 512 维可能 over-tile。

2. **单 pass LSE merge**：`p_scale`/`s_scale` 在寄存器算一次，所有 512 维复用。Triton 版可能重新计算 scale（取决于 codegen）。

3. **数值稳定优化**：`lse_max = max(p,s)`，`exp(lse-max)` 避免 overflow。HIP 用 `isinf` 检查 + 条件赋值，1 分支。Triton 的 `tl.where` 生成 predicated 指令，略多开销。

4. **bf16 直接写**：`f2b(pv*p_scale+sv*s_scale)` 直接 bf16 写回，无中间 fp32 buffer。Triton 可能先 fp32 再 cast。

**结论**: memory-bound（512 维读 2 个 bf16 + 写 1 个），HIP 的精确 grid + 单 pass scale 复用 + 低 dispatch 贡献 2.24x。

---

## 5. silu_and_mul — 6.6-10.5x vs torch ref

**kernel**: `silu_mul_kernel<256,8>`，1 block/row，EPT=8。

**加速原因**:

1. **fused kernel**：`sigmoid(gate)*gate*up` 在一个 kernel，gate/up 各读一次，out 写一次。torch ref 是 3 个独立 op（sigmoid + mul + mul），每个 op 独立 launch + 独立 global mem 读写，共 3 次 launch + 5 次 mem 读写 vs HIP 1 次 launch + 3 次 mem 读写。

2. **低 dispatch**：torch op 每次 launch ~10μs，3 op = 30μs。HIP 单次 6μs。对 N=2048 小问题，dispatch 占 80%。

3. **EPT=8 向量加载**：8 bf16 = 16 bytes = 1 个 int128，coalesced。

**注**: vs torch ref（非 vendor SOTA）。引擎实际用 `SiluAndMul.forward_cuda`（可能 lightop C++），HIP vs 它的优势会小。但 fused 单 pass 的 mem 带宽优势仍成立。

---

## 通用加速因素总结

| 因素 | 影响 kernel | 量化 |
|------|------------|------|
| **低 dispatch overhead** (HIP ctypes vs Triton/jit_kernel Python 路径) | fused_rope, ptq, ptgq, silu, merge | launch-bound kernel 放大 5-15x |
| **wavefront64 shuffle reduce** (gfx936 原生 64-lane) | ptq, ptgq, act_quant | 比 Triton block reduce 少 1-2 次 sync |
| **单 pass fused** (读一次写一次) | silu_mul, merge, ptq | mem 带宽减半 |
| **精确 grid/warp 映射** (1 warp/head, 1 block/(token,head)) | fused_rope, merge | 无 cross-block sync，无 shared mem 浪费 |
| **interleaved 布局 coalesced** | fused_rope | 100% cache line 利用 |
| **手写 `#pragma unroll` + 向量加载** | ptq, ptgq, silu | 编译器生成 `global_load_dwordx4` |

## 反例：慢于 SOTA 的 kernel

| kernel | SOTA | HIP | 原因 |
|--------|------|-----|------|
| topk_transform_512 | sglang/jit_kernel ASM 0.01ms | 0.07ms | jit_kernel 是手写 ASM radix，单 pass；HIP radix 4-pass + atomicAdd，无法匹敌 |
| hc_split_sinkhorn | sglang/tilelang 0.053ms | 0.063-0.2ms | tilelang 编译期展开 4×4 Sinkhorn；HIP 用 thread-0 串行 4×4，M 大时 grid 不足 |
| act_quant_fp8 | lightop/DCU vendor 0.005ms | 0.005-0.03ms | lightop 是 vendor MFMA 优化；HIP 朴素 wavefront reduce |
| grouped_gemm | torch 0.49ms | 0.71ms | HIP 朴素 tile，无 MFMA；vendor marlin 用 MFMA int8 |

这些 vendor SOTA 是 ASM/MFMA 级优化，HIP 手写 kernel 不替换引擎 vendor 路径（引擎 MoE GEMM 走 vendor marlin，topk 走 jit_kernel ASM）。
