# DeepSeek V4 Flash 推理 — 逐算子实际使用分析

> 基于 SGLang 0.5.10rc0+das 源码 + 实际运行时 env 配置（SGLANG_USE_LIGHTOP=1, SGLANG_ROCM_USE_AITER_MOE=0, SGLANG_USE_FP8_W8A8_MOE=0, SGLANG_GROUPGEMM=true）。
> 交叉验证：源码 grep + 实际运行 log + model_config + dsv4_ops_unit_tests。

## 一、推理 forward 逐算子分析

每个 Transformer 层的 forward 路径，**标注实际使用的实现**：

### 1.1 Attention 路径

| 步骤 | 算子 | 实际实现 | DSL/库 | 文件位置 | 备注 |
|------|------|---------|--------|---------|------|
| 1. Q/KV 投影 | `fused_qkv_a_proj_with_mqa` | slimquant `apply()` → `triton_scaled_mm` | Triton→lightop | deepseek_v2.py:2211 | W8A8 dense GEMM |
| 2. Q 吸收 | `q_b_proj` | slimquant `apply()` | Triton→lightop | deepseek_v2.py:2298 | W8A8 |
| 3. KV 量化 | `per_token_group_quant_mla_deep_gemm_masked_fp8` | lmslim Triton | Triton | deepseek_v2.py:116 | FP8（非W8A8） |
| 4. RoPE | `fused_rope` | sglang jit_kernel | TileLang | deepseek_v4.py:359 | BF16 |
| 5. MLA decode | `flash_mla_with_kvcache_q_nope_pe` | flash_mla 1.2.0+das | HIP/ASM | deepseek_v4_backend_radix.py | BF16 |
| 6. O 投影 | `o_proj` | slimquant `apply()` | Triton→lightop | deepseek_v2.py:2672 | W8A8 |

### 1.2 MHC (Multi-Head Compressed) 路径

| 步骤 | 算子 | 实际实现 | DSL/库 | 文件位置 | 备注 |
|------|------|---------|--------|---------|------|
| 7. MHC pre | `mhc_pre` → `mhc_pre_big_fuse_tilelang` | **TileLang** | TileLang | mhc.py:244 | hc_mult=4, BF16 |
| 8. Sinkhorn | `hc_split_sinkhorn` → `hc_split_sinkhorn_kernel` | **TileLang** | TileLang | mhc.py:27 | 20 iters |
| 9. MHC post | `mhc_post` → `mhc_post_tilelang` | **TileLang** | TileLang | mhc.py:693 | BF16 |
| 10. MHC GEMM | `mhc_pre_gemm_sqrsum_tilelang` | **TileLang** | TileLang | mhc.py:367 | GEMM+sqrsum 融合 |
| 11. MHC split-K | `mhc_pre_gemm_sqrsum_splitk_stage_0/1` | **TileLang** | TileLang | mhc.py:454,522 | split-K 优化 |

**MHC 全部用 TileLang**，没有 Triton 或 C++ fallback（有 `mhc_pre_torch` 但默认不启用）。

### 1.3 MoE / FFN 路径

| 步骤 | 算子 | 实际实现 | DSL/库 | 文件位置 | 备注 |
|------|------|---------|--------|---------|------|
| 12. RMSNorm+残差 | `fused_add_rms_norm` | lightop | C++/HIP | deepseek_v2.py:2232 | BF16 |
| 13. 量化 | `per_token_quant_int8` | lmslim | **Triton** | slimquant_w4a8.py:174 | W8A8 量化 ① |
| 14. gate_up GEMM | `triton_scaled_mm` → `gemm_w8a8_smooth` | lmslim→lightop | Triton→**ASM** | slimquant_w4a8.py:216 | W8A8 dense GEMM ② |
| 15. MoE gate | `F.linear` + `sigmoid` + `topk` | PyTorch | PyTorch | deepseek_v2.py:664 | BF16 |
| 16. MoE align | `moe_align_block_size` | lightop | C++/HIP | moe_align_block_size.py | INT32 |
| 17. MoE GEMM (decode) | `moe_gemm_marlin_w8a8` | lightop | **HIP ASM** | slimquant_w4a8.py MoE | W8A8 ③ |
| 18. MoE GEMM (prefill) | `m_grouped_w8a8_gemm_nt_masked` | deepgemm | **HIP** | slimquant_w4a8.py MoE | W8A8 ④ |
| 19. SiLU+Mul | `torch.sigmoid(gate)*gate*up` | PyTorch | **PyTorch** | deepseek_v2.py | BF16 |
| 20. SiLU+Mul+quant (EP) | `silu_and_mul_masked_post_quant` | sglang jit_kernel | **TileLang** | deepseek_v2.py:628 | 仅 EP 路径 |
| 21. 量化 (act) | `per_token_quant_int8` | lmslim | **Triton** | slimquant_w4a8.py:174 | W8A8 量化 ①(再次) |
| 22. down GEMM | `moe_gemm_marlin_w8a8` / `triton_scaled_mm` | lightop/deepgemm | **HIP ASM** | slimquant_w4a8.py | W8A8 ③(down) |
| 23. MoE sum | `moe_sum` | lightop | C++/HIP | moe | BF16 |
| 24. shared expert | `shared_experts.gate_up_proj` | slimquant `apply()` | Triton→lightop | deepseek_v2.py | W8A8 |

### 1.4 其他路径

| 步骤 | 算子 | 实际实现 | DSL/库 | 文件位置 |
|------|------|---------|--------|---------|
| 25. RMSNorm (fallback) | `_rms_normalize_kernel` | **Triton** | Triton | deepseek_v4.py:118 |
| 26. linear_bf16_fp32 | `linear_bf16_fp32` | sglang jit_kernel | **TileLang** | deepseek_v4.py:949 |
| 27. topk_transform | `topk_transform_512` | sglang jit_kernel | **TileLang** | deepseek_v4.py:302 |
| 28. NSA quant | `act_quant_kernel` | **TileLang** | TileLang | nsa/tilelang_kernel.py:57 |
| 29. NSA index | `fp8_index_kernel` | **TileLang** | TileLang | nsa/tilelang_kernel.py:139 |
| 30. SWA index | `tilelang_make_swa_prefill_indices` | **TileLang** | TileLang | deepseek_v4.py:556 |

## 二、Triton 算子清单（单独挑出）

| # | 算子名 | 文件 | 功能 | W8A8? | 实际使用? |
|---|--------|------|------|-------|----------|
| 1 | `_per_token_quant_int8` | lmslim/int8_utils.py | per-token INT8 量化 | ✅ | ✅ 每次 forward |
| 2 | `_per_token_group_quant_int8` | lmslim/int8_utils.py | group INT8 量化 | ✅ | ✅ |
| 3 | `_rms_normalize_kernel` | deepseek_v4.py:118 | RMSNorm fallback | ❌ | ⚠️ 仅 fallback |
| 4 | `triton_scaled_mm` → `lightop_channel_int8_mm` | lmslim/quant_ops.py | W8A8 dense GEMM | ✅ | ✅ gate_up/o_proj |
| 5 | `fused_moe_kernel` | fused_moe_triton_kernels.py:336 | MoE GEMM (INT8/FP8) | ✅ | ⚠️ 策略1时用 |
| 6 | `write_zeros_to_output` | fused_moe_triton_kernels.py:84 | 零输出 | ✅ | EP 路径 |
| 7 | `fused_moe_kernel_gptq_awq` | fused_moe_triton_kernels.py:104 | GPTQ MoE | ❌ | ❌ 不用 |
| 8 | `moe_align_block_size` (Triton版) | moe_align_block_size.py | MoE 排序 | ❌ | ⚠️ lightop版优先 |
| 9 | `merge_attn_states` | merge_state.py:8 | attention state 合并 | ❌ | ✅ |
| 10 | `init_compressed_metadata` | compressed_metadata.py:11 | 压缩 metadata | ❌ | ✅ |
| 11 | `rocm_mla_decode_rope` | rocm_mla_decode_rope.py:38 | MLA decode+RoPE | ❌ | ✅ ROCm |
| 12 | `rocm_mla_decode_rope_v2` | rocm_mla_decode_rope.py:44 | MLA decode+RoPE v2 | ❌ | ⚠️ |
| 13 | `extend_attention` | triton_ops/extend_attention.py | prefill attention | ❌ | ⚠️ fallback |
| 14 | `decode_attention` | triton_ops/decode_attention.py | decode attention | ❌ | ⚠️ fallback |
| 15 | `apply_rotary_emb_triton` | deepseek_v4_rope.py | RoPE (Triton版) | ❌ | ⚠️ TileLang版优先 |
| 16 | `sglang_per_token_group_quant_fp8` | fp8_kernel.py | FP8 group 量化 | ❌ | FP8路径 |
| 17-20 | `w8a8_int8_tools.py:116,209,302,388` | lmslim | W8A8 GEMM 变体 | ✅ | ⚠️ config选择 |
| 21-23 | `block_int8_tools.py:108,217,325` | lmslim | block INT8 GEMM | ✅ | ⚠️ |
| 24 | `fused_llama_mlp.py:131` | lmslim | 融合 MLP | ❌ | ❌ |
| 25-26 | `awq_triton.py:8,113` | lmslim | AWQ 量化 | ❌ | ❌ |
| 27 | `fused_moe_tools_w4a8.py:59` | lmslim | W4A8 MoE | ❌ | ❌ |

**实际 forward 中使用的 Triton 算子**：#1, #2, #4, #9, #10, #11（6 个）。

## 三、TileLang 算子清单（单独挑出）

| # | 算子名 | 文件 | 功能 | 实际使用? |
|---|--------|------|------|----------|
| 1 | `hc_split_sinkhorn_kernel` | mhc.py:27 | Sinkhorn 归一化 | ✅ MHC |
| 2 | `mhc_pre_big_fuse_tilelang` | mhc.py:244 | MHC pre 融合 | ✅ MHC |
| 3 | `mhc_pre_gemm_sqrsum_tilelang` | mhc.py:367 | MHC GEMM+sqrsum | ✅ MHC |
| 4 | `mhc_pre_gemm_sqrsum_splitk_stage_0` | mhc.py:454 | MHC split-K s0 | ✅ MHC |
| 5 | `mhc_pre_gemm_sqrsum_splitk_stage_1` | mhc.py:522 | MHC split-K s1 | ✅ MHC |
| 6 | `mhc_post_tilelang` | mhc.py:693 | MHC post | ✅ MHC |
| 7 | `fused_rope` | jit_kernel/deepseek_v4.py:542 | 融合 RoPE | ✅ |
| 8 | `topk_transform_512` | jit_kernel/deepseek_v4.py:302 | TopK 512 | ✅ |
| 9 | `topk_transform_512_v2` | jit_kernel/deepseek_v4.py:331 | TopK 512 v2 | ⚠️ |
| 10 | `silu_and_mul_clamp` | jit_kernel/deepseek_v4.py:774 | SiLU+Mul+Clamp | ⚠️ |
| 11 | `silu_and_mul_masked_post_quant` | jit_kernel/deepseek_v4.py:792 | SiLU+Mul+masked+quant | ✅ EP路径 |
| 12 | `silu_and_mul_contig_post_quant` | jit_kernel/deepseek_v4.py:826 | SiLU+Mul+contig+quant | ⚠️ |
| 13 | `rmsnorm_self` | jit_kernel/deepseek_v4.py:882 | RMSNorm | ⚠️ fallback |
| 14 | `linear_bf16_fp32` | jit_kernel/deepseek_v4.py:949 | BF16→FP32 linear | ✅ |
| 15 | `tilelang_make_swa_prefill_indices` | jit_kernel/deepseek_v4.py:556 | SWA prefill index | ✅ |
| 16 | `act_quant_kernel` | nsa/tilelang_kernel.py:57 | NSA 激活量化 | ✅ NSA |
| 17 | `fp8_index_kernel` | nsa/tilelang_kernel.py:139 | NSA FP8 index | ✅ NSA |
| 18 | RoPE (TileLang版) | deepseek_v4_rope.py | RoPE | ✅ |

**实际 forward 中使用的 TileLang 算子**：#1-6 (MHC), #7 (RoPE), #8 (TopK), #11 (EP SiLU), #14 (linear), #15 (SWA), #16-17 (NSA) = **12 个**。

## 四、非 DSL 算子（HIP/C++/ASM/PyTorch）

| # | 算子 | 库 | 功能 | W8A8? |
|---|------|-----|------|-------|
| 1 | `fused_add_rms_norm` | lightop C++ | 残差+RMSNorm | ❌ |
| 2 | `gemm_w8a8_smooth` | lightop ASM | W8A8 dense GEMM | ✅ |
| 3 | `moe_gemm_marlin_w8a8` | lightop ASM | W8A8 MoE decode | ✅ |
| 4 | `m_grouped_w8a8_gemm_nt_masked` | deepgemm HIP | W8A8 MoE prefill | ✅ |
| 5 | `moe_w8a8_i8_marlin_prefill_down` | deepgemm HIP | W8A8 Marlin down | ✅ |
| 6 | `flash_mla_with_kvcache_q_nope_pe` | flash_mla HIP/ASM | MLA decode | ❌ |
| 7 | `moe_fused_gate` | lightop C++ | MoE gate | ❌ |
| 8 | `moe_align_block_size` | lightop C++ | MoE 排序 | ❌ |
| 9 | `moe_sum` | lightop C++ | MoE reduce | ❌ |
| 10 | `torch.sigmoid*gate*up` | PyTorch | SiLU+Mul | ❌ |
| 11 | `F.linear` | PyTorch | MoE gate projection | ❌ |

## 五、交叉验证

### 5.1 运行时 log 验证
服务器启动 log 中出现：
- `[lightop] hipModuleLoad: gemm_w8a8_smooth_*.co` → 确认 W8A8 GEMM 用 lightop ASM
- `[HIP-GRAPH-SAFE] Patched` → 确认 per_token_quant 被 patch
- `Capture cuda graph` → 确认 CUDA graph 捕获

### 5.2 dsv4_ops_unit_tests 验证
测试集覆盖的算子与实际使用一致：
- `TestPerTokenQuantInt8` → 对应 #1 per_token_quant_int8 ✅
- `TestGroupedGemmNtW8A8Masked` → 对应 #4 m_grouped_w8a8_gemm ✅
- `TestMoEW8A8Marlin` → 对应 #3 moe_gemm_marlin_w8a8 ✅
- `TestSiluAndMul` → 对应 #10 PyTorch silu ✅
- `TestRmsnorm` → 对应 #13 rmsnorm_self ✅
- `TestFlashMlaWithKvCacheQNopePe` → 对应 #6 flash_mla ✅
- `TestFusedRope` → 对应 #7 fused_rope ✅

### 5.3 env 配置验证
实际启动 env（来自文档）：
- `SGLANG_USE_LIGHTOP=1` → 启用 lightop 融合算子
- `SGLANG_ROCM_USE_AITER_MOE=0` → 不用 aiter MoE
- `SGLANG_USE_FP8_W8A8_MOE=0` → 不用 FP8 W8A8 MoE
- `SGLANG_GROUPGEMM=true` → 启用 grouped GEMM
- `SGLANG_OPT_USE_FUSED_HASH_TOPK=true` → 启用融合 TopK
- `SGLANG_USE_FUSED_SILU_MUL_QUANT` → 未设置（默认 false），所以 SiLU+quant 分开做
