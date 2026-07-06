// launchers.hip - single compile entry: includes common + all k*.hip files.
// extern "C" wrappers live in each k*.hip (each in its own extern "C" block).

#include "common.hip"
#include "k01_per_token_quant.hip"
#include "k02_per_token_group_quant.hip"
#include "k03_rmsnorm.hip"
#include "k04_fused_rope.hip"
#include "k05_silu_mul.hip"
#include "k06_silu_mul_masked_quant.hip"
#include "k07_hc_split_sinkhorn.hip"
#include "k08_act_quant_fp8.hip"
#include "k09_merge_attn_states.hip"
#include "k10_topk_transform.hip"
#include "k11_mhc_pre.hip"
#include "k12_mhc_post.hip"
#include "k13_swa_prefill_indices.hip"
#include "k14_grouped_gemm.hip"
