// k09_merge_attn_states.hip - merge_attn_states (Triton, LSE merge)

#include "common.hip"

// ============================================================
// 9. merge_attn_states (Triton)
//   prefix_out/suffix_out:[tokens, heads, head_size] bf16
//   prefix_lse/suffix_lse:[heads, tokens] fp32  (head-major)
//   out:[tokens,heads,head_size] bf16, out_lse:[heads,tokens] fp32
//   alpha = exp(lse - lse_max); out = (p_out*p_se + s_out*s_se)/(p_se+s_se)
//   new_lse = log(p_se+s_se) + lse_max ; FA2: +inf lse treated as -inf.
// ============================================================
template<int BS>
__global__ void merge_attn_kernel(
    __hip_bfloat16* out, float* out_lse,
    const __hip_bfloat16* prefix_out, const float* prefix_lse,
    const __hip_bfloat16* suffix_out, const float* suffix_lse,
    int num_tokens, int num_heads, int head_size
) {
    int tok = blockIdx.x;
    int hd  = blockIdx.y;
    if (tok >= num_tokens || hd >= num_heads) return;
    int tid = threadIdx.x;

    float p_lse = prefix_lse[hd * num_tokens + tok];
    float s_lse = suffix_lse[hd * num_tokens + tok];
    if (isinf(p_lse) && p_lse > 0) p_lse = -INFINITY;
    if (isinf(s_lse) && s_lse > 0) s_lse = -INFINITY;
    float lse_max = fmaxf(p_lse, s_lse);
    float p_se = isinf(p_lse) ? 0.f : expf(p_lse - lse_max);
    float s_se = isinf(s_lse) ? 0.f : expf(s_lse - lse_max);
    float se_sum = p_se + s_se;
    float p_scale = p_se / se_sum;
    float s_scale = s_se / se_sum;
    if (tid == 0 && out_lse) {
        out_lse[hd * num_tokens + tok] = logf(se_sum) + lse_max;
    }
    int64_t off = (int64_t)tok * num_heads * head_size + (int64_t)hd * head_size;
    for (int i = tid; i < head_size; i += BS) {
        float pv = b2f(prefix_out[off + i]);
        float sv = b2f(suffix_out[off + i]);
        out[off + i] = f2b(pv * p_scale + sv * s_scale);
    }
}

extern "C" {
void launch_merge_attn_states(void* out,void* out_lse,
    const void* prefix_out,const void* prefix_lse,
    const void* suffix_out,const void* suffix_lse,
    int num_tokens,int num_heads,int head_size,hipStream_t st){
    constexpr int BS=128;
    dim3 grid(num_tokens,num_heads); dim3 block(BS);
    merge_attn_kernel<BS><<<grid,block,0,st>>>((__hip_bfloat16*)out,(float*)out_lse,
        (const __hip_bfloat16*)prefix_out,(const float*)prefix_lse,
        (const __hip_bfloat16*)suffix_out,(const float*)suffix_lse,
        num_tokens,num_heads,head_size);}
}
