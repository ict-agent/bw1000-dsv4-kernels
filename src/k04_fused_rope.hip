// k04_fused_rope.hip - fused_rope (DeepSeek V4 rotary, INTERLEAVED complex layout)

#include "common.hip"

// ============================================================
// 4. fused_rope (DeepSeek V4 rotary, INTERLEAVED complex layout)
//   q: [num_tokens, num_q_heads, rope_dim] bf16, rope_dim=64
//   k: [num_tokens, num_k_heads, rope_dim] bf16 (optional)
//   freqs_cis: [max_pos, rope_dim] fp32, interleaved (real,imag) pairs
//   positions: [num_tokens] int32
//   One warp (32 lanes) per (token, head); lane handles 1 complex (rope_dim/2=32).
// ============================================================
__global__ void fused_rope_kernel(
    __hip_bfloat16* q, __hip_bfloat16* k,
    const float* freqs_cis, const int* positions,
    int num_tokens, int num_q_heads, int num_k_heads, int rope_dim, int has_k
) {
    const int wpb = blockDim.x >> 5;            // warps per block
    const int gwp = (blockIdx.x * wpb) + (threadIdx.x >> 5); // global warp id
    const int lane = threadIdx.x & 31;
    const int total_heads = num_q_heads + (has_k ? num_k_heads : 0);
    if (total_heads == 0) return;
    const int tok = gwp / total_heads;
    const int h   = gwp % total_heads;
    if (tok >= num_tokens) return;
    const int n_pairs = rope_dim >> 1;
    if (lane >= n_pairs) return;

    bool is_q = (h < num_q_heads);
    int local_h = is_q ? h : (h - num_q_heads);
    __hip_bfloat16* dst = is_q ? q : k;
    if (!is_q && !has_k) return;

    int head_count = is_q ? num_q_heads : num_k_heads;
    int64_t off = (int64_t)tok * head_count * rope_dim + (int64_t)local_h * rope_dim;
    int pos = positions[tok];
    const float* freq = freqs_cis + (int64_t)pos * rope_dim + (lane << 1);
    float fr = freq[0], fi = freq[1];

    int idx = lane << 1;
    float xr = b2f(dst[off + idx]);
    float xi = b2f(dst[off + idx + 1]);
    float or_ = xr * fr - xi * fi;
    float oi_ = xr * fi + xi * fr;
    dst[off + idx]     = f2b(or_);
    dst[off + idx + 1] = f2b(oi_);
}

extern "C" {
void launch_fused_rope(void* q,void* k,const void* freqs_cis,const void* positions,
    int num_tokens,int num_q_heads,int num_k_heads,int rope_dim,int has_k,hipStream_t st){
    int total_heads = num_q_heads + (has_k ? num_k_heads : 0);
    int total_warps = num_tokens * total_heads;
    int warps_per_block = 4;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    int threads = warps_per_block * 32;
    fused_rope_kernel<<<blocks,threads,0,st>>>(
        (__hip_bfloat16*)q,(__hip_bfloat16*)k,(const float*)freqs_cis,(const int*)positions,
        num_tokens,num_q_heads,num_k_heads,rope_dim,has_k);}
}
