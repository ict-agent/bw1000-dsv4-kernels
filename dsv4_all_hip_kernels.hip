// dsv4_all_hip_kernels.hip - HIP C++ implementations of ALL Triton/TileLang kernels
// used in DeepSeek V4 Flash inference on Hygon DCU gfx936.
//
// All launch_* wrappers take a hipStream_t so they are CUDA-graph capturable:
// no host-side allocation, no host-side sync, pointer-driven.
//
// Kernels:
//  1. per_token_quant_int8         (Triton)        [bit-exact]
//  2. per_token_group_quant_int8   (Triton)        [cross-wavefront reduction fixed]
//  3. rmsnorm_self                 (TileLang)
//  4. fused_rope                   (jit_kernel rope.cuh, INTERLEAVED layout)
//  5. silu_and_mul                 (PyTorch)
//  6. silu_mul_masked_quant        (TileLang, EP MoE path)
//  7. hc_split_sinkhorn            (TileLang mhc.py)
//  8. act_quant_fp8                (TileLang NSA, fp8_e4m3 output)
//  9. merge_attn_states            (Triton, LSE merge)
// 10. topk_transform_512           (jit_kernel topk.cuh, page transform)
// 11. mhc_pre                      (TileLang, sigmoid+mix)
// 12. mhc_post                     (TileLang, c*d + einsum(a,b))
// 13. swa_prefill_indices          (TileLang, [num_q_tokens,128] indices)
// 14. grouped_gemm                 (MoE W8A8 int8 grouped GEMM, masked_m indexing)

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cfloat>

// ============================================================
// Common helpers
// ============================================================
__device__ __forceinline__ float b2f(const __hip_bfloat16 v){return __bfloat162float(v);}
__device__ __forceinline__ __hip_bfloat16 f2b(float v){return __float2bfloat16(v);}
__device__ __forceinline__ float wsum64(float v){for(int o=32;o>0;o>>=1)v+=__shfl_xor(v,o,64);return v;}
__device__ __forceinline__ float wmax64(float v){for(int o=32;o>0;o>>=1)v=fmaxf(v,__shfl_xor(v,o,64));return v;}
__device__ __forceinline__ float wsum32(float v){for(int o=16;o>0;o>>=1)v+=__shfl_xor(v,o,32);return v;}
__device__ __forceinline__ float wmax32(float v){for(int o=16;o>0;o>>=1)v=fmaxf(v,__shfl_xor(v,o,32));return v;}
__device__ __forceinline__ float round_he(float v){return __builtin_rintf(v);}

// FP8 e4m3fn encoding (max=448) with round-to-nearest-even, matches torch.float8_e4m3fn cast.
__device__ __forceinline__ uint8_t f32_to_e4m3(float f){
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t sign = (u >> 31) & 1;
    uint32_t abs_u = u & 0x7FFFFFFFu;          // |f| as float bits (sign cleared)
    if (abs_u == 0) return (uint8_t)(sign << 7);

    // Interpret |f|
    int exp_f = (int32_t)(abs_u >> 23) - 127;   // unbiased exp
    uint32_t mant_f = abs_u & 0x7FFFFFu;        // 23-bit fraction (without hidden 1)

    // e4m3: bias 7, normal exp biased in [1,14] OR 15 with mantissa<=6 (e4m3fn max=448).
    // value = (1.mmm) * 2^(e-7). exp=15,mant=7 reserved (NaN/inf). Subnormal when e4==0.
    int e4 = exp_f + 7;                          // tentative e4m3 biased exp

    // ---- true overflow (exp > 15, or exp==15 with mant that would round to 7) -> clamp 448 ----
    // handled below after mantissa rounding.

    uint32_t out;
    if (e4 >= 1) {
        // normal: keep 3 mantissa bits with round-to-nearest-even
        uint32_t m3 = mant_f >> 20;              // top 3 bits
        uint32_t round_bit = (mant_f >> 19) & 1;
        uint32_t sticky = (mant_f & ((1u << 19) - 1)) ? 1u : 0u;
        uint32_t roundup = round_bit & (sticky | (m3 & 1));
        m3 += roundup;
        if (m3 == 8) { m3 = 0; e4 += 1; }         // carried into exp
        // e4m3fn: exp=15 valid only for mantissa 0..6 (max 448). mant=7 -> clamp to 6.
        if (e4 > 15) return (uint8_t)((sign << 7) | 0x7E);
        if (e4 == 15 && m3 == 7) return (uint8_t)((sign << 7) | 0x7E);  // 0x7E = sign|111|110
        out = (sign << 7) | (e4 << 3) | m3;
    } else {
        // subnormal or underflow. e4m3 subnormal value = k * 2^-9, k in [1,7].
        // |f| = (1.mant_f) * 2^exp_f. Express as k * 2^-9:
        //   k = (1.mant_f) * 2^(exp_f + 9)
        // full fixed-point mantissa with hidden bit: M = (1<<23) | mant_f  (24-bit, value 1.f * 2^23)
        // k = M * 2^(exp_f + 9 - 23) = M * 2^(exp_f - 14)
        int shift = 14 - exp_f;                  // k = M >> shift (with rounding)
        uint32_t M = (1u << 23) | mant_f;
        if (shift >= 32) {
            // underflow -> 0
            out = (sign << 7);
        } else if (shift <= 0) {
            // k >= M (shouldn't happen for e4<1, but guard): normalizes up
            uint32_t k = M << (-shift);
            // k has hidden bit at position 3; extract m3
            uint32_t m3 = (k >> 3) & 0x7;
            out = (sign << 7) | m3;              // exp=0 subnormal
            (void)m3;
        } else {
            // round M to 3 fraction bits (4-bit result including hidden) at position shift
            uint32_t keep = M >> shift;          // 4-bit field: hidden at bit 3, m3 at bits 2:0
            uint32_t rem = M & ((1u << shift) - 1);
            uint32_t half = 1u << (shift - 1);
            if (rem > half || (rem == half && (keep & 1))) keep += 1;  // round-to-nearest-even
            if (keep >= 16) {
                // carried into normal range (hidden bit overflow -> exp becomes 1)
                uint32_t m3 = (keep >> 1) & 0x7;
                out = (sign << 7) | (1u << 3) | m3;
            } else {
                // subnormal: exp=0, mantissa = keep & 7 (hidden bit is the "8" which makes it subnormal k)
                uint32_t m3 = keep & 0x7;
                if (keep == 0) out = (sign << 7);   // flushed to 0
                else out = (sign << 7) | m3;
            }
        }
    }
    return (uint8_t)out;
}

// FP8 e5m2fnuz encoding (DCU gfx94x native): sign(1)+exp(5,bias=15)+mant(2).
// fnuz = flush subnormals to zero, no NaN/inf (all codes are numeric).
// Max representable = 1.75 * 2^(31-15) = 57344. Subnormals flushed to 0.
__device__ __forceinline__ uint8_t f32_to_e5m2fnuz(float f){
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t sign = (u >> 31) & 1;
    uint32_t abs_u = u & 0x7FFFFFFFu;
    if (abs_u == 0) return (uint8_t)(sign << 7);
    int32_t exp_f = (int32_t)(abs_u >> 23) - 127;
    uint32_t mant_f = abs_u & 0x7FFFFFu;
    // e5m2 normal range: exp_f in [-14, 15] (biased 1..31). Subnormal (exp_f < -14) -> 0.
    if (exp_f < -14) return (uint8_t)(sign << 7);   // flush to zero (fnuz)
    if (exp_f > 15) {
        // overflow -> max (sign|11111|11 = 0x7F/0xFF)
        return (uint8_t)((sign << 7) | 0x7F);
    }
    // normal: keep top 2 mantissa bits with round-to-nearest-even
    uint32_t m2 = mant_f >> 21;            // top 2 bits (bits 22:21)
    uint32_t round_bit = (mant_f >> 20) & 1;
    uint32_t sticky = (mant_f & ((1u << 20) - 1)) ? 1u : 0u;
    uint32_t roundup = round_bit & (sticky | (m2 & 1));
    m2 += roundup;
    uint32_t e5 = (uint32_t)(exp_f + 15);
    if (m2 == 8) { m2 = 0; e5 += 1; }       // mantissa carried
    if (e5 > 31) return (uint8_t)((sign << 7) | 0x7F);  // clamp to max
    return (uint8_t)((sign << 7) | (e5 << 2) | m2);
}
__device__ __forceinline__ float e4m3_to_f32(uint8_t b){
    uint32_t sign = (b >> 7) & 1;
    uint32_t e    = (b >> 3) & 0xF;
    uint32_t m    = b & 0x7;
    if (e == 0) {
        if (m == 0) return sign ? -0.f : 0.f;
        // subnormal e4m3 -> value = m * 2^-9
        float v = m * 0.001953125f;
        return sign ? -v : v;
    }
    if (e == 15 && m == 7) return sign ? -448.f : 448.f; // max
    uint32_t out = (sign << 31) | ((e + 127 - 7) << 23) | (m << 20);
    float f;
    memcpy(&f, &out, 4);
    return f;
}

// ============================================================
// 1. per_token_quant_int8
// ============================================================
template<int BS,int EPT>
__global__ void ptq_kernel(const __hip_bfloat16* x,int8_t* q,float* s,int M,int N){
    int row=blockIdx.x; if(row>=M)return;
    int tid=threadIdx.x, off=row*N;
    __shared__ float s_inv, s_red[BS/64+1];
    float vals[EPT], lmax=0.f;
    #pragma unroll
    for(int i=0;i<EPT;i++){int idx=tid+i*BS; if(idx<N){float v=b2f(x[off+idx]);vals[i]=v;lmax=fmaxf(lmax,fabsf(v));}}
    int lane=tid&63,wid=tid>>6; lmax=wmax64(lmax);
    if(lane==0)s_red[wid]=lmax; __syncthreads();
    if(wid==0){lmax=(lane<(BS>>6))?s_red[lane]:0.f;lmax=wmax64(lmax);
        if(lane==0){float am=fmaxf(lmax,1e-10f);s_inv=127.f/am;s[row]=am/127.f;}}
    __syncthreads(); float inv=s_inv;
    #pragma unroll
    for(int i=0;i<EPT;i++){int idx=tid+i*BS; if(idx<N){
        float qf=round_he(vals[i]*inv); int qi=(int)qf; qi=max(-128,min(127,qi)); q[off+idx]=(int8_t)qi;}}
}

// ============================================================
// 2. per_token_group_quant_int8 (group_size along last dim)
// Triton ref: y_s = max(|x|)/127;  y = clamp(x / y_s, -128, 127).to(int8)  [trunc toward zero]
// ============================================================
template<int BS>
__global__ void ptgq_kernel(const __hip_bfloat16* x,int8_t* q,float* s,
    int M,int N,int gs){
    int row=blockIdx.x;
    int group=blockIdx.y;
    int g_start=group*gs;
    if(row>=M||g_start>=N)return;
    int g_end=min(g_start+gs,N);
    int g_len=g_end-g_start;
    int tid=threadIdx.x;
    float lmax=0.f;
    for(int i=tid;i<g_len;i+=BS){float v=b2f(x[row*N+g_start+i]);lmax=fmaxf(lmax,fabsf(v));}
    int lane=tid&63, wid=tid>>6;
    lmax=wmax64(lmax);
    __shared__ float s_red[16];
    if(lane==0) s_red[wid]=lmax;
    __syncthreads();
    int num_wf=(BS+63)/64;
    float gmax=0.f;
    for(int w=0;w<num_wf;w++) gmax=fmaxf(gmax,s_red[w]);
    __shared__ float s_lmax;
    if(tid==0){float am=fmaxf(gmax,1e-10f);s_lmax=am/127.f;s[row*(N/gs)+group]=am/127.f;}
    __syncthreads();
    float scale=s_lmax;
    for(int i=tid;i<g_len;i+=BS){
        float v=b2f(x[row*N+g_start+i]);
        float qf=v/fmaxf(scale,1e-12f);   // = v/scale = v*127/amax
        qf=fmaxf(-128.f,fminf(127.f,qf));
        int qi=(int)qf;            // truncate toward zero (matches Triton int8 cast)
        qi=max(-128,min(127,qi));
        q[row*N+g_start+i]=(int8_t)qi;
    }
}

// ============================================================
// 3. rmsnorm_self (out-param: reads x, writes out)
// ============================================================
template<int BS,int EPT>
__global__ void rmsnorm_self_kernel(const __hip_bfloat16* x, __hip_bfloat16* out, int M, int N, float eps){
    int row=blockIdx.x; if(row>=M)return;
    int tid=threadIdx.x, off=row*N;
    __shared__ float s_rrms, s_red[BS/64+1];
    float vals[EPT], ss=0.f;
    #pragma unroll
    for(int i=0;i<EPT;i++){int idx=tid+i*BS; if(idx<N){float v=b2f(x[off+idx]);vals[i]=v;ss+=v*v;}}
    int lane=tid&63,wid=tid>>6; ss=wsum64(ss);
    if(lane==0)s_red[wid]=ss; __syncthreads();
    if(wid==0){ss=(lane<(BS>>6))?s_red[lane]:0.f;ss=wsum64(ss);
        if(lane==0)s_rrms=rsqrtf(ss/(float)N+eps);}
    __syncthreads(); float rrms=s_rrms;
    #pragma unroll
    for(int i=0;i<EPT;i++){int idx=tid+i*BS; if(idx<N)out[off+idx]=f2b(vals[i]*rrms);}
}

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

// Strided variant: engine passes q = q_full[..., -rope_dim:] which is NON-CONTIGUOUS
// (stride[1] = head_dim, not rope_dim). Must index with explicit strides to stay
// in-place on the original q_full memory. q/k element stride assumed 1 (interleaved
// real,imag pairs are adjacent in the last dim).
__global__ void fused_rope_kernel_strided(
    __hip_bfloat16* q, __hip_bfloat16* k,
    const float* freqs_cis, const int* positions,
    int num_tokens, int num_q_heads, int num_k_heads, int rope_dim, int has_k,
    int64_t qst_t, int64_t qst_h, int64_t kst_t, int64_t kst_h
) {
    const int wpb = blockDim.x >> 5;
    const int gwp = (blockIdx.x * wpb) + (threadIdx.x >> 5);
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
    int64_t st_t = is_q ? qst_t : kst_t;
    int64_t st_h = is_q ? qst_h : kst_h;

    int64_t base = (int64_t)tok * st_t + (int64_t)local_h * st_h;
    int pos = positions[tok];
    const float* freq = freqs_cis + (int64_t)pos * rope_dim + (lane << 1);
    float fr = freq[0], fi = freq[1];

    int idx = lane << 1;
    float xr = b2f(dst[base + idx]);
    float xi = b2f(dst[base + idx + 1]);
    float or_ = xr * fr - xi * fi;
    float oi_ = xr * fi + xi * fr;
    dst[base + idx]     = f2b(or_);
    dst[base + idx + 1] = f2b(oi_);
}

// ============================================================
// 5. silu_and_mul  out = sigmoid(gate)*gate*up
//   split layout: x = [gate | up] concatenated, x[M, 2N], gate=x[:,0:N], up=x[:,N:2N]
// ============================================================
template<int BS,int EPT>
__global__ void silu_mul_kernel(const __hip_bfloat16* g,const __hip_bfloat16* u,
                                __hip_bfloat16* out,int M,int N){
    int row=blockIdx.x; if(row>=M)return;
    int tid=threadIdx.x, off=row*N;
    #pragma unroll
    for(int i=0;i<EPT;i++){int idx=tid+i*BS; if(idx<N){
        float gv=b2f(g[off+idx]), uv=b2f(u[off+idx]);
        out[off+idx]=f2b((1.f/(1.f+expf(-gv)))*gv*uv);}}
}

// split-layout variant: x [M, 2N] with gate=x[:,0:N], up=x[:,N:2N]; out [M, N]
template<int BS,int EPT>
__global__ void silu_mul_split_kernel(const __hip_bfloat16* x, __hip_bfloat16* out, int M, int N){
    int row=blockIdx.x; if(row>=M)return;
    int tid=threadIdx.x;
    int off_g = row * 2 * N;          // gate row start
    int off_u = off_g + N;            // up row start
    int off_o = row * N;              // out row start
    #pragma unroll
    for(int i=0;i<EPT;i++){int idx=tid+i*BS; if(idx<N){
        float gv=b2f(x[off_g+idx]), uv=b2f(x[off_u+idx]);
        out[off_o+idx]=f2b((1.f/(1.f+expf(-gv)))*gv*uv);}}
}

// ============================================================
// 6. silu_mul_masked_quant (EP MoE path)
//   For valid tokens: silu(gate)*up then per-token int8 quant (round-to-nearest).
//   For masked (mask[row]==0) tokens: zero output q and scale (graph-safe, deterministic).
// ============================================================
template<int BS,int EPT>
__global__ void silu_mul_masked_quant_kernel(
    const __hip_bfloat16* g, const __hip_bfloat16* u,
    const int* mask, int8_t* q, float* s, int M, int N
) {
    int row=blockIdx.x; if(row>=M)return;
    int tid=threadIdx.x, off=row*N;
    if (mask && mask[row]==0) {
        s[row]=0.f;
        for(int i=tid;i<N;i+=BS) q[off+i]=(int8_t)0;
        return;
    }
    __shared__ float s_inv, s_red[BS/64+1];
    float vals[EPT], lmax=0.f;
    #pragma unroll
    for(int i=0;i<EPT;i++){int idx=tid+i*BS; if(idx<N){
        float gv=b2f(g[off+idx]),uv=b2f(u[off+idx]);
        float h=(1.f/(1.f+expf(-gv)))*gv*uv;vals[i]=h;lmax=fmaxf(lmax,fabsf(h));}}
    int lane=tid&63,wid=tid>>6; lmax=wmax64(lmax);
    if(lane==0)s_red[wid]=lmax; __syncthreads();
    if(wid==0){lmax=(lane<(BS>>6))?s_red[lane]:0.f;lmax=wmax64(lmax);
        if(lane==0){float am=fmaxf(lmax,1e-10f);s_inv=127.f/am;s[row]=am/127.f;}}
    __syncthreads(); float inv=s_inv;
    #pragma unroll
    for(int i=0;i<EPT;i++){int idx=tid+i*BS; if(idx<N){
        float qf=round_he(vals[i]*inv);int qi=(int)qf;qi=max(-128,min(127,qi));q[off+idx]=(int8_t)qi;}}
}

// ============================================================
// 7. hc_split_sinkhorn (TileLang mhc.py)
//   mixes: [n, mix_hc=24] fp32; hc_scale:[3] fp32; hc_base:[24] fp32
//   pre:[n,4] fp32; post:[n,4] fp32; comb:[n,4,4] fp32
//   pre[j]  = sigmoid(mixes[j]*hc_scale[0]   + hc_base[j])        + eps
//   post[j] = 2*sigmoid(mixes[j+hc]*hc_scale[1] + hc_base[j+hc])
//   comb[j,k]= mixes[j*hc+k+2*hc]*hc_scale[2] + hc_base[j*hc+k+2*hc]
//   Sinkhorn (iters=20): first round exp(comb-rowmax) then /=rowsum+eps then /=colsum+eps;
//                        subsequent rounds: /=rowsum+eps, /=colsum+eps.
//   hc=4 (compile-time). One block per token; thread 0 computes the 4x4, all threads write.
// ============================================================
template<int HC, int ITERS>
__global__ void sinkhorn_kernel(
    const float* mixes, const float* hc_scale, const float* hc_base,
    float* pre, float* post, float* comb, int n, float eps
) {
    int row = blockIdx.x;
    if (row >= n) return;
    int tid = threadIdx.x;
    const int mix_hc = (2 + HC) * HC; // 24

    __shared__ float s_mix[24];
    __shared__ float s_comb[16];      // 4x4
    __shared__ float s_pre[4], s_post[4];

    if (tid < mix_hc) s_mix[tid] = mixes[row * mix_hc + tid];
    __syncthreads();

    if (tid == 0) {
        float sc0 = hc_scale[0], sc1 = hc_scale[1], sc2 = hc_scale[2];
        #pragma unroll
        for (int j = 0; j < HC; j++) {
            s_pre[j]  = 1.f / (1.f + expf(-(s_mix[j] * sc0 + hc_base[j]))) + eps;
            s_post[j] = 2.f * (1.f / (1.f + expf(-(s_mix[j + HC] * sc1 + hc_base[j + HC]))));
        }
        // comb[j][k]
        #pragma unroll
        for (int j = 0; j < HC; j++)
            #pragma unroll
            for (int k = 0; k < HC; k++) {
                int idx = j * HC + k + 2 * HC;
                s_comb[j * HC + k] = s_mix[idx] * sc2 + hc_base[idx];
            }
        // Sinkhorn first round: exp(comb - rowmax), /= rowsum+eps, /= colsum+eps
        #pragma unroll
        for (int j = 0; j < HC; j++) {
            float rmax = -FLT_MAX;
            #pragma unroll
            for (int k = 0; k < HC; k++) rmax = fmaxf(rmax, s_comb[j*HC+k]);
            #pragma unroll
            for (int k = 0; k < HC; k++) s_comb[j*HC+k] = expf(s_comb[j*HC+k] - rmax);
        }
        #pragma unroll
        for (int j = 0; j < HC; j++) {
            float rsum = 0.f;
            #pragma unroll
            for (int k = 0; k < HC; k++) rsum += s_comb[j*HC+k];
            rsum += eps;
            #pragma unroll
            for (int k = 0; k < HC; k++) s_comb[j*HC+k] /= rsum;
        }
        #pragma unroll
        for (int k = 0; k < HC; k++) {
            float csum = 0.f;
            #pragma unroll
            for (int j = 0; j < HC; j++) csum += s_comb[j*HC+k];
            csum += eps;
            #pragma unroll
            for (int j = 0; j < HC; j++) s_comb[j*HC+k] /= csum;
        }
        // subsequent rounds
        for (int it = 1; it < ITERS; it++) {
            #pragma unroll
            for (int j = 0; j < HC; j++) {
                float rsum = 0.f;
                #pragma unroll
                for (int k = 0; k < HC; k++) rsum += s_comb[j*HC+k];
                rsum += eps;
                #pragma unroll
                for (int k = 0; k < HC; k++) s_comb[j*HC+k] /= rsum;
            }
            #pragma unroll
            for (int k = 0; k < HC; k++) {
                float csum = 0.f;
                #pragma unroll
                for (int j = 0; j < HC; j++) csum += s_comb[j*HC+k];
                csum += eps;
                #pragma unroll
                for (int j = 0; j < HC; j++) s_comb[j*HC+k] /= csum;
            }
        }
    }
    __syncthreads();
    if (tid < HC) {
        pre[row * HC + tid]  = s_pre[tid];
        post[row * HC + tid] = s_post[tid];
    }
    if (tid < HC * HC) comb[row * HC * HC + tid] = s_comb[tid];
}

// ============================================================
// 8. act_quant_fp8 (NSA, TileLang)
//   x:[M,N] bf16 -> y:[M,N] fp8_e4m3fn, scale:[M, N/gs] fp32, group_size=gs(=128)
//   amax = max(|x|) per (row,group); amax=max(amax,1e-4); scale=amax/448;
//   y = clamp(x/scale, -448, 448) -> e4m3
// ============================================================
template<int BS>
__global__ void act_quant_fp8_kernel(
    const __hip_bfloat16* x, uint8_t* y, float* scale,
    int M, int N, int gs, float fp8_max
) {
    int row = blockIdx.x;
    int grp = blockIdx.y;
    int g_start = grp * gs;
    if (row >= M || g_start >= N) return;
    int g_end = min(g_start + gs, N);
    int g_len = g_end - g_start;
    int tid = threadIdx.x;
    float lmax = 0.f;
    for (int i = tid; i < g_len; i += BS) {
        float v = b2f(x[row * N + g_start + i]);
        lmax = fmaxf(lmax, fabsf(v));
    }
    int lane = tid & 63, wid = tid >> 6;
    lmax = wmax64(lmax);
    __shared__ float s_red[16];
    if (lane == 0) s_red[wid] = lmax;
    __syncthreads();
    int num_wf = (BS + 63) / 64;
    float gmax = 0.f;
    for (int w = 0; w < num_wf; w++) gmax = fmaxf(gmax, s_red[w]);
    __shared__ float s_sc;
    if (tid == 0) {
        gmax = fmaxf(gmax, 1e-4f);
        s_sc = gmax / fp8_max;
        scale[row * (N / gs) + grp] = s_sc;
    }
    __syncthreads();
    float sc = s_sc;
    for (int i = tid; i < g_len; i += BS) {
        float v = b2f(x[row * N + g_start + i]) / fmaxf(sc, 1e-12f);  // real value, f32_to_e4m3 clamps
        v = fmaxf(-fp8_max, fminf(fp8_max, v));
        y[row * N + g_start + i] = f32_to_e4m3(v);
    }
}

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

// ============================================================
// 10. topk_transform_512 (jit_kernel topk.cuh)
//   scores:[batch, seq_len_cap] fp32; seq_lens:[batch] int32;
//   page_table:[batch, page_table_stride] int32; out_indices:[batch, 512] int32;
//   page_size power-of-2. page_to_indices(i) = (page_table[i>>page_bits] << page_bits) | (i & mask).
//   Fast path (seq_len<=512): indices[t<seq_len]=page_to_indices(t); indices[t>=seq_len]=-1.
//   Long path (seq_len>512): radix top-512 by score (descending), then page-transform.
// ============================================================
__device__ __forceinline__ int page_to_indices(const int* page_table, int i, int page_bits) {
    int page = page_table[i >> page_bits];
    int mask = (1 << page_bits) - 1;
    return (page << page_bits) | (i & mask);
}

// float -> order-preserving uint32 key (monotonic) for radix
__device__ __forceinline__ uint32_t f32_to_key(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t mask = (u >> 31) ? 0xFFFFFFFFu : 0x80000000u;
    return u ^ mask;
}

// Block-wide top-k via radix select: find the k-th largest key (threshold) in 4
// passes of 8-bit radix (MSB first), then scan once to select all keys > threshold
// plus enough ==threshold to reach k. O(5*sl) work, ~100x faster than k-round argmax.
// shared layout (caller-provided): keys[sl] uint32 + hist[258] int.
__device__ __forceinline__ void topk_radix(
    const float* sc, int sl, int k_top, int* sel,
    uint32_t* smem_keys, int* smem_hist
) {
    int tid = threadIdx.x;
    // load order-preserving keys
    for (int i = tid; i < sl; i += blockDim.x) smem_keys[i] = f32_to_key(sc[i]);
    __syncthreads();

    uint32_t lo = 0u;            // current lower bound of threshold prefix (bytes fixed so far)
    int need = k_top;            // remaining slots
    // smem_hist[256]=t, [257]=cum_gt, [258..]=unused
    for (int pass = 0; pass < 4 && need > 0; pass++) {
        int shift = (3 - pass) * 8;
        // bytes above the current byte (already fixed). pass 0 -> no prefix.
        uint32_t prefix_mask = (pass == 0) ? 0u : (0xFFFFFFFFu << (shift + 8));
        // zero histogram 256 bins
        for (int i = tid; i < 256; i += blockDim.x) smem_hist[i] = 0;
        __syncthreads();
        // histogram current byte over candidates whose fixed prefix matches lo
        for (int i = tid; i < sl; i += blockDim.x) {
            uint32_t key = smem_keys[i];
            if ((key & prefix_mask) == (lo & prefix_mask)) {
                int b = (key >> shift) & 0xFF;
                atomicAdd(&smem_hist[b], 1);
            }
        }
        __syncthreads();
        // find threshold byte t: largest t with count(byte>t) < need
        if (tid == 0) {
            int cum = 0, t = 0, cgt = 0;
            for (int b = 255; b >= 0; b--) {
                cum += smem_hist[b];
                if (cum >= need) { t = b; cgt = cum - smem_hist[b]; break; }
            }
            smem_hist[256] = t;
            smem_hist[257] = cgt;
        }
        __syncthreads();
        int t = smem_hist[256];
        int cgt = smem_hist[257];
        need -= cgt;                        // byte>t candidates confirmed in
        // fix this byte of threshold to t
        uint32_t byte_mask = 0xFFu << shift;
        lo = (lo & ~byte_mask) | ((uint32_t)t << shift);
    }
    uint32_t threshold = lo;

    // stage 2: count key>threshold and key==threshold
    if (tid == 0) { smem_hist[256] = 0; smem_hist[257] = 0; }
    __syncthreads();
    for (int i = tid; i < sl; i += blockDim.x) {
        uint32_t key = smem_keys[i];
        if (key > threshold) atomicAdd(&smem_hist[256], 1);
        else if (key == threshold) atomicAdd(&smem_hist[257], 1);
    }
    __syncthreads();
    int n_gt = smem_hist[256];
    int n_eq = smem_hist[257];
    int need_eq = k_top - n_gt;
    if (need_eq < 0) need_eq = 0;
    if (need_eq > n_eq) need_eq = n_eq;

    // fill sel: positions [0,n_gt) with key>threshold, [n_gt, n_gt+need_eq) with key==threshold
    if (tid == 0) { smem_hist[256] = 0; smem_hist[257] = 0; }  // pos counters
    __syncthreads();
    for (int i = tid; i < sl; i += blockDim.x) {
        uint32_t key = smem_keys[i];
        if (key > threshold) {
            int p = atomicAdd(&smem_hist[256], 1);
            if (p < n_gt) sel[p] = i;
        } else if (need_eq > 0 && key == threshold) {
            int p = atomicAdd(&smem_hist[257], 1);
            if (p < need_eq) sel[n_gt + p] = i;
        }
    }
    __syncthreads();
    int total = n_gt + need_eq;
    for (int i = tid; i < k_top; i += blockDim.x) {
        if (i >= total) sel[i] = -1;   // pad if under-filled
    }
}

// ============================================================
// 10. topk_transform_512  (256 threads/block — gfx936 max 256)
// ============================================================
__global__ void topk_transform_kernel(
    const float* scores, const int* seq_lens, const int* page_tables,
    int* out_indices, int batch, int seq_len_cap, int page_table_stride,
    int page_size, int k_top
) {
    int b = blockIdx.x;
    if (b >= batch) return;
    int tid = threadIdx.x;
    int sl = seq_lens[b];
    if (sl > seq_len_cap) sl = seq_len_cap;
    const float* sc = scores + b * seq_len_cap;
    const int* pt = page_tables + b * page_table_stride;
    int* out = out_indices + b * k_top;
    int page_bits = 0; { int ps = page_size; while (ps > 1) { ps >>= 1; page_bits++; } }

    if (sl <= 0) {
        for (int i = tid; i < k_top; i += blockDim.x) out[i] = -1;
        return;
    }

    if (sl <= k_top) {
        // fast path: take all valid indices in ascending order, pad -1
        for (int i = tid; i < k_top; i += blockDim.x) {
            if (i < sl) out[i] = page_to_indices(pt, i, page_bits);
            else out[i] = -1;
        }
        return;
    }

    // long path: radix select top-k
    extern __shared__ unsigned char smem[];
    uint32_t* s_keys = (uint32_t*)smem;                    // [sl]
    int*      s_hist = (int*)(s_keys + sl);               // [259]
    int*      sel    = s_hist + 259;                        // [k_top]
    topk_radix(sc, sl, k_top, sel, s_keys, s_hist);
    __syncthreads();
    // page-transform each selected logical index
    for (int i = tid; i < k_top; i += blockDim.x) {
        int s = sel[i];
        out[i] = (s < 0) ? -1 : page_to_indices(pt, s, page_bits);
    }
}

// ============================================================
// 11. mhc_pre (sigmoid + mix)
//   output_mix[n,j] = sigmoid(input_mix[n,j]*mhc_scale[0] + mhc_base[j]) + eps
// ============================================================
template<int BS>
__global__ void mhc_pre_kernel(
    const __hip_bfloat16* input_mix, const float* mhc_scale, const float* mhc_base,
    __hip_bfloat16* output_mix, int num_tokens, int mhc_mult, float eps
) {
    int row = blockIdx.x;
    if (row >= num_tokens) return;
    int tid = threadIdx.x;
    for (int j = tid; j < mhc_mult; j += BS) {
        float v = b2f(input_mix[row * mhc_mult + j]);
        float result = 1.0f / (1.0f + expf(-(v * mhc_scale[0] + mhc_base[j]))) + eps;
        output_mix[row * mhc_mult + j] = f2b(result);
    }
}

// ============================================================
// 12. mhc_post (TileLang)
//   a:[n,hc,hc] fp32, b:[n,hc,h] bf16, c:[n,hc] fp32, d:[n,h] bf16
//   x[n,hc,h] = c[n,hc]*d[n,h] + sum_hci a[n,hci,hc]*b[n,hci,h]   (bf16 out)
//   hc=4, h=hidden. grid=n; block=128; h tiled by HBLK.
// ============================================================
template<int HC, int HBLK>
__global__ void mhc_post_kernel(
    const float* a, const __hip_bfloat16* b, const float* c, const __hip_bfloat16* d,
    __hip_bfloat16* x, int n, int hidden
) {
    int row = blockIdx.x;
    if (row >= n) return;
    int tid = threadIdx.x;
    const int hblk = HBLK;
    // load a[HC][HC] and c[HC] into registers (tiny)
    float a_reg[HC][HC], c_reg[HC];
    if (tid < HC * HC) {
        int j = tid / HC, k = tid % HC;
        a_reg[j][k] = a[row * HC * HC + j * HC + k];
    }
    if (tid < HC) c_reg[tid] = c[row * HC + tid];
    __syncthreads();
    // broadcast via shared
    __shared__ float s_a[HC][HC], s_c[HC];
    if (tid < HC * HC) { int j=tid/HC,k=tid%HC; s_a[j][k]=a_reg[j][k]; }
    if (tid < HC) s_c[tid] = c_reg[tid];
    __syncthreads();

    for (int h0 = 0; h0 < hidden; h0 += hblk) {
        // load d tile [hblk] into shared
        __shared__ float s_d[HBLK];
        for (int i = tid; i < hblk; i += blockDim.x) {
            int hi = h0 + i;
            s_d[i] = (hi < hidden) ? b2f(d[row * hidden + hi]) : 0.f;
        }
        // load b tile [HC][hblk] into shared
        __shared__ float s_b[HC][HBLK];
        for (int j = 0; j < HC; j++)
            for (int i = tid; i < hblk; i += blockDim.x) {
                int hi = h0 + i;
                s_b[j][i] = (hi < hidden) ? b2f(b[row * HC * hidden + j * hidden + hi]) : 0.f;
            }
        __syncthreads();
        // each thread computes one (hc, h) output
        // out[n,hc,h] = c[n,hc]*d[n,h] + sum_hci a[n,hc,hci]*b[n,hci,h]
        for (int i = tid; i < hblk; i += blockDim.x) {
            int hi = h0 + i;
            if (hi >= hidden) continue;
            #pragma unroll
            for (int hc = 0; hc < HC; hc++) {
                float acc = s_c[hc] * s_d[i];
                #pragma unroll
                for (int hci = 0; hci < HC; hci++)
                    acc += s_a[hc][hci] * s_b[hci][i];
                x[row * HC * hidden + hc * hidden + hi] = f2b(acc);
            }
        }
        __syncthreads();
    }
}

// ============================================================
// 13. swa_prefill_indices (TileLang)
//   seq_lens_k:[batch], seq_lens_q:[batch], cu_seqlens_q:[batch+1]
//   out:[num_q_tokens, window] int32 (per-query KV index array, -1 pad)
//   grid=ceil(num_q_tokens/4), block=128 (4 warps); one warp per query token.
// ============================================================
__global__ void swa_prefill_indices_kernel(
    int* indices, const int* seq_lens_k, const int* seq_lens_q,
    const int* cu_seqlens_q, int num_q_tokens, int batch, int window
) {
    const int wpb = blockDim.x >> 5;
    const int gwp = (blockIdx.x * wpb) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (gwp >= num_q_tokens) return;
    // find batch_id: largest b with cu_seqlens_q[b] <= gwp
    __shared__ int s_bid[8];
    int bid = -1;
    for (int b = lane; b < batch; b += 32) {
        if (cu_seqlens_q[b] <= gwp && gwp < cu_seqlens_q[b + 1]) bid = b;
    }
    // reduce: any lane that found a bid -> take min valid (unique)
    bid = (bid < 0) ? 0x7fffffff : bid;
    int mbid = bid;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        int v = __shfl_xor(bid, o);
        if (v < mbid) mbid = v;
    }
    if (lane == 0) s_bid[0] = (mbid == 0x7fffffff) ? 0 : mbid;
    __syncthreads();
    int seq_idx = s_bid[0];
    int kv_len = seq_lens_k[seq_idx];
    int qo_len = seq_lens_q[seq_idx];
    int cum_qo_len = cu_seqlens_q[seq_idx];
    int prefix_len = kv_len - qo_len;
    int curr_qo = gwp - cum_qo_len;
    int end_abs = prefix_len + curr_qo + 1;
    int start_abs = max(end_abs - window, 0);
    int old_kv_start = batch * window;            // old KV pool (prefix) per-seq window slots
    int new_kv_start = batch * window + cum_qo_len;
    int* out_row = indices + gwp * window;
    for (int j = lane; j < window; j += 32) {
        int abs_pos = start_abs + j;
        int val;
        if (abs_pos >= end_abs) val = -1;
        else if (abs_pos < prefix_len) val = old_kv_start + (abs_pos % window);
        else val = new_kv_start + (abs_pos - prefix_len);
        out_row[j] = val;
    }
}

// ============================================================
// 14. grouped_gemm (MoE W8A8 int8) — tiled, vectorized loads
//   A:[E,M,K] int8, B:[E,N,K] int8 (TN), sa:[E,M] fp32, sb:[E,N] fp32
//   C:[E,M,N] bf16; masked_m:[E]; C = (A @ B^T) * (sa[M,None]*sb[None,N])
//   grid: (tiles_N, tiles_M, E); block 256; tile TM x TN.
//   Each thread computes TM*TN/256 outputs; K loop with BK tiles, int4 vector loads.
// ============================================================
template<int TM, int TN, int BK>
__global__ __launch_bounds__(256)
void grouped_gemm_int8_kernel(
    const int8_t* A, const int8_t* B, const float* sa, const float* sb,
    __hip_bfloat16* C, const int* masked_m,
    int E, int M, int N, int K
) {
    int e = blockIdx.z;
    if (e >= E) return;
    int mm = masked_m[e];
    int tile_m = blockIdx.y * TM;
    int tile_n = blockIdx.x * TN;
    if (tile_m >= mm) return;
    int tid = threadIdx.x;
    constexpr int NTH = 256;
    int rows_in_tile = min(TM, mm - tile_m);
    int valid_cols = min(TN, N - tile_n);
    if (valid_cols <= 0) return;

    const int8_t* A_e = A + (int64_t)e * M * K;
    const int8_t* B_e = B + (int64_t)e * N * K;
    __hip_bfloat16* C_e = C + (int64_t)e * M * N;
    const float* sa_e = sa + (int64_t)e * M;
    const float* sb_e = sb + (int64_t)e * N;

    constexpr int OUTS = (TM * TN) / NTH;   // outputs per thread
    int acc[OUTS];
    #pragma unroll
    for (int o = 0; o < OUTS; o++) acc[o] = 0;

    __shared__ int8_t s_A[TM][BK];
    __shared__ int8_t s_B[TN][BK];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // cooperative load A tile [TM x BK] and B tile [TN x BK]
        for (int i = tid; i < TM * BK; i += NTH) {
            int r = i / BK, c = i % BK;
            int ar = tile_m + r, ac = k0 + c;
            s_A[r][c] = (ar < mm && ac < K) ? A_e[ar * K + ac] : (int8_t)0;
        }
        for (int i = tid; i < TN * BK; i += NTH) {
            int r = i / BK, c = i % BK;
            int br = tile_n + r, bc = k0 + c;
            s_B[r][c] = (br < N && bc < K) ? B_e[br * K + bc] : (int8_t)0;
        }
        __syncthreads();
        #pragma unroll
        for (int o = 0; o < OUTS; o++) {
            int idx = tid + o * NTH;
            int r = idx / TN, c = idx % TN;
            if (r < rows_in_tile && c < valid_cols) {
                int sum = 0;
                #pragma unroll
                for (int kk = 0; kk < BK; kk++)
                    sum += (int)s_A[r][kk] * (int)s_B[c][kk];
                acc[o] += sum;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int o = 0; o < OUTS; o++) {
        int idx = tid + o * NTH;
        int r = idx / TN, c = idx % TN;
        if (r < rows_in_tile && c < valid_cols) {
            int ar = tile_m + r, nc = tile_n + c;
            float val = (float)acc[o] * sa_e[ar] * sb_e[nc];
            C_e[ar * N + nc] = f2b(val);
        }
    }
}

// ============================================================
// C wrappers  (all stream-driven for CUDA-graph capture)
// ============================================================
extern "C" {

void launch_ptq(const void* x,void* q,void* s,int M,int N,hipStream_t st){
    constexpr int BS=256,EPT=16; int sm=(BS/64+1)*4;
    ptq_kernel<BS,EPT><<<M,BS,sm,st>>>((const __hip_bfloat16*)x,(int8_t*)q,(float*)s,M,N);}
void launch_ptgq(const void* x,void* q,void* s,int M,int N,int gs,hipStream_t st){
    constexpr int BS=128;
    int ngroups=(N+gs-1)/gs;
    dim3 grid(M,ngroups); dim3 block(BS);
    ptgq_kernel<BS><<<grid,block,0,st>>>((const __hip_bfloat16*)x,(int8_t*)q,(float*)s,M,N,gs);}
void launch_rmsnorm_self(const void* x, void* out, int M, int N, float eps, hipStream_t st){
    constexpr int BS=256,EPT=16; int sm=(BS/64+1)*4;
    rmsnorm_self_kernel<BS,EPT><<<M,BS,sm,st>>>((const __hip_bfloat16*)x,(__hip_bfloat16*)out,M,N,eps);}
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
void launch_fused_rope_strided(void* q,void* k,const void* freqs_cis,const void* positions,
    int num_tokens,int num_q_heads,int num_k_heads,int rope_dim,int has_k,
    int64_t qst_t,int64_t qst_h,int64_t kst_t,int64_t kst_h,hipStream_t st){
    int total_heads = num_q_heads + (has_k ? num_k_heads : 0);
    int total_warps = num_tokens * total_heads;
    int warps_per_block = 4;
    int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    int threads = warps_per_block * 32;
    fused_rope_kernel_strided<<<blocks,threads,0,st>>>(
        (__hip_bfloat16*)q,(__hip_bfloat16*)k,(const float*)freqs_cis,(const int*)positions,
        num_tokens,num_q_heads,num_k_heads,rope_dim,has_k,qst_t,qst_h,kst_t,kst_h);}
void launch_silu_mul(const void* g,const void* u,void* out,int M,int N,hipStream_t st){
    constexpr int BS=256,EPT=8;
    silu_mul_kernel<BS,EPT><<<M,BS,0,st>>>((const __hip_bfloat16*)g,(const __hip_bfloat16*)u,(__hip_bfloat16*)out,M,N);}
void launch_silu_mul_split(const void* x,void* out,int M,int N,hipStream_t st){
    constexpr int BS=256,EPT=8;
    silu_mul_split_kernel<BS,EPT><<<M,BS,0,st>>>((const __hip_bfloat16*)x,(__hip_bfloat16*)out,M,N);}
void launch_silu_mul_masked_quant(const void* g,const void* u,const int* mask,
    void* q,void* s,int M,int N,hipStream_t st){
    constexpr int BS=256,EPT=8; int sm=(BS/64+1)*4;
    silu_mul_masked_quant_kernel<BS,EPT><<<M,BS,sm,st>>>((const __hip_bfloat16*)g,(const __hip_bfloat16*)u,
        mask,(int8_t*)q,(float*)s,M,N);}
void launch_hc_split_sinkhorn(const void* mixes,const void* hc_scale,const void* hc_base,
    void* pre,void* post,void* comb,int n,hipStream_t st){
    constexpr int HC=4, ITERS=20;
    sinkhorn_kernel<HC,ITERS><<<n,64,0,st>>>((const float*)mixes,(const float*)hc_scale,(const float*)hc_base,
        (float*)pre,(float*)post,(float*)comb,n,1e-6f);}
void launch_act_quant_fp8(const void* x,void* y,void* s,int M,int N,int gs,hipStream_t st){
    constexpr int BS=128;
    int ngroups=(N+gs-1)/gs;
    dim3 grid(M,ngroups); dim3 block(BS);
    act_quant_fp8_kernel<BS><<<grid,block,0,st>>>((const __hip_bfloat16*)x,(uint8_t*)y,(float*)s,M,N,gs,448.0f);}
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
void launch_topk_transform(const void* scores,const void* seq_lens,const void* page_tables,
    void* out,int batch,int seq_len_cap,int page_table_stride,int page_size,int k_top,hipStream_t st){
    // shared: keys[cap] uint32 + hist[259] int + sel[k_top] int
    int smem = seq_len_cap * 4 + 259 * 4 + k_top * 4;
    topk_transform_kernel<<<batch,256,smem,st>>>(
        (const float*)scores,(const int*)seq_lens,(const int*)page_tables,(int*)out,
        batch,seq_len_cap,page_table_stride,page_size,k_top);}
void launch_mhc_pre(const void* input_mix,const void* mhc_scale,const void* mhc_base,
    void* output_mix,int num_tokens,int mhc_mult,hipStream_t st){
    constexpr int BS=128;
    mhc_pre_kernel<BS><<<num_tokens,BS,0,st>>>((const __hip_bfloat16*)input_mix,
        (const float*)mhc_scale,(const float*)mhc_base,(__hip_bfloat16*)output_mix,num_tokens,mhc_mult,1e-6f);}
void launch_mhc_post(const void* a,const void* b,const void* c,const void* d,
    void* x,int n,int hidden,hipStream_t st){
    constexpr int HC=4, HBLK=128;
    mhc_post_kernel<HC,HBLK><<<n,128,0,st>>>((const float*)a,(const __hip_bfloat16*)b,
        (const float*)c,(const __hip_bfloat16*)d,(__hip_bfloat16*)x,n,hidden);}
void launch_swa_prefill_indices(void* indices,const void* seq_lens_k,const void* seq_lens_q,
    const void* cu_seqlens_q,int num_q_tokens,int batch,int window,hipStream_t st){
    int warps_per_block = 4;
    int blocks = (num_q_tokens + warps_per_block - 1) / warps_per_block;
    swa_prefill_indices_kernel<<<blocks,warps_per_block*32,0,st>>>((int*)indices,
        (const int*)seq_lens_k,(const int*)seq_lens_q,(const int*)cu_seqlens_q,
        num_q_tokens,batch,window);}
void launch_grouped_gemm_int8(const void* A,const void* B,const void* sa,const void* sb,
    void* C,const void* masked_m,int E,int M,int N,int K,hipStream_t st){
    constexpr int TM=16, TN=64, BK=64;
    int blocks_n = (N + TN - 1) / TN;
    int blocks_m = (M + TM - 1) / TM;
    dim3 grid(blocks_n, blocks_m, E); dim3 block(256);
    grouped_gemm_int8_kernel<TM,TN,BK><<<grid,block,0,st>>>(
        (const int8_t*)A,(const int8_t*)B,(const float*)sa,(const float*)sb,
        (__hip_bfloat16*)C,(const int*)masked_m,E,M,N,K);}
}
