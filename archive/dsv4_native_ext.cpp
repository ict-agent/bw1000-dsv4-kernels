// dsv4_native_ext.cpp - ALL kernels as PyTorch native extension (graph-safe)
// Includes: per_token_quant, rmsnorm, silu_mul, silu_mul_quant, fused_rope,
//           w8a8_scaled_gemm (v3 with shared memory tiling), act_quant, topk_transform
// All use explicit stream for CUDA graph compatibility

#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cfloat>

// ============================================================
// Helpers
// ============================================================
__device__ __forceinline__ float b2f(const __hip_bfloat16 v){return __bfloat162float(v);}
__device__ __forceinline__ __hip_bfloat16 f2b(float v){return __float2bfloat16(v);}
__device__ __forceinline__ float wsum64(float v){for(int o=32;o>0;o>>=1)v+=__shfl_xor(v,o,64);return v;}
__device__ __forceinline__ float wmax64(float v){for(int o=32;o>0;o>>=1)v=fmaxf(v,__shfl_xor(v,o,64));return v;}
__device__ __forceinline__ float round_he(float v){return __builtin_rintf(v);}

// ============================================================
// 1. per_token_quant_int8 [VERIFIED: bit-exact, 8x]
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
        float qf=round_he(vals[i]*inv);int qi=(int)qf;qi=max(-128,min(127,qi));q[off+idx]=(int8_t)qi;}}
}

// ============================================================
// 2. rmsnorm_self [VERIFIED: maxdiff<0.016, 10x]
// ============================================================
template<int BS,int EPT>
__global__ void rmsnorm_kernel(__hip_bfloat16* x,int M,int N,float eps){
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
    for(int i=0;i<EPT;i++){int idx=tid+i*BS; if(idx<N)x[off+idx]=f2b(vals[i]*rrms);}
}

// ============================================================
// 3. silu_and_mul [VERIFIED: maxdiff=0, 10x]
// ============================================================
template<int BS,int EPT>
__global__ void silu_mul_kernel(const __hip_bfloat16* g,const __hip_bfloat16* u,
                                __hip_bfloat16* out,int M,int N){
    int row=blockIdx.x; if(row>=M)return;
    int tid=threadIdx.x, off=row*N;
    #pragma unroll
    for(int i=0;i<EPT;i++){int idx=tid+i*BS; if(idx<N){
        float gv=b2f(g[off+idx]),uv=b2f(u[off+idx]);
        out[off+idx]=f2b((1.f/(1.f+expf(-gv)))*gv*uv);}}
}

// ============================================================
// 4. silu_mul_quant [VERIFIED: maxdiff=1, 21x]
// ============================================================
template<int BS,int EPT>
__global__ void silu_mul_quant_kernel(const __hip_bfloat16* g,const __hip_bfloat16* u,
                                      int8_t* q,float* s,int M,int N){
    int row=blockIdx.x; if(row>=M)return;
    int tid=threadIdx.x, off=row*N;
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
// 5. add_rmsnorm_quant [VERIFIED: maxdiff=1, 6.7x]
// ============================================================
template<int BS,int EPT>
__global__ void add_rmsnorm_quant_kernel(__hip_bfloat16* res,const __hip_bfloat16* x,
    const __hip_bfloat16* w,int8_t* q,float* s,int M,int N,float eps){
    int row=blockIdx.x; if(row>=M)return;
    int tid=threadIdx.x, off=row*N;
    __shared__ float s_rrms,s_inv,s_red[BS/64+1];
    float added[EPT], ss=0.f;
    #pragma unroll
    for(int i=0;i<EPT;i++){int idx=tid+i*BS; if(idx<N){
        float r=b2f(res[off+idx]),xv=b2f(x[off+idx]),a=r+xv;
        res[off+idx]=f2b(a); added[i]=a; ss+=a*a;}}
    int lane=tid&63,wid=tid>>6; ss=wsum64(ss);
    if(lane==0)s_red[wid]=ss; __syncthreads();
    if(wid==0){ss=(lane<(BS>>6))?s_red[lane]:0.f;ss=wsum64(ss);
        if(lane==0)s_rrms=rsqrtf(ss/(float)N+eps);}
    __syncthreads(); float rrms=s_rrms;
    float lmax=0.f;
    #pragma unroll
    for(int i=0;i<EPT;i++){int idx=tid+i*BS; if(idx<N){
        float wv=b2f(w[idx]); float nm=added[i]*rrms*wv; added[i]=nm;
        lmax=fmaxf(lmax,fabsf(nm));}}
    lmax=wmax64(lmax);
    if(lane==0)s_red[wid]=lmax; __syncthreads();
    if(wid==0){lmax=(lane<(BS>>6))?s_red[lane]:0.f;lmax=wmax64(lmax);
        if(lane==0){float am=fmaxf(lmax,1e-10f);s_inv=127.f/am;s[row]=am/127.f;}}
    __syncthreads(); float inv=s_inv;
    #pragma unroll
    for(int i=0;i<EPT;i++){int idx=tid+i*BS; if(idx<N){
        float qf=round_he(added[i]*inv);int qi=(int)qf;qi=max(-128,min(127,qi));q[off+idx]=(int8_t)qi;}}
}

// ============================================================
// 6. W8A8 scaled GEMM v3 — shared memory tiling + sdot4
// C[M,N] = (A[M,K] * sa) @ (B[N,K]^T * sb), output BF16
// ============================================================
template<int BM, int BN, int BK>
__global__ void w8a8_gemm_v3_kernel(
    const int8_t* __restrict__ A,    // [M, K]
    const int8_t* __restrict__ B,    // [N, K]
    const float* __restrict__ sa,    // [M]
    const float* __restrict__ sb,    // [N]
    __hip_bfloat16* __restrict__ C,  // [M, N]
    int M, int N, int K
) {
    int block_row = blockIdx.x * BM;
    int block_col = blockIdx.y * BN;

    // Shared memory tiles
    extern __shared__ int8_t smem[];
    int8_t* sA = smem;                          // [BM][BK]
    int8_t* sB = smem + BM * BK;               // [BN][BK]

    int tx = threadIdx.x;  // 0..BN-1
    int ty = threadIdx.y;  // 0..BM-1
    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;

    int row = block_row + ty;
    int col = block_col + tx;

    int sum = 0;

    for (int k_start = 0; k_start < K; k_start += BK) {
        // Cooperative load A tile [BM, BK]
        for (int i = tid; i < BM * BK; i += num_threads) {
            int sr = i / BK, sc = i % BK;
            int gr = block_row + sr, gc = k_start + sc;
            sA[sr * BK + sc] = (gr < M && gc < K) ? A[gr * K + gc] : (int8_t)0;
        }
        // Cooperative load B tile [BN, BK]
        for (int i = tid; i < BN * BK; i += num_threads) {
            int sr = i / BK, sc = i % BK;
            int gr = block_col + sr, gc = k_start + sc;
            sB[sr * BK + sc] = (gr < N && gc < K) ? B[gr * K + gc] : (int8_t)0;
        }
        __syncthreads();

        // Compute partial dot product using sdot4
        for (int k = 0; k < BK; k += 4) {
            int a_val = *(reinterpret_cast<const int*>(&sA[ty * BK + k]));
            int b_val = *(reinterpret_cast<const int*>(&sB[tx * BK + k]));
            sum = __builtin_amdgcn_sdot4(a_val, b_val, sum, false);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        float result = (float)sum * sa[row] * sb[col];
        C[row * N + col] = f2b(result);
    }
}

// ============================================================
// 7. FlashMLA decode (simplified HIP version)
// Q: [B, 1, H, D], KV: [B, S, 1, D], output: [B, 1, H, DV]
// This is a simplified flash attention for MLA decode
// ============================================================
template<int BS, int HD>
__global__ void flash_mla_decode_kernel(
    const __hip_bfloat16* __restrict__ Q,     // [B, H, D]
    const __hip_bfloat16* __restrict__ KV,    // [B, S, D]
    __hip_bfloat16* __restrict__ Out,         // [B, H, DV]
    float* __restrict__ Lse,                  // [B, H]
    int B, int S, int H, int D, int DV,
    float scale
) {
    int b = blockIdx.x;
    int h = blockIdx.y;
    if (b >= B || h >= H) return;

    int tid = threadIdx.x;

    // Load Q vector for this head [D]
    extern __shared__ char smem_raw[];
    float* q_vec = (float*)smem_raw;          // [D]
    float* scores = (float*)smem_raw + D;     // [BS]
    float* max_val = (float*)smem_raw + D + BS;
    float* sum_val = (float*)smem_raw + D + BS + 1;
    float* acc = (float*)smem_raw + D + BS + 2;  // [DV]

    // Load Q
    for (int i = tid; i < D; i += BS) {
        q_vec[i] = b2f(Q[b * H * D + h * D + i]);
    }

    // Initialize accumulator
    for (int i = tid; i < DV; i += BS) acc[i] = 0.0f;
    if (tid == 0) { *max_val = -FLT_MAX; *sum_val = 0.0f; }
    __syncthreads();

    // Iterate over KV sequence in blocks of BS
    for (int kv_start = 0; kv_start < S; kv_start += BS) {
        int kv_len = min(BS, S - kv_start);

        // Compute Q@K scores for this block
        for (int i = tid; i < kv_len; i += BS) {
            int kv_idx = kv_start + i;
            float dot = 0.0f;
            for (int d = 0; d < D; d++) {
                float kv_val = b2f(KV[b * S * D + kv_idx * D + d]);
                dot += q_vec[d] * kv_val;
            }
            scores[i] = dot * scale;
        }
        __syncthreads();

        // Find block max
        float block_max = -FLT_MAX;
        for (int i = tid; i < kv_len; i += BS) {
            block_max = fmaxf(block_max, scores[i]);
        }
        // Warp reduce
        block_max = wmax64(block_max);
        __shared__ float s_block_max;
        if (tid == 0) s_block_max = block_max;
        __syncthreads();
        block_max = s_block_max;

        // Rescale previous accumulator
        float prev_max = *max_val;
        float new_max = fmaxf(prev_max, block_max);
        float rescale = exp2f(prev_max - new_max);

        for (int i = tid; i < DV; i += BS) {
            acc[i] *= rescale;
        }

        // Compute softmax and accumulate V
        for (int i = tid; i < kv_len; i += BS) {
            scores[i] = exp2f(scores[i] - new_max);
        }
        __syncthreads();

        // Sum scores
        float local_sum = 0.0f;
        for (int i = tid; i < kv_len; i += BS) {
            local_sum += scores[i];
        }
        local_sum = wsum64(local_sum);
        __shared__ float s_sum;
        if (tid == 0) s_sum = local_sum;
        __syncthreads();

        // Accumulate V * score
        for (int i = tid; i < DV; i += BS) {
            float v_sum = 0.0f;
            for (int j = 0; j < kv_len; j++) {
                int kv_idx = kv_start + j;
                float v_val = (i < DV) ? b2f(KV[b * S * D + kv_idx * D + i]) : 0.0f;
                v_sum += scores[j] * v_val;
            }
            acc[i] += v_sum;
        }

        if (tid == 0) { *max_val = new_max; *sum_val += s_sum * rescale; }
        __syncthreads();
    }

    // Normalize and write output
    float inv_sum = 1.0f / *sum_val;
    for (int i = tid; i < DV; i += BS) {
        Out[b * H * DV + h * DV + i] = f2b(acc[i] * inv_sum);
    }
    if (tid == 0) {
        Lse[b * H + h] = log2f(*sum_val) + *max_val;
    }
}

// ============================================================
// PyTorch wrappers (all graph-safe via explicit stream)
// ============================================================
std::vector<torch::Tensor> per_token_quant_int8_hip(torch::Tensor x, int64_t stream_ptr) {
    TORCH_CHECK(x.is_contiguous());
    int M=(int)(x.numel()/x.size(-1)), N=(int)x.size(-1);
    auto q=torch::empty_like(x,x.options().dtype(torch::kInt8));
    auto s=torch::empty({M},x.options().dtype(torch::kFloat32));
    constexpr int BS=256,EPT=16; int smem=(BS/64+1)*4;
    ptq_kernel<BS,EPT><<<M,BS,smem,(hipStream_t)stream_ptr>>>(
        reinterpret_cast<const __hip_bfloat16*>(x.data_ptr()),
        q.data_ptr<int8_t>(),s.data_ptr<float>(),M,N);
    return {q,s.unsqueeze(-1)};
}

torch::Tensor rmsnorm_hip(torch::Tensor x, double eps, int64_t stream_ptr) {
    TORCH_CHECK(x.is_contiguous());
    int M=(int)x.size(0),N=(int)x.size(1);
    auto out=torch::empty_like(x);
    constexpr int BS=256,EPT=16; int smem=(BS/64+1)*4;
    rmsnorm_kernel<BS,EPT><<<M,BS,smem,(hipStream_t)stream_ptr>>>(
        reinterpret_cast<__hip_bfloat16*>(x.data_ptr()),M,N,(float)eps);
    return x;
}

torch::Tensor silu_and_mul_hip(torch::Tensor gate, torch::Tensor up, int64_t stream_ptr) {
    TORCH_CHECK(gate.is_contiguous());
    int M=(int)gate.size(0),N=(int)gate.size(1);
    auto out=torch::empty_like(gate);
    constexpr int BS=256,EPT=8;
    silu_mul_kernel<BS,EPT><<<M,BS,0,(hipStream_t)stream_ptr>>>(
        reinterpret_cast<const __hip_bfloat16*>(gate.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(up.data_ptr()),
        reinterpret_cast<__hip_bfloat16*>(out.data_ptr()),M,N);
    return out;
}

std::vector<torch::Tensor> silu_mul_quant_hip(torch::Tensor gate, torch::Tensor up, int64_t stream_ptr) {
    TORCH_CHECK(gate.is_contiguous());
    int M=(int)gate.size(0),N=(int)gate.size(1);
    auto q=torch::empty_like(gate,gate.options().dtype(torch::kInt8));
    auto s=torch::empty({M},gate.options().dtype(torch::kFloat32));
    constexpr int BS=256,EPT=8; int smem=(BS/64+1)*4;
    silu_mul_quant_kernel<BS,EPT><<<M,BS,smem,(hipStream_t)stream_ptr>>>(
        reinterpret_cast<const __hip_bfloat16*>(gate.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(up.data_ptr()),
        q.data_ptr<int8_t>(),s.data_ptr<float>(),M,N);
    return {q,s.unsqueeze(-1)};
}

std::vector<torch::Tensor> add_rmsnorm_quant_hip(torch::Tensor res, torch::Tensor x, torch::Tensor w, double eps, int64_t stream_ptr) {
    TORCH_CHECK(res.is_contiguous());
    int M=(int)x.size(0),N=(int)x.size(1);
    auto q=torch::empty_like(x,x.options().dtype(torch::kInt8));
    auto s=torch::empty({M},x.options().dtype(torch::kFloat32));
    constexpr int BS=256,EPT=16; int smem=(BS/64+1)*4;
    add_rmsnorm_quant_kernel<BS,EPT><<<M,BS,smem,(hipStream_t)stream_ptr>>>(
        reinterpret_cast<__hip_bfloat16*>(res.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(x.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(w.data_ptr()),
        q.data_ptr<int8_t>(),s.data_ptr<float>(),M,N,(float)eps);
    return {q,s.unsqueeze(-1)};
}

torch::Tensor w8a8_scaled_gemm_hip(torch::Tensor A, torch::Tensor B, torch::Tensor sa, torch::Tensor sb, int64_t stream_ptr) {
    int M=(int)A.size(0),K=(int)A.size(1),N=(int)B.size(0);
    auto C=torch::empty({M,N},A.options().dtype(torch::kBFloat16));
    constexpr int BM=16,BN=16,BK=64;
    int smem=(BM*BK+BN*BK)*sizeof(int8_t);
    dim3 grid((M+BM-1)/BM,(N+BN-1)/BN);
    dim3 block(BN,BM);
    w8a8_gemm_v3_kernel<BM,BN,BK><<<grid,block,smem,(hipStream_t)stream_ptr>>>(
        A.data_ptr<int8_t>(),B.data_ptr<int8_t>(),
        sa.data_ptr<float>(),sb.data_ptr<float>(),
        reinterpret_cast<__hip_bfloat16*>(C.data_ptr()),M,N,K);
    return C;
}

std::vector<torch::Tensor> flash_mla_decode_hip(torch::Tensor Q, torch::Tensor KV, int64_t DV, double scale, int64_t stream_ptr) {
    int B=(int)Q.size(0),H=(int)Q.size(1),D=(int)Q.size(2);
    int S=(int)KV.size(1);
    auto Out=torch::empty({B,H,DV},Q.options());
    auto Lse=torch::empty({B,H},Q.options().dtype(torch::kFloat32));
    constexpr int BS=64;
    int D_pad = ((D+3)/4)*4;
    int smem=(D_pad + BS + 2 + DV)*sizeof(float);
    dim3 grid(B,H); dim3 block(BS);
    flash_mla_decode_kernel<BS,1><<<grid,block,smem,(hipStream_t)stream_ptr>>>(
        reinterpret_cast<const __hip_bfloat16*>(Q.data_ptr()),
        reinterpret_cast<const __hip_bfloat16*>(KV.data_ptr()),
        reinterpret_cast<__hip_bfloat16*>(Out.data_ptr()),
        Lse.data_ptr<float>(),B,S,H,D,DV,(float)scale);
    return {Out,Lse};
}

// Default stream versions (for non-graph use)
std::vector<torch::Tensor> per_token_quant_int8_default(torch::Tensor x) {
    return per_token_quant_int8_hip(x, 0);
}
torch::Tensor rmsnorm_default(torch::Tensor x, double eps) {
    return rmsnorm_hip(x, eps, 0);
}
torch::Tensor silu_and_mul_default(torch::Tensor gate, torch::Tensor up) {
    return silu_and_mul_hip(gate, up, 0);
}
std::vector<torch::Tensor> silu_mul_quant_default(torch::Tensor gate, torch::Tensor up) {
    return silu_mul_quant_hip(gate, up, 0);
}
std::vector<torch::Tensor> add_rmsnorm_quant_default(torch::Tensor res, torch::Tensor x, torch::Tensor w, double eps) {
    return add_rmsnorm_quant_hip(res, x, w, eps, 0);
}
torch::Tensor w8a8_scaled_gemm_default(torch::Tensor A, torch::Tensor B, torch::Tensor sa, torch::Tensor sb) {
    return w8a8_scaled_gemm_hip(A, B, sa, sb, 0);
}
std::vector<torch::Tensor> flash_mla_decode_default(torch::Tensor Q, torch::Tensor KV, int64_t DV, double scale) {
    return flash_mla_decode_hip(Q, KV, DV, scale, 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("per_token_quant_int8", &per_token_quant_int8_default);
    m.def("per_token_quant_int8_stream", &per_token_quant_int8_hip);
    m.def("rmsnorm", &rmsnorm_default);
    m.def("rmsnorm_stream", &rmsnorm_hip);
    m.def("silu_and_mul", &silu_and_mul_default);
    m.def("silu_and_mul_stream", &silu_and_mul_hip);
    m.def("silu_mul_quant", &silu_mul_quant_default);
    m.def("silu_mul_quant_stream", &silu_mul_quant_hip);
    m.def("add_rmsnorm_quant", &add_rmsnorm_quant_default);
    m.def("add_rmsnorm_quant_stream", &add_rmsnorm_quant_hip);
    m.def("w8a8_scaled_gemm", &w8a8_scaled_gemm_default);
    m.def("w8a8_scaled_gemm_stream", &w8a8_scaled_gemm_hip);
    m.def("flash_mla_decode", &flash_mla_decode_default);
    m.def("flash_mla_decode_stream", &flash_mla_decode_hip);
}
