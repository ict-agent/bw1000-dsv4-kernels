// dsv4_torch_ext_combined.cpp - HIP kernel + PyTorch wrapper
// Graph-safe: stream passed from Python (avoids CUDA compat header conflict)
#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

// ========== HIP kernels ==========
__device__ __forceinline__ float b2f(const __hip_bfloat16 v){return __bfloat162float(v);}
__device__ __forceinline__ __hip_bfloat16 f2b(float v){return __float2bfloat16(v);}
__device__ __forceinline__ float wsum64(float v){for(int o=32;o>0;o>>=1)v+=__shfl_xor(v,o,64);return v;}
__device__ __forceinline__ float wmax64(float v){for(int o=32;o>0;o>>=1)v=fmaxf(v,__shfl_xor(v,o,64));return v;}
__device__ __forceinline__ float round_half_even(float v){return __builtin_rintf(v);}

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
        float qf=round_half_even(vals[i]*inv); int qi=(int)qf; qi=max(-128,min(127,qi)); q[off+idx]=(int8_t)qi;}}
}

// ========== PyTorch wrapper ==========
// Stream passed as int64 from Python (torch.cuda.current_stream().cuda_stream)
std::vector<torch::Tensor> per_token_quant_int8_hip(torch::Tensor x, int64_t stream_ptr) {
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    int M = (int)(x.numel() / x.size(-1));
    int N = (int)x.size(-1);
    auto q = torch::empty_like(x, x.options().dtype(torch::kInt8));
    auto s = torch::empty({M}, x.options().dtype(torch::kFloat32));
    constexpr int BS=256, EPT=16;
    int smem=(BS/64+1)*sizeof(float);
    hipStream_t stream = (hipStream_t)stream_ptr;
    ptq_kernel<BS,EPT><<<M,BS,smem,stream>>>(reinterpret_cast<const __hip_bfloat16*>(x.data_ptr()),
                                              q.data_ptr<int8_t>(), s.data_ptr<float>(), M, N);
    return {q, s.unsqueeze(-1)};
}

// Default stream version (for non-graph use)
std::vector<torch::Tensor> per_token_quant_int8_default(torch::Tensor x) {
    return per_token_quant_int8_hip(x, 0);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("per_token_quant_int8", &per_token_quant_int8_default, "HIP per_token_quant_int8 (default stream)");
    m.def("per_token_quant_int8_stream", &per_token_quant_int8_hip, "HIP per_token_quant_int8 (explicit stream)");
}
