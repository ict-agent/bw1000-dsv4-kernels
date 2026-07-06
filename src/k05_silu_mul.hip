// k05_silu_mul.hip - silu_and_mul (PyTorch)

#include "common.hip"

// ============================================================
// 5. silu_and_mul  out = sigmoid(gate)*gate*up
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

extern "C" {
void launch_silu_mul(const void* g,const void* u,void* out,int M,int N,hipStream_t st){
    constexpr int BS=256,EPT=8;
    silu_mul_kernel<BS,EPT><<<M,BS,0,st>>>((const __hip_bfloat16*)g,(const __hip_bfloat16*)u,(__hip_bfloat16*)out,M,N);}
}
