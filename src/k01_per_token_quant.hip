// k01_per_token_quant.hip - per_token_quant_int8 (Triton, bit-exact)

#include "common.hip"

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

extern "C" {
void launch_ptq(const void* x,void* q,void* s,int M,int N,hipStream_t st){
    constexpr int BS=256,EPT=16; int sm=(BS/64+1)*4;
    ptq_kernel<BS,EPT><<<M,BS,sm,st>>>((const __hip_bfloat16*)x,(int8_t*)q,(float*)s,M,N);}
}
