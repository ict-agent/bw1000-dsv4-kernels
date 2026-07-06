// k03_rmsnorm.hip - rmsnorm_self (TileLang, in-place)

#include "common.hip"

// ============================================================
// 3. rmsnorm_self (in-place)
// ============================================================
template<int BS,int EPT>
__global__ void rmsnorm_self_kernel(__hip_bfloat16* x,int M,int N,float eps){
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

extern "C" {
void launch_rmsnorm_self(void* x,int M,int N,float eps,hipStream_t st){
    constexpr int BS=256,EPT=16; int sm=(BS/64+1)*4;
    rmsnorm_self_kernel<BS,EPT><<<M,BS,sm,st>>>((__hip_bfloat16*)x,M,N,eps);}
}
