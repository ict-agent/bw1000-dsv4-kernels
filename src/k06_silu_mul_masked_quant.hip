// k06_silu_mul_masked_quant.hip - silu_mul_masked_quant (TileLang, EP MoE path)

#include "common.hip"

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

extern "C" {
void launch_silu_mul_masked_quant(const void* g,const void* u,const int* mask,
    void* q,void* s,int M,int N,hipStream_t st){
    constexpr int BS=256,EPT=8; int sm=(BS/64+1)*4;
    silu_mul_masked_quant_kernel<BS,EPT><<<M,BS,sm,st>>>((const __hip_bfloat16*)g,(const __hip_bfloat16*)u,
        mask,(int8_t*)q,(float*)s,M,N);}
}
