#include "common.hip"

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


extern "C" {
void launch_ptgq(const void* x,void* q,void* s,int M,int N,int gs,hipStream_t st){
    constexpr int BS=128;
    int ngroups=(N+gs-1)/gs;
    dim3 grid(M,ngroups); dim3 block(BS);
    ptgq_kernel<BS><<<grid,block,0,st>>>((const __hip_bfloat16*)x,(int8_t*)q,(float*)s,M,N,gs);}
}
