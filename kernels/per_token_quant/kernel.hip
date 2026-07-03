// dsv4_ops_hip.hip - HIP C++ kernels bit-exact to mmt-at/dsv4_ops_unit_tests refs
//
// Reference contracts (from the repo):
//   ref_rmsnorm(x, eps): x * rsqrt(x.pow(2).mean(-1,keepdim=True) + eps)
//   _quantize_input(x): absmax=x.abs().amax(-1,keepdim=True).float();
//                       scale=absmax/127; qx=(x.float()/scale).round().clamp(-128,127).char()
//   silu_and_mul: hidden = sigmoid(gate.float()) * gate * up
//
// All kernels target gfx936 (Hygon DCU BW), 2-pass + register tiling.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

__device__ __forceinline__ float b2f(const __hip_bfloat16 v){return __bfloat162float(v);}
__device__ __forceinline__ __hip_bfloat16 f2b(float v){return __float2bfloat16(v);}
__device__ __forceinline__ float wsum64(float v){for(int o=32;o>0;o>>=1)v+=__shfl_xor(v,o,64);return v;}
__device__ __forceinline__ float wmax64(float v){for(int o=32;o>0;o>>=1)v=fmaxf(v,__shfl_xor(v,o,64));return v;}

// torch.round() = round-half-to-even (banker's). __builtin_rintf matches.
__device__ __forceinline__ float round_half_even(float v){return __builtin_rintf(v);}

// ============================================================
// per_token_quant_int8 — bit-exact to _quantize_input
//   absmax (fp32), scale=absmax/127, qx=rint(x/scale).clamp(-128,127)
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
        // _quantize_input: (x.float()/scale).round() — DIVISION form, rint
        float qf=round_half_even(vals[i]*inv); // x*(127/absmax) == x/scale, rint
        int qi=(int)qf; qi=max(-128,min(127,qi)); q[off+idx]=(int8_t)qi;}}
}

// ============================================================
// fused_silu_mul — bit-exact to silu_and_mul ref
//   hidden = sigmoid(gate.float()) * gate * up   (no quant here, BF16 out)
// ============================================================
template<int BS,int EPT>
__global__ void silu_mul_kernel(const __hip_bfloat16* g,const __hip_bfloat16* u,
                                __hip_bfloat16* out,int M,int N){
    int row=blockIdx.x; if(row>=M)return;
    int tid=threadIdx.x, off=row*N;
    #pragma unroll
    for(int i=0;i<EPT;i++){int idx=tid+i*BS; if(idx<N){
        float gv=b2f(g[off+idx]), uv=b2f(u[off+idx]);
        float sg=1.f/(1.f+expf(-gv));
        float h=sg*gv*uv;
        out[off+idx]=f2b(h);}}
}

// ============================================================
// fused_silu_mul_quant — silu_and_mul + _quantize_input (fused)
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
        float gv=b2f(g[off+idx]), uv=b2f(u[off+idx]);
        float sg=1.f/(1.f+expf(-gv));
        float h=sg*gv*uv; vals[i]=h; lmax=fmaxf(lmax,fabsf(h));}}
    int lane=tid&63,wid=tid>>6; lmax=wmax64(lmax);
    if(lane==0)s_red[wid]=lmax; __syncthreads();
    if(wid==0){lmax=(lane<(BS>>6))?s_red[lane]:0.f;lmax=wmax64(lmax);
        if(lane==0){float am=fmaxf(lmax,1e-10f);s_inv=127.f/am;s[row]=am/127.f;}}
    __syncthreads(); float inv=s_inv;
    #pragma unroll
    for(int i=0;i<EPT;i++){int idx=tid+i*BS; if(idx<N){
        float qf=round_half_even(vals[i]*inv); int qi=(int)qf;
        qi=max(-128,min(127,qi)); q[off+idx]=(int8_t)qi;}}
}

// ============================================================
// rmsnorm — bit-exact to ref_rmsnorm (NO weight; weight applied by caller)
//   out = x * rsqrt(mean(x^2) + eps)
// ============================================================
template<int BS,int EPT>
__global__ void rmsnorm_kernel(const __hip_bfloat16* x,__hip_bfloat16* out,int M,int N,float eps){
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
// fused_add_rmsnorm_quant — residual+=x; rmsnorm(residual,weight); quant
// Matches engine lightop contract + _quantize_input for the quant part.
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
        float qf=round_half_even(added[i]*inv);int qi=(int)qf;
        qi=max(-128,min(127,qi));q[off+idx]=(int8_t)qi;}}
}

extern "C" {
void launch_ptq(const void* x,void* q,void* s,int M,int N,hipStream_t st){
    constexpr int BS=256,EPT=16; int sm=(BS/64+1)*4;
    ptq_kernel<BS,EPT><<<M,BS,sm,st>>>((const __hip_bfloat16*)x,(int8_t*)q,(float*)s,M,N);}
void launch_silu_mul(const void* g,const void* u,void* out,int M,int N,hipStream_t st){
    constexpr int BS=256,EPT=8; silu_mul_kernel<BS,EPT><<<M,BS,0,st>>>(
        (const __hip_bfloat16*)g,(const __hip_bfloat16*)u,(__hip_bfloat16*)out,M,N);}
void launch_silu_mul_quant(const void* g,const void* u,void* q,void* s,int M,int N,hipStream_t st){
    constexpr int BS=256,EPT=8; int sm=(BS/64+1)*4;
    silu_mul_quant_kernel<BS,EPT><<<M,BS,sm,st>>>((const __hip_bfloat16*)g,(const __hip_bfloat16*)u,(int8_t*)q,(float*)s,M,N);}
void launch_rmsnorm(const void* x,void* out,int M,int N,float eps,hipStream_t st){
    constexpr int BS=256,EPT=16; int sm=(BS/64+1)*4;
    rmsnorm_kernel<BS,EPT><<<M,BS,sm,st>>>((const __hip_bfloat16*)x,(__hip_bfloat16*)out,M,N,eps);}
void launch_add_rmsnorm_quant(void* r,const void* x,const void* w,void* q,void* s,int M,int N,float eps,hipStream_t st){
    constexpr int BS=256,EPT=16; int sm=(BS/64+1)*4;
    add_rmsnorm_quant_kernel<BS,EPT><<<M,BS,sm,st>>>((__hip_bfloat16*)r,(const __hip_bfloat16*)x,(const __hip_bfloat16*)w,(int8_t*)q,(float*)s,M,N,eps);}
}
