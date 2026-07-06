// k08_act_quant_fp8.hip - act_quant_fp8 (TileLang NSA, fp8_e4m3 output)

#include "common.hip"

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

extern "C" {
void launch_act_quant_fp8(const void* x,void* y,void* s,int M,int N,int gs,hipStream_t st){
    constexpr int BS=128;
    int ngroups=(N+gs-1)/gs;
    dim3 grid(M,ngroups); dim3 block(BS);
    act_quant_fp8_kernel<BS><<<grid,block,0,st>>>((const __hip_bfloat16*)x,(uint8_t*)y,(float*)s,M,N,gs,448.0f);}
}
