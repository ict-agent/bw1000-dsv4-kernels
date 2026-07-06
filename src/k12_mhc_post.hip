#include "common.hip"

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


extern "C" {
void launch_mhc_post(const void* a,const void* b,const void* c,const void* d,
    void* x,int n,int hidden,hipStream_t st){
    constexpr int HC=4, HBLK=128;
    mhc_post_kernel<HC,HBLK><<<n,128,0,st>>>((const float*)a,(const __hip_bfloat16*)b,
        (const float*)c,(const __hip_bfloat16*)d,(__hip_bfloat16*)x,n,hidden);}
}
