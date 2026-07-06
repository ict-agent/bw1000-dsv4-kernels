#include "common.hip"

// 14. grouped_gemm (MoE W8A8 int8) — tiled, vectorized loads
//   A:[E,M,K] int8, B:[E,N,K] int8 (TN), sa:[E,M] fp32, sb:[E,N] fp32
//   C:[E,M,N] bf16; masked_m:[E]; C = (A @ B^T) * (sa[M,None]*sb[None,N])
//   grid: (tiles_N, tiles_M, E); block 256; tile TM x TN.
//   Each thread computes TM*TN/256 outputs; K loop with BK tiles, int4 vector loads.
// ============================================================
template<int TM, int TN, int BK>
__global__ __launch_bounds__(256)
void grouped_gemm_int8_kernel(
    const int8_t* A, const int8_t* B, const float* sa, const float* sb,
    __hip_bfloat16* C, const int* masked_m,
    int E, int M, int N, int K
) {
    int e = blockIdx.z;
    if (e >= E) return;
    int mm = masked_m[e];
    int tile_m = blockIdx.y * TM;
    int tile_n = blockIdx.x * TN;
    if (tile_m >= mm) return;
    int tid = threadIdx.x;
    constexpr int NTH = 256;
    int rows_in_tile = min(TM, mm - tile_m);
    int valid_cols = min(TN, N - tile_n);
    if (valid_cols <= 0) return;

    const int8_t* A_e = A + (int64_t)e * M * K;
    const int8_t* B_e = B + (int64_t)e * N * K;
    __hip_bfloat16* C_e = C + (int64_t)e * M * N;
    const float* sa_e = sa + (int64_t)e * M;
    const float* sb_e = sb + (int64_t)e * N;

    constexpr int OUTS = (TM * TN) / NTH;   // outputs per thread
    int acc[OUTS];
    #pragma unroll
    for (int o = 0; o < OUTS; o++) acc[o] = 0;

    __shared__ int8_t s_A[TM][BK];
    __shared__ int8_t s_B[TN][BK];

    for (int k0 = 0; k0 < K; k0 += BK) {
        // cooperative load A tile [TM x BK] and B tile [TN x BK]
        for (int i = tid; i < TM * BK; i += NTH) {
            int r = i / BK, c = i % BK;
            int ar = tile_m + r, ac = k0 + c;
            s_A[r][c] = (ar < mm && ac < K) ? A_e[ar * K + ac] : (int8_t)0;
        }
        for (int i = tid; i < TN * BK; i += NTH) {
            int r = i / BK, c = i % BK;
            int br = tile_n + r, bc = k0 + c;
            s_B[r][c] = (br < N && bc < K) ? B_e[br * K + bc] : (int8_t)0;
        }
        __syncthreads();
        #pragma unroll
        for (int o = 0; o < OUTS; o++) {
            int idx = tid + o * NTH;
            int r = idx / TN, c = idx % TN;
            if (r < rows_in_tile && c < valid_cols) {
                int sum = 0;
                #pragma unroll
                for (int kk = 0; kk < BK; kk++)
                    sum += (int)s_A[r][kk] * (int)s_B[c][kk];
                acc[o] += sum;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int o = 0; o < OUTS; o++) {
        int idx = tid + o * NTH;
        int r = idx / TN, c = idx % TN;
        if (r < rows_in_tile && c < valid_cols) {
            int ar = tile_m + r, nc = tile_n + c;
            float val = (float)acc[o] * sa_e[ar] * sb_e[nc];
            C_e[ar * N + nc] = f2b(val);
        }
    }
}

extern "C" {
void launch_grouped_gemm_int8(const void* A,const void* B,const void* sa,const void* sb,
    void* C,const void* masked_m,int E,int M,int N,int K,hipStream_t st){
    constexpr int TM=16, TN=64, BK=64;
    int blocks_n = (N + TN - 1) / TN;
    int blocks_m = (M + TM - 1) / TM;
    dim3 grid(blocks_n, blocks_m, E); dim3 block(256);
    grouped_gemm_int8_kernel<TM,TN,BK><<<grid,block,0,st>>>(
        (const int8_t*)A,(const int8_t*)B,(const float*)sa,(const float*)sb,
        (__hip_bfloat16*)C,(const int*)masked_m,E,M,N,K);}
}
