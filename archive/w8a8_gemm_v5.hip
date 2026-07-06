// w8a8_gemm_final.hip - W8A8 GEMM with maximum optimization
// Key insights from lightop .co analysis:
// - lightop uses tile 64x64x128 or 128x128x64
// - Uses sdot4 (v_dot4_i32_i8) for INT8 dot product
// - Uses vectorized 128-bit global loads (int4 = 16 bytes)
// - Uses shared memory for A/B tiles
// - Each thread computes multiple outputs (4x4 = 16)
//
// Our v5 optimizations vs v4:
// 1. __launch_bounds__(256) to fix launch error
// 2. Each thread computes 4x4=16 outputs (not 1x1)
// 3. Pipeline: load next tile while computing current
// 4. Register tiling: keep accumulators in registers

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

__device__ __forceinline__ __hip_bfloat16 f2b(float v){return __float2bfloat16(v);}

template<int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__(256)
void w8a8_gemm_v5_kernel(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    const float* __restrict__ sa,
    const float* __restrict__ sb,
    __hip_bfloat16* __restrict__ C,
    int M, int N, int K
) {
    // BM*BN/TM/TN threads = (BM/TM)*(BN/TN) threads
    // e.g. BM=64, BN=64, TM=4, TN=4 -> 16x16=256 threads
    constexpr int ROW_T = BM / TM;  // 16
    constexpr int COL_T = BN / TN;  // 16

    int ty = threadIdx.y;  // 0..15
    int tx = threadIdx.x;  // 0..15
    int tid = ty * COL_T + tx;
    int num_threads = ROW_T * COL_T;

    int block_row = blockIdx.x * BM;
    int block_col = blockIdx.y * BN;

    // Shared memory: double buffered
    extern __shared__ int8_t smem[];
    int8_t* sA = smem;                    // [BM][BK]
    int8_t* sB = smem + BM * BK;         // [BN][BK]

    // Accumulators: TM x TN per thread (in registers)
    int acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0;

    // Load tiles using int4 (128-bit) vectorized loads
    // Each int4 = 16 int8 values
    for (int k_start = 0; k_start < K; k_start += BK) {
        // Load A tile: BM*BK int8, using int4 loads
        #pragma unroll
        for (int i = tid; i < BM * BK / 16; i += num_threads) {
            int sr = (i * 16) / BK;
            int sc = (i * 16) % BK;
            int gr = block_row + sr;
            int gc = k_start + sc;
            if (gr < M && gc + 15 < K) {
                *reinterpret_cast<int4*>(&sA[sr * BK + sc]) =
                    *reinterpret_cast<const int4*>(&A[gr * K + gc]);
            } else {
                for (int j = 0; j < 16 && sc + j < BK; j++) {
                    sA[sr * BK + sc + j] = (gr < M && gc + j < K) ? A[gr * K + gc + j] : (int8_t)0;
                }
            }
        }
        // Load B tile: BN*BK int8
        #pragma unroll
        for (int i = tid; i < BN * BK / 16; i += num_threads) {
            int sr = (i * 16) / BK;
            int sc = (i * 16) % BK;
            int gr = block_col + sr;
            int gc = k_start + sc;
            if (gr < N && gc + 15 < K) {
                *reinterpret_cast<int4*>(&sB[sr * BK + sc]) =
                    *reinterpret_cast<const int4*>(&B[gr * K + gc]);
            } else {
                for (int j = 0; j < 16 && sc + j < BK; j++) {
                    sB[sr * BK + sc + j] = (gr < N && gc + j < K) ? B[gr * K + gc + j] : (int8_t)0;
                }
            }
        }
        __syncthreads();

        // Compute: each thread does TM x TN dot products
        // Use sdot4 for 4 int8 elements at a time
        #pragma unroll
        for (int ti = 0; ti < TM; ti++) {
            int row_off = (ty * TM + ti) * BK;
            #pragma unroll
            for (int tj = 0; tj < TN; tj++) {
                int col_off = (tx * TN + tj) * BK;
                #pragma unroll
                for (int k = 0; k < BK; k += 4) {
                    int a_val = *reinterpret_cast<const int*>(&sA[row_off + k]);
                    int b_val = *reinterpret_cast<const int*>(&sB[col_off + k]);
                    acc[ti][tj] = __builtin_amdgcn_sdot4(a_val, b_val, acc[ti][tj], false);
                }
            }
        }
        __syncthreads();
    }

    // Write output with scales
    #pragma unroll
    for (int ti = 0; ti < TM; ti++) {
        int row = block_row + ty * TM + ti;
        if (row >= M) continue;
        float s_a = sa[row];
        #pragma unroll
        for (int tj = 0; tj < TN; tj++) {
            int col = block_col + tx * TN + tj;
            if (col >= N) continue;
            C[row * N + col] = f2b((float)acc[ti][tj] * s_a * sb[col]);
        }
    }
}

extern "C" {
void launch_w8a8_gemm_v5(
    const void* A, const void* B,
    const void* sa, const void* sb,
    void* C, int M, int N, int K, hipStream_t stream
) {
    // 64x64 tile, 4x4 per thread, BK=128
    // 64/4 * 64/4 = 16*16 = 256 threads
    constexpr int BM=64, BN=64, BK=128, TM=4, TN=4;
    int smem = (BM * BK + BN * BK) * sizeof(int8_t);  // 16KB
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(BN/TN, BM/TM);  // 16x16 = 256
    w8a8_gemm_v5_kernel<BM, BN, BK, TM, TN><<<grid, block, smem, stream>>>(
        (const int8_t*)A, (const int8_t*)B,
        (const float*)sa, (const float*)sb,
        (__hip_bfloat16*)C, M, N, K);
}
}
