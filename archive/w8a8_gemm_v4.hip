// w8a8_gemm_v4.hip - W8A8 INT8 GEMM v4: large tile + vectorized + pipelined
// Target:接近 lightop ASM 性能
// Key optimizations vs v3:
// 1. Tile 64x64 (vs 16x16) — better compute/memory ratio
// 2. Each thread computes 4x4=16 output elements via sdot4
// 3. Double buffering: pipeline shared memory loads with compute
// 4. Vectorized int32 loads (4 int8 per load)

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

__device__ __forceinline__ __hip_bfloat16 f2b(float v){return __float2bfloat16(v);}
__device__ __forceinline__ float b2f(const __hip_bfloat16 v){return __bfloat162float(v);}

// v4: 64x64 tile, each thread handles 4x4 outputs
// Block: 16x16 threads = 256, each computes 4x4 = 16 outputs
// Shared memory: 2 x [64][BK] for double buffering A and B
template<int BM, int BN, int BK>
__global__ void w8a8_gemm_v4_kernel(
    const int8_t* __restrict__ A,    // [M, K]
    const int8_t* __restrict__ B,    // [N, K]
    const float* __restrict__ sa,    // [M]
    const float* __restrict__ sb,    // [N]
    __hip_bfloat16* __restrict__ C,  // [M, N]
    int M, int N, int K
) {
    // 16x16 threads, each does 4x4 = 16 outputs (64x64 tile)
    constexpr int TM = 4, TN = 4;  // each thread: TM x TN outputs
    constexpr int ROW_THREADS = BM / TM;  // 16
    constexpr int COL_THREADS = BN / TN;  // 16

    int block_row = blockIdx.x * BM;
    int block_col = blockIdx.y * BN;

    int ty = threadIdx.y;  // 0..15
    int tx = threadIdx.x;  // 0..15
    int tid = ty * COL_THREADS + tx;

    // Double buffer shared memory
    extern __shared__ int8_t smem[];
    int8_t* sA[2];
    int8_t* sB[2];
    sA[0] = smem;
    sA[1] = smem + BM * BK;
    sB[0] = smem + 2 * BM * BK;
    sB[1] = smem + 2 * BM * BK + BN * BK;

    // Accumulators: TM x TN per thread
    int acc[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++)
        #pragma unroll
        for (int j = 0; j < TN; j++)
            acc[i][j] = 0;

    int num_threads = ROW_THREADS * COL_THREADS;

    // Pre-load first tile
    int buf = 0;
    for (int i = tid; i < BM * BK; i += num_threads) {
        int sr = i / BK, sc = i % BK;
        int gr = block_row + sr, gc = sc;
        sA[buf][sr * BK + sc] = (gr < M && gc < K) ? A[gr * K + gc] : (int8_t)0;
    }
    for (int i = tid; i < BN * BK; i += num_threads) {
        int sr = i / BK, sc = i % BK;
        int gr = block_col + sr, gc = sc;
        sB[buf][sr * BK + sc] = (gr < N && gc < K) ? B[gr * K + gc] : (int8_t)0;
    }
    __syncthreads();

    for (int k_start = 0; k_start < K; k_start += BK) {
        // Load next tile into the other buffer
        int next_buf = 1 - buf;
        int next_k = k_start + BK;
        if (next_k < K) {
            for (int i = tid; i < BM * BK; i += num_threads) {
                int sr = i / BK, sc = i % BK;
                int gr = block_row + sr, gc = next_k + sc;
                sA[next_buf][sr * BK + sc] = (gr < M && gc < K) ? A[gr * K + gc] : (int8_t)0;
            }
            for (int i = tid; i < BN * BK; i += num_threads) {
                int sr = i / BK, sc = i % BK;
                int gr = block_col + sr, gc = next_k + sc;
                sB[next_buf][sr * BK + sc] = (gr < N && gc < K) ? B[gr * K + gc] : (int8_t)0;
            }
        }

        // Compute on current buffer
        // Each thread: TM rows x TN cols, using sdot4 for 4 elements at a time
        #pragma unroll
        for (int ti = 0; ti < TM; ti++) {
            int row_off = (ty * TM + ti) * BK;
            #pragma unroll
            for (int tj = 0; tj < TN; tj++) {
                int col_off = (tx * TN + tj) * BK;
                // sdot4 loop
                for (int k = 0; k < BK; k += 4) {
                    int a_val = *(reinterpret_cast<const int*>(&sA[buf][row_off + k]));
                    int b_val = *(reinterpret_cast<const int*>(&sB[buf][col_off + k]));
                    acc[ti][tj] = __builtin_amdgcn_sdot4(a_val, b_val, acc[ti][tj], false);
                }
            }
        }

        __syncthreads();
        buf = next_buf;
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
            float s_b = sb[col];
            C[row * N + col] = f2b((float)acc[ti][tj] * s_a * s_b);
        }
    }
}

// Grouped GEMM for MoE: multiple expert GEMMs in one kernel
// Each block handles one (expert, output_tile) pair
template<int BM, int BN, int BK>
__global__ void grouped_w8a8_gemm_kernel(
    const int8_t* __restrict__ A,    // [total_tokens, K]
    const int8_t* __restrict__ B,    // [E, N, K] — stacked expert weights
    const float* __restrict__ sa,    // [total_tokens]
    const float* __restrict__ sb,    // [E, N]
    __hip_bfloat16* __restrict__ C,  // [total_tokens, N]
    const int* __restrict__ expert_ids,  // [num_blocks] — which expert for this block
    int M, int N, int K, int E
) {
    int block_row = blockIdx.x * BM;
    int block_col = blockIdx.y * BN;
    int expert_id = expert_ids[blockIdx.x];

    int ty = threadIdx.y, tx = threadIdx.x;
    int tid = ty * blockDim.x + tx;

    int sum = 0;
    // Simple per-element dot product (no shared memory for simplicity)
    int row = block_row + ty;
    int col = block_col + tx;
    if (row >= M || col >= N) return;

    int k4 = K & ~3;
    for (int k = 0; k < k4; k += 4) {
        int a_val = *(reinterpret_cast<const int*>(&A[row * K + k]));
        int b_val = *(reinterpret_cast<const int*>(&B[expert_id * N * K + col * K + k]));
        sum = __builtin_amdgcn_sdot4(a_val, b_val, sum, false);
    }
    for (int k = k4; k < K; k++) {
        sum += (int)A[row * K + k] * (int)B[expert_id * N * K + col * K + k];
    }

    C[row * N + col] = f2b((float)sum * sa[row] * sb[expert_id * N + col]);
}

extern "C" {
void launch_w8a8_gemm_v4(
    const void* A, const void* B, const void* sa, const void* sb,
    void* C, int M, int N, int K, hipStream_t stream
) {
    constexpr int BM=64, BN=64, BK=64;
    int smem = 2 * (BM*BK + BN*BK) * sizeof(int8_t);  // double buffer
    dim3 grid((M+BM-1)/BM, (N+BN-1)/BN);
    dim3 block(BN/4, BM/4);  // 16x16 = 256 threads
    w8a8_gemm_v4_kernel<BM,BN,BK><<<grid,block,smem,stream>>>(
        (const int8_t*)A,(const int8_t*)B,(const float*)sa,(const float*)sb,
        (__hip_bfloat16*)C,M,N,K);
}

void launch_grouped_w8a8_gemm(
    const void* A, const void* B, const void* sa, const void* sb,
    void* C, const void* expert_ids,
    int M, int N, int K, int E, hipStream_t stream
) {
    constexpr int BM=16, BN=16, BK=64;
    dim3 grid((M+BM-1)/BM, (N+BN-1)/BN);
    dim3 block(BN, BM);
    grouped_w8a8_gemm_kernel<BM,BN,BK><<<grid,block,0,stream>>>(
        (const int8_t*)A,(const int8_t*)B,(const float*)sa,(const float*)sb,
        (__hip_bfloat16*)C,(const int*)expert_ids,M,N,K,E);
}
}
