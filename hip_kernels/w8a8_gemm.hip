// w8a8_gemm.hip - W8A8 INT8 scaled GEMM kernel for DCU gfx936
// Replaces triton_scaled_mm / gemm_w8a8_smooth
// Computes: C = (A_int8 * scale_a) @ (B_int8 * scale_b)^T, output BF16
// A: [M, K] int8, B: [N, K] int8 (non-contiguous transpose),
// scale_a: [M, 1] f32, scale_b: [1, N] f32, C: [M, N] bf16
//
// Key: MFMA f32_i8 instruction for INT8 matrix multiply on gfx936

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

// gfx936 MFMA: m_grouped_w8a8_gemm uses m_i8 instructions
// __builtin_amdgcn_mfma_f32_16x16x16_i8 performs 16x16x16 INT8 matmul
// But we confirmed earlier that MFMA intrinsics have issues on this DTK.
// Alternative: use DP4A (dot product 4 int8) for accumulation.

__device__ __forceinline__ int dp4a(int a, int b, int c) {
    // Dot product of 4 int8 pairs, accumulate in int32
    // HIP equivalent of __dp4a
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// W8A8 GEMM kernel: C[M,N] = (A[M,K] * sa) @ (B[N,K]^T * sb)
// Tiling: BLOCK_M x BLOCK_N, iterate over K in BLOCK_K chunks
template <int BM, int BN, int BK>
__global__ void w8a8_scaled_gemm_kernel(
    const int8_t* __restrict__ A,    // [M, K] contiguous
    const int8_t* __restrict__ B,    // [N, K] (B is [N,K], we access B[n][k])
    const float* __restrict__ scale_a,  // [M]
    const float* __restrict__ scale_b,  // [N] or [1,N]
    __hip_bfloat16* __restrict__ C,  // [M, N]
    int M, int N, int K
) {
    int pid_m = blockIdx.x;
    int pid_n = blockIdx.y;
    int tid = threadIdx.x;

    // Each block computes BM x BN tile of C
    int row_start = pid_m * BM;
    int col_start = pid_n * BN;

    if (row_start >= M || col_start >= N) return;

    // Accumulator
    float acc[BM][BN/4]; // Each thread handles 4 columns via dp4a
    // Simplified: each thread handles one (row, col) pair, iterate K
    // For small M (decode batch=1), this is efficient enough

    int row = row_start + tid / (BN/4);
    int col_quad = tid % (BN/4);  // which group of 4 columns
    int col = col_start + col_quad * 4;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    // Iterate over K in chunks of 4 (dp4a processes 4 elements)
    for (int k = 0; k < K; k += 4) {
        // Load 4 int8 from A[row, k:k+4]
        int a_val = *(reinterpret_cast<const int*>(&A[row * K + k]));
        // Load 4 int8 from B[col, k:k+4] for each of 4 columns
        // Actually we need B[col:col+4, k:k+4] for 4 columns at once
        // But B is [N, K], so B[col+i, k:k+4]
        for (int c = 0; c < 4 && col + c < N; c++) {
            int b_val = *(reinterpret_cast<const int*>(&B[(col + c) * K + k]));
            sum = __builtin_amdgcn_sdot4(a_val, b_val, (int)sum, false);
            // This gives sum of A[row,k:k+4] * B[col+c, k:k+4] accumulated
        }
    }

    // Apply scales and write output
    if (col < N && row < M) {
        float sa = scale_a[row];
        float sb = scale_b[col];
        float result = sum * sa * sb;
        // Write to C[row, col]
        C[row * N + col] = __float2bfloat16(result);
    }
}

extern "C" {
void launch_w8a8_scaled_gemm(
    const void* A, const void* B,
    const void* scale_a, const void* scale_b,
    void* C,
    int M, int N, int K,
    hipStream_t stream
) {
    constexpr int BM = 1, BN = 64, BK = 4;
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    int threads = BN / 4;  // each thread handles 4 columns
    w8a8_scaled_gemm_kernel<BM, BN, BK><<<grid, threads, 0, stream>>>(
        (const int8_t*)A, (const int8_t*)B,
        (const float*)scale_a, (const float*)scale_b,
        (__hip_bfloat16*)C, M, N, K);
}
}
