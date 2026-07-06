// w8a8_gemm.hip - W8A8 INT8 scaled GEMM kernel for DCU gfx936
// Computes: C = (A_int8 * scale_a) @ (B_int8)^T * scale_b, output BF16
// A: [M, K] int8, B: [N, K] int8, scale_a: [M], scale_b: [N], C: [M, N] bf16
//
// Bug fixes vs v1:
// 1. sdot4 returns int, need cast properly
// 2. scale_b should be [N] not [1,N], apply per-column
// 3. Each thread does ONE (row,col) dot product, not 4 columns
// 4. Fix loop: accumulate int32, apply scales at the end

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

__device__ __forceinline__ __hip_bfloat16 f2b(float v) { return __float2bfloat16(v); }

template <int BM, int BN>
__global__ void w8a8_scaled_gemm_kernel(
    const int8_t* __restrict__ A,    // [M, K] contiguous
    const int8_t* __restrict__ B,    // [N, K] contiguous
    const float* __restrict__ scale_a,  // [M]
    const float* __restrict__ scale_b,  // [N]
    __hip_bfloat16* __restrict__ C,  // [M, N]
    int M, int N, int K
) {
    // One thread per output element C[row, col]
    int row = blockIdx.x * BM + threadIdx.y;
    int col = blockIdx.y * BN + threadIdx.x;

    if (row >= M || col >= N) return;

    // INT32 accumulator
    int sum = 0;

    // Iterate over K in chunks of 4 (sdot4 processes 4 int8 pairs)
    for (int k = 0; k < K; k += 4) {
        // Load 4 int8 from A[row, k:k+4] as a single int32
        int a_val = *(reinterpret_cast<const int*>(&A[row * K + k]));
        // Load 4 int8 from B[col, k:k+4] as a single int32
        int b_val = *(reinterpret_cast<const int*>(&B[col * K + k]));
        // sdot4: dot product of 4 int8 pairs, accumulate in int32
        // signature: __builtin_amdgcn_sdot4(int a, int b, int c, bool clamp)
        sum = __builtin_amdgcn_sdot4(a_val, b_val, sum, false);
    }

    // Apply per-token and per-channel scales
    float result = (float)sum * scale_a[row] * scale_b[col];
    C[row * N + col] = f2b(result);
}

// Also handle K not divisible by 4
template <int BM, int BN>
__global__ void w8a8_scaled_gemm_kernel_general(
    const int8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    __hip_bfloat16* __restrict__ C,
    int M, int N, int K
) {
    int row = blockIdx.x * BM + threadIdx.y;
    int col = blockIdx.y * BN + threadIdx.x;
    if (row >= M || col >= N) return;

    int sum = 0;
    int k = 0;
    // Process in chunks of 4
    int k4 = K & ~3;  // largest multiple of 4 <= K
    for (; k < k4; k += 4) {
        int a_val = *(reinterpret_cast<const int*>(&A[row * K + k]));
        int b_val = *(reinterpret_cast<const int*>(&B[col * K + k]));
        sum = __builtin_amdgcn_sdot4(a_val, b_val, sum, false);
    }
    // Handle remaining elements
    for (; k < K; k++) {
        sum += (int)A[row * K + k] * (int)B[col * K + k];
    }

    float result = (float)sum * scale_a[row] * scale_b[col];
    C[row * N + col] = f2b(result);
}

extern "C" {
void launch_w8a8_scaled_gemm(
    const void* A, const void* B,
    const void* scale_a, const void* scale_b,
    void* C,
    int M, int N, int K,
    hipStream_t stream
) {
    // Use general kernel (handles K not divisible by 4)
    constexpr int BM = 16, BN = 16;
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(BN, BM);  // threadIdx.x = col, threadIdx.y = row
    w8a8_scaled_gemm_kernel_general<BM, BN><<<grid, block, 0, stream>>>(
        (const int8_t*)A, (const int8_t*)B,
        (const float*)scale_a, (const float*)scale_b,
        (__hip_bfloat16*)C, M, N, K);
}
}
