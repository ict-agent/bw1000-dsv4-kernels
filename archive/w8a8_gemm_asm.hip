// test_asm3.hip - Test which inline ASM instructions work on gfx936
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

// sdot4 via inline ASM - known working from __builtin_amdgcn_sdot4
__device__ int test_sdot4(int a, int b, int c) {
    return __builtin_amdgcn_sdot4(a, b, c, false);
}

// Try different MFMA opcode formats
// Format 1: v_mfma_f32_16x16x16i8 (if exists)
__global__ void test_kernel(int* out) {
    int a = 0x01010101;
    int b = 0x01010101;
    out[0] = test_sdot4(a, b, 0); // Should be 4

    // Try MFMA via different syntax
    // The instruction might be: v_mfma_f32_16x16x16_i8
    // Or: v_mfma_i32_16x16x16i8
    // Or: v_mfma_f32_32x32x16i8

    // Let's try a simpler approach: use buffer_load for vectorized global memory
    // This is what lightop uses for efficient memory access
}

// Test vectorized global load via inline ASM
__global__ void test_vec_load(const int* __restrict__ in, int* __restrict__ out) {
    int tid = threadIdx.x;
    // Load 4 int32 (16 bytes) at once
    int4 val;
    // Try GCN global_load_dwordx4
    // On gfx936, use flat_load_dwordx4
    const int* ptr = in + tid * 4;

    // Method 1: C++ reinterpret_cast (always works)
    val = *reinterpret_cast<const int4*>(ptr);

    out[tid*4] = val.x;
    out[tid*4+1] = val.y;
    out[tid*4+2] = val.z;
    out[tid*4+3] = val.w;
}

// Test shared memory store via inline ASM
__global__ void test_smem_store(int* smem, int val) {
    int tid = threadIdx.x;
    // Use C++ shared memory
    extern __shared__ int s[];
    s[tid] = val + tid;

    // Try inline ASM for ds_write
    // asm volatile("ds_write_b32 %0, %1" : : "r"(&s[tid]), "r"(val));
    // Actually, just use normal C++ for shared memory

    __syncthreads();
    smem[tid] = s[tid];
}

// W8A8 GEMM using sdot4 + vectorized loads + shared memory
// Key optimization: use int4 (128-bit) loads for A and B
// This is what lightop ASM does but we can do it in C++
template<int BM, int BN, int BK>
__global__ void w8a8_gemm_asm_kernel(
    const int8_t* __restrict__ A,    // [M, K]
    const int8_t* __restrict__ B,    // [N, K]
    const float* __restrict__ sa,    // [M]
    const float* __restrict__ sb,    // [N]
    __hip_bfloat16* __restrict__ C,  // [M, N]
    int M, int N, int K
) {
    // Use int4 (128-bit) for vectorized loads of 16 int8 values
    // This matches lightop's memory access pattern

    extern __shared__ int8_t smem[];
    int8_t* sA = smem;              // [BM][BK]
    int8_t* sB = smem + BM * BK;   // [BN][BK]

    int ty = threadIdx.y;  // 0..BM-1
    int tx = threadIdx.x;  // 0..BN-1
    int tid = ty * blockDim.x + tx;
    int num_threads = blockDim.x * blockDim.y;

    int row = blockIdx.x * BM + ty;
    int col = blockIdx.y * BN + tx;

    int sum = 0;

    for (int k_start = 0; k_start < K; k_start += BK) {
        // Vectorized load: each thread loads 16 int8 (1 int4 = 128 bits)
        // Use reinterpret_cast for vectorized load
        #pragma unroll
        for (int i = tid; i < BM * BK / 16; i += num_threads) {
            int sr = (i * 16) / BK;
            int sc = (i * 16) % BK;
            int gr = blockIdx.x * BM + sr;
            int gc = k_start + sc;
            if (gr < M && gc < K) {
                *reinterpret_cast<int4*>(&sA[sr * BK + sc]) =
                    *reinterpret_cast<const int4*>(&A[gr * K + gc]);
            }
        }
        #pragma unroll
        for (int i = tid; i < BN * BK / 16; i += num_threads) {
            int sr = (i * 16) / BK;
            int sc = (i * 16) % BK;
            int gr = blockIdx.y * BN + sr;
            int gc = k_start + sc;
            if (gr < N && gc < K) {
                *reinterpret_cast<int4*>(&sB[sr * BK + sc]) =
                    *reinterpret_cast<const int4*>(&B[gr * K + gc]);
            }
        }
        __syncthreads();

        // Compute: use sdot4 for 4 elements at a time
        // Each thread does one dot product of sA[ty][:] @ sB[tx][:]
        #pragma unroll
        for (int k = 0; k < BK; k += 4) {
            int a_val = *reinterpret_cast<const int*>(&sA[ty * BK + k]);
            int b_val = *reinterpret_cast<const int*>(&sB[tx * BK + k]);
            sum = __builtin_amdgcn_sdot4(a_val, b_val, sum, false);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = __float2bfloat16((float)sum * sa[row] * sb[col]);
    }
}

extern "C" {
void launch_w8a8_gemm_asm(
    const void* A, const void* B,
    const void* sa, const void* sb,
    void* C, int M, int N, int K, hipStream_t stream
) {
    // Use 32x32 tile with BK=128
    // 32*128 + 32*128 = 8192 int8 = 8KB shared memory (well within 64KB limit)
    constexpr int BM = 32, BN = 32, BK = 128;
    int smem = (BM * BK + BN * BK) * sizeof(int8_t);
    dim3 grid((M + BM - 1) / BM, (N + BN - 1) / BN);
    dim3 block(BN, BM);
    w8a8_gemm_asm_kernel<BM, BN, BK><<<grid, block, smem, stream>>>(
        (const int8_t*)A, (const int8_t*)B,
        (const float*)sa, (const float*)sb,
        (__hip_bfloat16*)C, M, N, K);
}
}
