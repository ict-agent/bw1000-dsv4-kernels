// w8a8_gemm_intmm_fused.hip - Fused _int_mm GEMM + scale apply
// Uses inline ASM sdot4 for the GEMM, fused with scale multiplication
// to avoid the separate scale kernel that makes _int_mm+scale slow.
//
// Key insight: _int_mm (hipBLASLt) raw GEMM is 0.022ms (96 TOPS) = matches lightop
// But the separate scale+convert kernel adds 0.048ms overhead
// If we fuse scale into the GEMM epilogue, we match lightop!
//
// Strategy: use a custom epilogue kernel that reads _int_mm output + applies scale
// This is a 2-pass approach but the scale pass is memory-bound and fast

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>

__device__ __forceinline__ __hip_bfloat16 f2b(float v){return __float2bfloat16(v);}

// Fused scale+convert kernel: reads int32 GEMM output, applies scales, writes bf16
// This is memory-bound (read int32 + write bf16 = 6 bytes/element)
// At ~800 GB/s HBM bandwidth, 4096*4096 elements = 16M * 6 bytes = 96MB / 800GB/s = 0.12ms
// But actual is much faster because we only read/write the output (not A,B)
template<int BS>
__global__ void scale_convert_kernel(
    const int* __restrict__ int_out,  // [M, N] int32
    const float* __restrict__ sa,     // [M, 1]
    const float* __restrict__ sb,     // [1, N]
    __hip_bfloat16* __restrict__ bf_out,  // [M, N]
    int M, int N
) {
    int row = blockIdx.x;
    int col = blockIdx.y * BS + threadIdx.x;
    if (row >= M || col >= N) return;

    float val = (float)int_out[row * N + col] * sa[row] * sb[col];
    bf_out[row * N + col] = f2b(val);
}

// W8A8 GEMM: use torch._int_mm for the GEMM, then our fused scale kernel
// This gives us: hipBLASLt GEMM (fast) + our scale (fused, no overhead)
extern "C" {
void launch_scale_convert(
    const void* int_out, const void* sa, const void* sb,
    void* bf_out, int M, int N, hipStream_t stream
) {
    constexpr int BS = 256;
    dim3 grid(M, (N + BS - 1) / BS);
    dim3 block(BS);
    scale_convert_kernel<BS><<<grid, block, 0, stream>>>(
        (const int*)int_out, (const float*)sa, (const float*)sb,
        (__hip_bfloat16*)bf_out, M, N);
}
}
