// fused_ops.hip - Optimized fused HIP kernels for DeepSeek V4 inference
// Target: Hygon DCU gfx936, to be used in SGLang engine
// Operations:
//   1. fused_add_rmsnorm_quant: residual += x; y = rmsnorm(residual); quant(y) -> int8
//   2. fused_silu_mul_quant: y = silu(gate) * up; quant(y) -> int8
//   3. fused_rmsnorm_quant: y = rmsnorm(x); quant(y) -> int8

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cfloat>

// ============================================================
// Utilities
// ============================================================

__device__ __forceinline__ float bf16_to_float(__hip_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __hip_bfloat16 float_to_bf16(float v) {
    return __float2bfloat16(v);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 32; offset > 0; offset >>= 1) {
        val += __shfl_xor(val, offset, 64);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 32; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor(val, offset, 64));
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % 64;
    int warp_id = threadIdx.x / 64;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    int num_warps = blockDim.x / 64;
    if (warp_id == 0) {
        val = (lane < num_warps) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    __syncthreads();
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x % 64;
    int warp_id = threadIdx.x / 64;

    val = warp_reduce_max(val);

    if (lane == 0) shared[warp_id] = val;
    __syncthreads();

    int num_warps = blockDim.x / 64;
    if (warp_id == 0) {
        val = (lane < num_warps) ? shared[lane] : -FLT_MAX;
        val = warp_reduce_max(val);
    }
    __syncthreads();
    return val;
}

// ============================================================
// KERNEL 1: Fused Add + RMSNorm + Per-token INT8 Quantization
// Input: residual [M, N] bf16, x [M, N] bf16, weight [N] bf16
// Output: residual updated in-place, out_int8 [M, N] int8, out_scale [M] f32
// One block per row. Each thread handles N/blockDim.x elements.
// ============================================================

template <int BLOCK_SIZE, int VEC_SIZE>
__global__ void fused_add_rmsnorm_quant_kernel(
    __hip_bfloat16* __restrict__ residual,    // [M, N] - updated in-place
    const __hip_bfloat16* __restrict__ x,     // [M, N]
    const __hip_bfloat16* __restrict__ weight, // [N]
    int8_t* __restrict__ out_int8,            // [M, N]
    float* __restrict__ out_scale,            // [M]
    int M, int N, float eps
) {
    int row = blockIdx.x;
    if (row >= M) return;

    extern __shared__ float smem[];

    const int tid = threadIdx.x;
    const int stride = BLOCK_SIZE;
    const int row_offset = row * N;

    // Pass 1: Add residual + x, compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < N; i += stride) {
        float r_val = bf16_to_float(residual[row_offset + i]);
        float x_val = bf16_to_float(x[row_offset + i]);
        float added = r_val + x_val;
        residual[row_offset + i] = float_to_bf16(added);
        sum_sq += added * added;
    }

    // Reduce sum of squares
    sum_sq = block_reduce_sum(sum_sq, smem);
    float rrms;
    if (tid == 0) {
        smem[0] = rsqrtf(sum_sq / (float)N + eps);
    }
    __syncthreads();
    rrms = smem[0];

    // Pass 2: Apply RMSNorm, find abs_max for quantization
    float local_max = 0.0f;
    for (int i = tid; i < N; i += stride) {
        float r_val = bf16_to_float(residual[row_offset + i]);
        float w_val = bf16_to_float(weight[i]);
        float normed = r_val * rrms * w_val;
        // Store normed temporarily in shared memory? No - recompute in pass 3
        // Just find max for scale
        local_max = fmaxf(local_max, fabsf(normed));
    }

    local_max = block_reduce_max(local_max, smem);
    float scale;
    if (tid == 0) {
        scale = local_max / 127.0f;
        if (scale <= 0.0f) scale = 1.0f;
        out_scale[row] = scale;
        smem[0] = scale;
    }
    __syncthreads();
    scale = smem[0];

    // Pass 3: Quantize to INT8
    float inv_scale = 1.0f / scale;
    for (int i = tid; i < N; i += stride) {
        float r_val = bf16_to_float(residual[row_offset + i]);
        float w_val = bf16_to_float(weight[i]);
        float normed = r_val * rrms * w_val;
        int quantized = __float2int_rn(normed * inv_scale);
        quantized = max(-128, min(127, quantized));
        out_int8[row_offset + i] = (int8_t)quantized;
    }
}

// ============================================================
// KERNEL 2: Fused SiLU(gate) * up + Per-token INT8 Quantization
// Input: gate [M, N] bf16, up [M, N] bf16
// Output: out_int8 [M, N] int8, out_scale [M] f32
// ============================================================

template <int BLOCK_SIZE>
__global__ void fused_silu_mul_quant_kernel(
    const __hip_bfloat16* __restrict__ gate,  // [M, N]
    const __hip_bfloat16* __restrict__ up,    // [M, N]
    int8_t* __restrict__ out_int8,            // [M, N]
    float* __restrict__ out_scale,            // [M]
    int M, int N
) {
    int row = blockIdx.x;
    if (row >= M) return;

    extern __shared__ float smem[];

    const int tid = threadIdx.x;
    const int stride = BLOCK_SIZE;
    const int row_offset = row * N;

    // Pass 1: Compute SiLU(gate) * up, find abs_max
    float local_max = 0.0f;
    for (int i = tid; i < N; i += stride) {
        float g = bf16_to_float(gate[row_offset + i]);
        float u = bf16_to_float(up[row_offset + i]);
        float silu_g = g / (1.0f + expf(-g));
        float val = silu_g * u;
        local_max = fmaxf(local_max, fabsf(val));
    }

    local_max = block_reduce_max(local_max, smem);
    float scale;
    if (tid == 0) {
        scale = local_max / 127.0f;
        if (scale <= 0.0f) scale = 1.0f;
        out_scale[row] = scale;
        smem[0] = scale;
    }
    __syncthreads();
    scale = smem[0];

    // Pass 2: Recompute and quantize
    float inv_scale = 1.0f / scale;
    for (int i = tid; i < N; i += stride) {
        float g = bf16_to_float(gate[row_offset + i]);
        float u = bf16_to_float(up[row_offset + i]);
        float silu_g = g / (1.0f + expf(-g));
        float val = silu_g * u;
        int quantized = __float2int_rn(val * inv_scale);
        quantized = max(-128, min(127, quantized));
        out_int8[row_offset + i] = (int8_t)quantized;
    }
}

// ============================================================
// KERNEL 3: Fused RMSNorm + Per-token INT8 Quantization
// Input: x [M, N] bf16, weight [N] bf16
// Output: out_int8 [M, N] int8, out_scale [M] f32
// ============================================================

template <int BLOCK_SIZE>
__global__ void fused_rmsnorm_quant_kernel(
    const __hip_bfloat16* __restrict__ x,     // [M, N]
    const __hip_bfloat16* __restrict__ weight, // [N]
    int8_t* __restrict__ out_int8,            // [M, N]
    float* __restrict__ out_scale,            // [M]
    int M, int N, float eps
) {
    int row = blockIdx.x;
    if (row >= M) return;

    extern __shared__ float smem[];

    const int tid = threadIdx.x;
    const int stride = BLOCK_SIZE;
    const int row_offset = row * N;

    // Pass 1: Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = tid; i < N; i += stride) {
        float val = bf16_to_float(x[row_offset + i]);
        sum_sq += val * val;
    }

    sum_sq = block_reduce_sum(sum_sq, smem);
    float rrms;
    if (tid == 0) {
        smem[0] = rsqrtf(sum_sq / (float)N + eps);
    }
    __syncthreads();
    rrms = smem[0];

    // Pass 2: Apply norm, find abs_max
    float local_max = 0.0f;
    for (int i = tid; i < N; i += stride) {
        float val = bf16_to_float(x[row_offset + i]);
        float w = bf16_to_float(weight[i]);
        float normed = val * rrms * w;
        local_max = fmaxf(local_max, fabsf(normed));
    }

    local_max = block_reduce_max(local_max, smem);
    float scale;
    if (tid == 0) {
        scale = local_max / 127.0f;
        if (scale <= 0.0f) scale = 1.0f;
        out_scale[row] = scale;
        smem[0] = scale;
    }
    __syncthreads();
    scale = smem[0];

    // Pass 3: Quantize
    float inv_scale = 1.0f / scale;
    for (int i = tid; i < N; i += stride) {
        float val = bf16_to_float(x[row_offset + i]);
        float w = bf16_to_float(weight[i]);
        float normed = val * rrms * w;
        int quantized = __float2int_rn(normed * inv_scale);
        quantized = max(-128, min(127, quantized));
        out_int8[row_offset + i] = (int8_t)quantized;
    }
}

// ============================================================
// C wrapper functions (extern "C" for PyTorch extension)
// ============================================================

extern "C" {

void launch_fused_add_rmsnorm_quant(
    void* residual, const void* x, const void* weight,
    void* out_int8, void* out_scale,
    int M, int N, float eps, hipStream_t stream
) {
    const int block_size = 256;
    int smem_size = (block_size / 64 + 1) * sizeof(float);
    fused_add_rmsnorm_quant_kernel<block_size, 4><<<M, block_size, smem_size, stream>>>(
        (__hip_bfloat16*)residual, (const __hip_bfloat16*)x, (const __hip_bfloat16*)weight,
        (int8_t*)out_int8, (float*)out_scale, M, N, eps
    );
}

void launch_fused_silu_mul_quant(
    const void* gate, const void* up,
    void* out_int8, void* out_scale,
    int M, int N, hipStream_t stream
) {
    const int block_size = 256;
    int smem_size = (block_size / 64 + 1) * sizeof(float);
    fused_silu_mul_quant_kernel<block_size><<<M, block_size, smem_size, stream>>>(
        (const __hip_bfloat16*)gate, (const __hip_bfloat16*)up,
        (int8_t*)out_int8, (float*)out_scale, M, N
    );
}

void launch_fused_rmsnorm_quant(
    const void* x, const void* weight,
    void* out_int8, void* out_scale,
    int M, int N, float eps, hipStream_t stream
) {
    const int block_size = 256;
    int smem_size = (block_size / 64 + 1) * sizeof(float);
    fused_rmsnorm_quant_kernel<block_size><<<M, block_size, smem_size, stream>>>(
        (const __hip_bfloat16*)x, (const __hip_bfloat16*)weight,
        (int8_t*)out_int8, (float*)out_scale, M, N, eps
    );
}

} // extern "C"
