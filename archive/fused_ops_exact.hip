// fused_ops_exact.hip - BIT-EXACT match to lmslim Triton per_token_quant_int8
//
// Key precision fixes vs v1/v3:
// 1. absmax floor = 1e-10 (not 1.0 fallback)
// 2. Multiplicative form: q = x * (127/absmax), NOT x / (absmax/127)
// 3. Round-to-nearest-even (libdevice.nearbyint equivalent) not round-half-up
//
// This makes output bit-exact with the Triton baseline, so the Marlin GEMM
// downstream scale contract is preserved -> safe to drop-in replace.

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cfloat>

__device__ __forceinline__ float bf16_to_f(const __hip_bfloat16 v) {
    return __bfloat162float(v);
}
__device__ __forceinline__ __hip_bfloat16 f_to_bf16(float v) {
    return __float2bfloat16(v);
}

__device__ __forceinline__ float warp_sum(float val) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) val += __shfl_xor(val, off, 64);
    return val;
}
__device__ __forceinline__ float warp_max(float val) {
    #pragma unroll
    for (int off = 32; off > 0; off >>= 1) val = fmaxf(val, __shfl_xor(val, off, 64));
    return val;
}

// Round-to-nearest-even (banker's rounding), matching libdevice.nearbyint
__device__ __forceinline__ float nearbyint_rne(float v) {
    // HIP has __builtin_rintf which uses current rounding mode (default = round-to-nearest-even)
    return __builtin_rintf(v);
}

// ============================================================
// EXACT per_token_quant_int8 (matches lmslim Triton bit-for-bit)
// ============================================================
template <int BLOCK_SIZE, int ELEMS_PER_THREAD>
__global__ void per_token_quant_int8_exact_kernel(
    const __hip_bfloat16* __restrict__ x,
    int8_t* __restrict__ out_int8,
    float* __restrict__ out_scale,   // shape [M, 1]
    int M, int N
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int row_off = row * N;

    __shared__ float s_inv_absmax;
    __shared__ float s_reduce[BLOCK_SIZE / 64 + 1];

    float vals[ELEMS_PER_THREAD];
    float local_max = 0.0f;

    // Pass 1: load + absmax (fp32, matching Triton .to(tl.float32))
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = tid + i * BLOCK_SIZE;
        if (idx < N) {
            float v = bf16_to_f(x[row_off + idx]);  // fp32
            vals[i] = v;
            local_max = fmaxf(local_max, fabsf(v));
        }
    }

    int lane = tid & 63;
    int wid = tid >> 6;
    local_max = warp_max(local_max);
    if (lane == 0) s_reduce[wid] = local_max;
    __syncthreads();
    if (wid == 0) {
        local_max = (lane < (BLOCK_SIZE >> 6)) ? s_reduce[lane] : 0.0f;
        local_max = warp_max(local_max);
        if (lane == 0) {
            // EXACT: absmax = max(absmax, 1e-10)
            float absmax = fmaxf(local_max, 1e-10f);
            // scale = absmax / 127 ; we store inv = 127/absmax for multiplicative form
            s_inv_absmax = 127.0f / absmax;
            out_scale[row] = absmax / 127.0f;  // store scale (shape [M,1] handled by caller)
        }
    }
    __syncthreads();
    float inv_absmax = s_inv_absmax;

    // Pass 2: quantize - EXACT: x_q = nearbyint(x * (127/absmax))
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = tid + i * BLOCK_SIZE;
        if (idx < N) {
            // Triton: x_q = libdevice.nearbyint(x * (127/absmax)).to(int8)
            // nearbyint on the product; clamp happens via int8 cast semantics
            // but Triton stores directly to int8 (saturating on overflow).
            // Match: round then saturate to [-128,127].
            float qf = nearbyint_rne(vals[i] * inv_absmax);
            // Saturate: Triton .to(tl.int8) saturates to int8 range
            int q = (int)qf;
            q = max(-128, min(127, q));
            out_int8[row_off + idx] = (int8_t)q;
        }
    }
}

// ============================================================
// EXACT fused_add_rmsnorm_quant (residual+=x, rmsnorm, quant)
// residual is bf16 input/output (matched to lightop contract).
// ============================================================
template <int BLOCK_SIZE, int ELEMS_PER_THREAD>
__global__ void fused_add_rmsnorm_quant_exact_kernel(
    __hip_bfloat16* __restrict__ residual,
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ weight,
    int8_t* __restrict__ out_int8,
    float* __restrict__ out_scale,
    int M, int N, float eps
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int row_off = row * N;

    __shared__ float s_rrms;
    __shared__ float s_inv_absmax;
    __shared__ float s_reduce[BLOCK_SIZE / 64 + 1];

    float added_reg[ELEMS_PER_THREAD];
    float sum_sq = 0.0f;

    // Pass 1: residual += x (store back as bf16), accumulate sum_sq
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = tid + i * BLOCK_SIZE;
        if (idx < N) {
            float r = bf16_to_f(residual[row_off + idx]);
            float xv = bf16_to_f(x[row_off + idx]);
            float a = r + xv;
            residual[row_off + idx] = f_to_bf16(a);  // store back bf16
            added_reg[i] = a;
            sum_sq += a * a;
        }
    }

    int lane = tid & 63;
    int wid = tid >> 6;
    sum_sq = warp_sum(sum_sq);
    if (lane == 0) s_reduce[wid] = sum_sq;
    __syncthreads();
    if (wid == 0) {
        sum_sq = (lane < (BLOCK_SIZE >> 6)) ? s_reduce[lane] : 0.0f;
        sum_sq = warp_sum(sum_sq);
        if (lane == 0) s_rrms = rsqrtf(sum_sq / (float)N + eps);
    }
    __syncthreads();
    float rrms = s_rrms;

    // Compute normed = added * rrms * weight, find absmax (fp32)
    float local_max = 0.0f;
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = tid + i * BLOCK_SIZE;
        if (idx < N) {
            float w = bf16_to_f(weight[idx]);
            float normed = added_reg[i] * rrms * w;
            added_reg[i] = normed;  // reuse register
            local_max = fmaxf(local_max, fabsf(normed));
        }
    }

    local_max = warp_max(local_max);
    if (lane == 0) s_reduce[wid] = local_max;
    __syncthreads();
    if (wid == 0) {
        local_max = (lane < (BLOCK_SIZE >> 6)) ? s_reduce[lane] : 0.0f;
        local_max = warp_max(local_max);
        if (lane == 0) {
            float absmax = fmaxf(local_max, 1e-10f);
            s_inv_absmax = 127.0f / absmax;
            out_scale[row] = absmax / 127.0f;
        }
    }
    __syncthreads();
    float inv_absmax = s_inv_absmax;

    // Pass 2: quantize (multiplicative, round-to-nearest-even)
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = tid + i * BLOCK_SIZE;
        if (idx < N) {
            float qf = nearbyint_rne(added_reg[i] * inv_absmax);
            int q = (int)qf;
            q = max(-128, min(127, q));
            out_int8[row_off + idx] = (int8_t)q;
        }
    }
}

// ============================================================
// EXACT fused_silu_mul_quant
// ============================================================
template <int BLOCK_SIZE, int ELEMS_PER_THREAD>
__global__ void fused_silu_mul_quant_exact_kernel(
    const __hip_bfloat16* __restrict__ gate,
    const __hip_bfloat16* __restrict__ up,
    int8_t* __restrict__ out_int8,
    float* __restrict__ out_scale,
    int M, int N
) {
    const int row = blockIdx.x;
    if (row >= M) return;
    const int tid = threadIdx.x;
    const int row_off = row * N;

    __shared__ float s_inv_absmax;
    __shared__ float s_reduce[BLOCK_SIZE / 64 + 1];

    float vals[ELEMS_PER_THREAD];
    float local_max = 0.0f;

    // Pass 1: silu(gate)*up, fp32, find absmax
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = tid + i * BLOCK_SIZE;
        if (idx < N) {
            float g = bf16_to_f(gate[row_off + idx]);
            float u = bf16_to_f(up[row_off + idx]);
            float silu_g = g / (1.0f + expf(-g));
            float v = silu_g * u;
            vals[i] = v;
            local_max = fmaxf(local_max, fabsf(v));
        }
    }

    int lane = tid & 63;
    int wid = tid >> 6;
    local_max = warp_max(local_max);
    if (lane == 0) s_reduce[wid] = local_max;
    __syncthreads();
    if (wid == 0) {
        local_max = (lane < (BLOCK_SIZE >> 6)) ? s_reduce[lane] : 0.0f;
        local_max = warp_max(local_max);
        if (lane == 0) {
            float absmax = fmaxf(local_max, 1e-10f);
            s_inv_absmax = 127.0f / absmax;
            out_scale[row] = absmax / 127.0f;
        }
    }
    __syncthreads();
    float inv_absmax = s_inv_absmax;

    #pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {
        int idx = tid + i * BLOCK_SIZE;
        if (idx < N) {
            float qf = nearbyint_rne(vals[i] * inv_absmax);
            int q = (int)qf;
            q = max(-128, min(127, q));
            out_int8[row_off + idx] = (int8_t)q;
        }
    }
}

extern "C" {

void launch_per_token_quant_int8_exact(
    const void* x, void* out_int8, void* out_scale,
    int M, int N, hipStream_t stream
) {
    constexpr int BLOCK = 256;
    // N could be 4096 (hidden) or 2048 (moe intermediate)
    constexpr int EPT = 16; // covers N up to 4096
    int smem = (BLOCK / 64 + 1) * sizeof(float);
    per_token_quant_int8_exact_kernel<BLOCK, EPT><<<M, BLOCK, smem, stream>>>(
        (const __hip_bfloat16*)x, (int8_t*)out_int8, (float*)out_scale, M, N);
}

void launch_fused_add_rmsnorm_quant_exact(
    void* residual, const void* x, const void* weight,
    void* out_int8, void* out_scale,
    int M, int N, float eps, hipStream_t stream
) {
    constexpr int BLOCK = 256;
    constexpr int EPT = 16;
    int smem = (BLOCK / 64 + 1) * sizeof(float);
    fused_add_rmsnorm_quant_exact_kernel<BLOCK, EPT><<<M, BLOCK, smem, stream>>>(
        (__hip_bfloat16*)residual, (const __hip_bfloat16*)x, (const __hip_bfloat16*)weight,
        (int8_t*)out_int8, (float*)out_scale, M, N, eps);
}

void launch_fused_silu_mul_quant_exact(
    const void* gate, const void* up,
    void* out_int8, void* out_scale,
    int M, int N, hipStream_t stream
) {
    constexpr int BLOCK = 256;
    constexpr int EPT = 8; // N=2048
    int smem = (BLOCK / 64 + 1) * sizeof(float);
    fused_silu_mul_quant_exact_kernel<BLOCK, EPT><<<M, BLOCK, smem, stream>>>(
        (const __hip_bfloat16*)gate, (const __hip_bfloat16*)up,
        (int8_t*)out_int8, (float*)out_scale, M, N);
}

} // extern "C"
