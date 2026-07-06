// fused_ops_v2.hip - Additional HIP kernels for complete DSV4 coverage
// Adds: per_token_quant_int8, rope, moe_sum, moe_align
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cfloat>

__device__ __forceinline__ float bf16_to_float(__hip_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __hip_bfloat16 float_to_bf16(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ float warp_reduce_max_64(float val) {
    for (int offset = 32; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor(val, offset, 64));
    return val;
}
__device__ __forceinline__ float warp_reduce_sum_64(float val) {
    for (int offset = 32; offset > 0; offset >>= 1)
        val += __shfl_xor(val, offset, 64);
    return val;
}

// ============================================================
// KERNEL 4: per_token_quant_int8 (standalone)
// Input: x [M, N] bf16
// Output: out_int8 [M, N] int8, out_scale [M, 1] f32
// ============================================================
template <int BLOCK_SIZE>
__global__ void per_token_quant_int8_kernel(
    const __hip_bfloat16* __restrict__ x,
    int8_t* __restrict__ out_int8,
    float* __restrict__ out_scale,
    int M, int N
) {
    int row = blockIdx.x;
    if (row >= M) return;
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    const int row_off = row * N;

    // Pass 1: find abs_max
    float local_max = 0.0f;
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        float val = bf16_to_float(x[row_off + i]);
        local_max = fmaxf(local_max, fabsf(val));
    }
    // Block reduce max
    int lane = tid % 64;
    int wid = tid / 64;
    local_max = warp_reduce_max_64(local_max);
    if (lane == 0) smem[wid] = local_max;
    __syncthreads();
    if (wid == 0) {
        local_max = (lane < BLOCK_SIZE/64) ? smem[lane] : 0.0f;
        local_max = warp_reduce_max_64(local_max);
    }
    __syncthreads();
    if (tid == 0) {
        float scale = local_max / 127.0f;
        if (scale <= 0.0f) scale = 1.0f;
        out_scale[row] = scale;
        smem[0] = scale;
    }
    __syncthreads();
    float inv_scale = 1.0f / smem[0];

    // Pass 2: quantize
    for (int i = tid; i < N; i += BLOCK_SIZE) {
        float val = bf16_to_float(x[row_off + i]);
        int q = __float2int_rn(val * inv_scale);
        q = max(-128, min(127, q));
        out_int8[row_off + i] = (int8_t)q;
    }
}

// ============================================================
// KERNEL 5: RoPE (Rotary Position Embedding)
// Applies rotary embedding in-place to q and k
// Input: q [M, num_heads, head_dim], k [M, num_kv_heads, head_dim]
//        cos [M, head_dim/2], sin [M, head_dim/2]
// ============================================================
__global__ void rope_kernel(
    __hip_bfloat16* __restrict__ q,    // [M, H, D]
    __hip_bfloat16* __restrict__ k,    // [M, Hkv, D]
    const float* __restrict__ cos_cache,  // [M, D/2]
    const float* __restrict__ sin_cache,  // [M, D/2]
    int M, int num_heads, int num_kv_heads, int head_dim, int rope_dim
) {
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;

    if (token_idx >= M) return;
    if (head_idx >= num_heads + num_kv_heads) return;

    int half_rope = rope_dim / 2;
    bool is_q = (head_idx < num_heads);
    int actual_head = is_q ? head_idx : (head_idx - num_heads);
    int num_actual_heads = is_q ? num_heads : num_kv_heads;

    if (actual_head >= num_actual_heads) return;

    __hip_bfloat16* ptr = is_q ? q : k;
    int offset = (token_idx * num_actual_heads + actual_head) * head_dim;

    for (int i = tid; i < half_rope; i += blockDim.x) {
        int cos_off = token_idx * half_rope + i;
        float c = cos_cache[cos_off];
        float s = sin_cache[cos_off];

        float x0 = bf16_to_float(ptr[offset + i]);
        float x1 = bf16_to_float(ptr[offset + i + half_rope]);

        float y0 = x0 * c - x1 * s;
        float y1 = x0 * s + x1 * c;

        ptr[offset + i] = float_to_bf16(y0);
        ptr[offset + i + half_rope] = float_to_bf16(y1);
    }
}

// ============================================================
// KERNEL 6: moe_sum (reduce expert outputs)
// Input: expert_out [num_tokens * topk, hidden], topk_weights [num_tokens, topk]
// Output: output [num_tokens, hidden]
// ============================================================
template <int BLOCK_SIZE>
__global__ void moe_sum_kernel(
    const __hip_bfloat16* __restrict__ expert_out,  // [M*topk, N]
    const float* __restrict__ topk_weights,          // [M, topk]
    __hip_bfloat16* __restrict__ output,             // [M, N]
    int M, int N, int topk
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;
    if (token_idx >= M) return;

    for (int i = tid; i < N; i += BLOCK_SIZE) {
        float acc = 0.0f;
        for (int k = 0; k < topk; k++) {
            float w = topk_weights[token_idx * topk + k];
            float val = bf16_to_float(expert_out[(token_idx * topk + k) * N + i]);
            acc += w * val;
        }
        output[token_idx * N + i] = float_to_bf16(acc);
    }
}

// ============================================================
// KERNEL 7: moe_align_block_size
// Computes sorted_token_ids, expert_ids, num_tokens_post_padded
// for the MoE dispatch pattern
// ============================================================
__global__ void moe_align_simple_kernel(
    const int* __restrict__ topk_ids,     // [M, topk]
    int* __restrict__ sorted_token_ids,   // [E * block_size]
    int* __restrict__ expert_ids,         // [E]
    int* __restrict__ num_tokens_post,    // [1]
    int M, int topk, int num_experts, int block_size
) {
    // Simple single-block implementation for correctness
    // Each thread handles one expert
    int expert_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert_id >= num_experts) return;

    int count = 0;
    int out_base = expert_id * block_size;

    // Count tokens for this expert
    for (int t = 0; t < M; t++) {
        for (int k = 0; k < topk; k++) {
            if (topk_ids[t * topk + k] == expert_id) {
                if (count < block_size) {
                    sorted_token_ids[out_base + count] = t * topk + k;
                }
                count++;
            }
        }
    }
    // Pad remaining
    for (int i = count; i < block_size; i++) {
        sorted_token_ids[out_base + i] = M * topk;  // padding token
    }
    expert_ids[expert_id] = expert_id;
    if (expert_id == 0) {
        *num_tokens_post = num_experts * block_size;
    }
}

// ============================================================
// C wrappers
// ============================================================
extern "C" {

void launch_per_token_quant_int8(
    const void* x, void* out_int8, void* out_scale,
    int M, int N, hipStream_t stream
) {
    const int block_size = 256;
    int smem = (block_size / 64 + 1) * sizeof(float);
    per_token_quant_int8_kernel<block_size><<<M, block_size, smem, stream>>>(
        (const __hip_bfloat16*)x, (int8_t*)out_int8, (float*)out_scale, M, N);
}

void launch_rope(
    void* q, void* k, const void* cos_cache, const void* sin_cache,
    int M, int num_heads, int num_kv_heads, int head_dim, int rope_dim,
    hipStream_t stream
) {
    dim3 grid(M, num_heads + num_kv_heads);
    int threads = min(256, rope_dim / 2);
    rope_kernel<<<grid, threads, 0, stream>>>(
        (__hip_bfloat16*)q, (__hip_bfloat16*)k,
        (const float*)cos_cache, (const float*)sin_cache,
        M, num_heads, num_kv_heads, head_dim, rope_dim);
}

void launch_moe_sum(
    const void* expert_out, const void* topk_weights, void* output,
    int M, int N, int topk, hipStream_t stream
) {
    const int block_size = 256;
    moe_sum_kernel<block_size><<<M, block_size, 0, stream>>>(
        (const __hip_bfloat16*)expert_out, (const float*)topk_weights,
        (__hip_bfloat16*)output, M, N, topk);
}

void launch_moe_align(
    const void* topk_ids, void* sorted_token_ids, void* expert_ids,
    void* num_tokens_post, int M, int topk, int num_experts, int block_size,
    hipStream_t stream
) {
    int threads = min(256, num_experts);
    int blocks = (num_experts + threads - 1) / threads;
    moe_align_simple_kernel<<<blocks, threads, 0, stream>>>(
        (const int*)topk_ids, (int*)sorted_token_ids, (int*)expert_ids,
        (int*)num_tokens_post, M, topk, num_experts, block_size);
}

} // extern "C"
