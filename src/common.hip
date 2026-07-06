// common.hip - Common helpers shared across all dsv4 HIP kernels.
// Includes + device helper functions only (no kernels).
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdint>
#include <cfloat>

// ============================================================
// Common helpers
// ============================================================
__device__ __forceinline__ float b2f(const __hip_bfloat16 v){return __bfloat162float(v);}
__device__ __forceinline__ __hip_bfloat16 f2b(float v){return __float2bfloat16(v);}
__device__ __forceinline__ float wsum64(float v){for(int o=32;o>0;o>>=1)v+=__shfl_xor(v,o,64);return v;}
__device__ __forceinline__ float wmax64(float v){for(int o=32;o>0;o>>=1)v=fmaxf(v,__shfl_xor(v,o,64));return v;}
__device__ __forceinline__ float wsum32(float v){for(int o=16;o>0;o>>=1)v+=__shfl_xor(v,o,32);return v;}
__device__ __forceinline__ float wmax32(float v){for(int o=16;o>0;o>>=1)v=fmaxf(v,__shfl_xor(v,o,32));return v;}
__device__ __forceinline__ float round_he(float v){return __builtin_rintf(v);}

// FP8 e4m3fn encoding (max=448) with round-to-nearest-even, matches torch.float8_e4m3fn cast.
__device__ __forceinline__ uint8_t f32_to_e4m3(float f){
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t sign = (u >> 31) & 1;
    uint32_t abs_u = u & 0x7FFFFFFFu;          // |f| as float bits (sign cleared)
    if (abs_u == 0) return (uint8_t)(sign << 7);

    // Interpret |f|
    int exp_f = (int32_t)(abs_u >> 23) - 127;   // unbiased exp
    uint32_t mant_f = abs_u & 0x7FFFFFu;        // 23-bit fraction (without hidden 1)

    // e4m3: bias 7, normal exp biased in [1,14] OR 15 with mantissa<=6 (e4m3fn max=448).
    // value = (1.mmm) * 2^(e-7). exp=15,mant=7 reserved (NaN/inf). Subnormal when e4==0.
    int e4 = exp_f + 7;                          // tentative e4m3 biased exp

    // ---- true overflow (exp > 15, or exp==15 with mant that would round to 7) -> clamp 448 ----
    // handled below after mantissa rounding.

    uint32_t out;
    if (e4 >= 1) {
        // normal: keep 3 mantissa bits with round-to-nearest-even
        uint32_t m3 = mant_f >> 20;              // top 3 bits
        uint32_t round_bit = (mant_f >> 19) & 1;
        uint32_t sticky = (mant_f & ((1u << 19) - 1)) ? 1u : 0u;
        uint32_t roundup = round_bit & (sticky | (m3 & 1));
        m3 += roundup;
        if (m3 == 8) { m3 = 0; e4 += 1; }         // carried into exp
        // e4m3fn: exp=15 valid only for mantissa 0..6 (max 448). mant=7 -> clamp to 6.
        if (e4 > 15) return (uint8_t)((sign << 7) | 0x7E);
        if (e4 == 15 && m3 == 7) return (uint8_t)((sign << 7) | 0x7E);  // 0x7E = sign|111|110
        out = (sign << 7) | (e4 << 3) | m3;
    } else {
        // subnormal or underflow. e4m3 subnormal value = k * 2^-9, k in [1,7].
        // |f| = (1.mant_f) * 2^exp_f. Express as k * 2^-9:
        //   k = (1.mant_f) * 2^(exp_f + 9)
        // full fixed-point mantissa with hidden bit: M = (1<<23) | mant_f  (24-bit, value 1.f * 2^23)
        // k = M * 2^(exp_f + 9 - 23) = M * 2^(exp_f - 14)
        int shift = 14 - exp_f;                  // k = M >> shift (with rounding)
        uint32_t M = (1u << 23) | mant_f;
        if (shift >= 32) {
            // underflow -> 0
            out = (sign << 7);
        } else if (shift <= 0) {
            // k >= M (shouldn't happen for e4<1, but guard): normalizes up
            uint32_t k = M << (-shift);
            // k has hidden bit at position 3; extract m3
            uint32_t m3 = (k >> 3) & 0x7;
            out = (sign << 7) | m3;              // exp=0 subnormal
            (void)m3;
        } else {
            // round M to 3 fraction bits (4-bit result including hidden) at position shift
            uint32_t keep = M >> shift;          // 4-bit field: hidden at bit 3, m3 at bits 2:0
            uint32_t rem = M & ((1u << shift) - 1);
            uint32_t half = 1u << (shift - 1);
            if (rem > half || (rem == half && (keep & 1))) keep += 1;  // round-to-nearest-even
            if (keep >= 16) {
                // carried into normal range (hidden bit overflow -> exp becomes 1)
                uint32_t m3 = (keep >> 1) & 0x7;
                out = (sign << 7) | (1u << 3) | m3;
            } else {
                // subnormal: exp=0, mantissa = keep & 7 (hidden bit is the "8" which makes it subnormal k)
                uint32_t m3 = keep & 0x7;
                if (keep == 0) out = (sign << 7);   // flushed to 0
                else out = (sign << 7) | m3;
            }
        }
    }
    return (uint8_t)out;
}
__device__ __forceinline__ float e4m3_to_f32(uint8_t b){
    uint32_t sign = (b >> 7) & 1;
    uint32_t e    = (b >> 3) & 0xF;
    uint32_t m    = b & 0x7;
    if (e == 0) {
        if (m == 0) return sign ? -0.f : 0.f;
        // subnormal e4m3 -> value = m * 2^-9
        float v = m * 0.001953125f;
        return sign ? -v : v;
    }
    if (e == 15 && m == 7) return sign ? -448.f : 448.f; // max
    uint32_t out = (sign << 31) | ((e + 127 - 7) << 23) | (m << 20);
    float f;
    memcpy(&f, &out, 4);
    return f;
}
