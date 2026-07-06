// k11_mhc_pre.hip - mhc_pre (TileLang, sigmoid+mix)

#include "common.hip"

// ============================================================
// 11. mhc_pre (sigmoid + mix)
//   output_mix[n,j] = sigmoid(input_mix[n,j]*mhc_scale[0] + mhc_base[j]) + eps
// ============================================================
template<int BS>
__global__ void mhc_pre_kernel(
    const __hip_bfloat16* input_mix, const float* mhc_scale, const float* mhc_base,
    __hip_bfloat16* output_mix, int num_tokens, int mhc_mult, float eps
) {
    int row = blockIdx.x;
    if (row >= num_tokens) return;
    int tid = threadIdx.x;
    for (int j = tid; j < mhc_mult; j += BS) {
        float v = b2f(input_mix[row * mhc_mult + j]);
        float result = 1.0f / (1.0f + expf(-(v * mhc_scale[0] + mhc_base[j]))) + eps;
        output_mix[row * mhc_mult + j] = f2b(result);
    }
}

extern "C" {
void launch_mhc_pre(const void* input_mix,const void* mhc_scale,const void* mhc_base,
    void* output_mix,int num_tokens,int mhc_mult,hipStream_t st){
    constexpr int BS=128;
    mhc_pre_kernel<BS><<<num_tokens,BS,0,st>>>((const __hip_bfloat16*)input_mix,
        (const float*)mhc_scale,(const float*)mhc_base,(__hip_bfloat16*)output_mix,num_tokens,mhc_mult,1e-6f);}
}
