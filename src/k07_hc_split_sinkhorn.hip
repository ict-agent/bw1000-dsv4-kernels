// k07_hc_split_sinkhorn.hip - hc_split_sinkhorn (TileLang mhc.py)

#include "common.hip"

// ============================================================
// 7. hc_split_sinkhorn (TileLang mhc.py)
//   mixes: [n, mix_hc=24] fp32; hc_scale:[3] fp32; hc_base:[24] fp32
//   pre:[n,4] fp32; post:[n,4] fp32; comb:[n,4,4] fp32
//   pre[j]  = sigmoid(mixes[j]*hc_scale[0]   + hc_base[j])        + eps
//   post[j] = 2*sigmoid(mixes[j+hc]*hc_scale[1] + hc_base[j+hc])
//   comb[j,k]= mixes[j*hc+k+2*hc]*hc_scale[2] + hc_base[j*hc+k+2*hc]
//   Sinkhorn (iters=20): first round exp(comb-rowmax) then /=rowsum+eps then /=colsum+eps;
//                        subsequent rounds: /=rowsum+eps, /=colsum+eps.
//   hc=4 (compile-time). One block per token; thread 0 computes the 4x4, all threads write.
// ============================================================
template<int HC, int ITERS>
__global__ void sinkhorn_kernel(
    const float* mixes, const float* hc_scale, const float* hc_base,
    float* pre, float* post, float* comb, int n, float eps
) {
    int row = blockIdx.x;
    if (row >= n) return;
    int tid = threadIdx.x;
    const int mix_hc = (2 + HC) * HC; // 24

    __shared__ float s_mix[24];
    __shared__ float s_comb[16];      // 4x4
    __shared__ float s_pre[4], s_post[4];

    if (tid < mix_hc) s_mix[tid] = mixes[row * mix_hc + tid];
    __syncthreads();

    if (tid == 0) {
        float sc0 = hc_scale[0], sc1 = hc_scale[1], sc2 = hc_scale[2];
        #pragma unroll
        for (int j = 0; j < HC; j++) {
            s_pre[j]  = 1.f / (1.f + expf(-(s_mix[j] * sc0 + hc_base[j]))) + eps;
            s_post[j] = 2.f * (1.f / (1.f + expf(-(s_mix[j + HC] * sc1 + hc_base[j + HC]))));
        }
        // comb[j][k]
        #pragma unroll
        for (int j = 0; j < HC; j++)
            #pragma unroll
            for (int k = 0; k < HC; k++) {
                int idx = j * HC + k + 2 * HC;
                s_comb[j * HC + k] = s_mix[idx] * sc2 + hc_base[idx];
            }
        // Sinkhorn first round: exp(comb - rowmax), /= rowsum+eps, /= colsum+eps
        #pragma unroll
        for (int j = 0; j < HC; j++) {
            float rmax = -FLT_MAX;
            #pragma unroll
            for (int k = 0; k < HC; k++) rmax = fmaxf(rmax, s_comb[j*HC+k]);
            #pragma unroll
            for (int k = 0; k < HC; k++) s_comb[j*HC+k] = expf(s_comb[j*HC+k] - rmax);
        }
        #pragma unroll
        for (int j = 0; j < HC; j++) {
            float rsum = 0.f;
            #pragma unroll
            for (int k = 0; k < HC; k++) rsum += s_comb[j*HC+k];
            rsum += eps;
            #pragma unroll
            for (int k = 0; k < HC; k++) s_comb[j*HC+k] /= rsum;
        }
        #pragma unroll
        for (int k = 0; k < HC; k++) {
            float csum = 0.f;
            #pragma unroll
            for (int j = 0; j < HC; j++) csum += s_comb[j*HC+k];
            csum += eps;
            #pragma unroll
            for (int j = 0; j < HC; j++) s_comb[j*HC+k] /= csum;
        }
        // subsequent rounds
        for (int it = 1; it < ITERS; it++) {
            #pragma unroll
            for (int j = 0; j < HC; j++) {
                float rsum = 0.f;
                #pragma unroll
                for (int k = 0; k < HC; k++) rsum += s_comb[j*HC+k];
                rsum += eps;
                #pragma unroll
                for (int k = 0; k < HC; k++) s_comb[j*HC+k] /= rsum;
            }
            #pragma unroll
            for (int k = 0; k < HC; k++) {
                float csum = 0.f;
                #pragma unroll
                for (int j = 0; j < HC; j++) csum += s_comb[j*HC+k];
                csum += eps;
                #pragma unroll
                for (int j = 0; j < HC; j++) s_comb[j*HC+k] /= csum;
            }
        }
    }
    __syncthreads();
    if (tid < HC) {
        pre[row * HC + tid]  = s_pre[tid];
        post[row * HC + tid] = s_post[tid];
    }
    if (tid < HC * HC) comb[row * HC * HC + tid] = s_comb[tid];
}

extern "C" {
void launch_hc_split_sinkhorn(const void* mixes,const void* hc_scale,const void* hc_base,
    void* pre,void* post,void* comb,int n,hipStream_t st){
    constexpr int HC=4, ITERS=20;
    sinkhorn_kernel<HC,ITERS><<<n,64,0,st>>>((const float*)mixes,(const float*)hc_scale,(const float*)hc_base,
        (float*)pre,(float*)post,(float*)comb,n,1e-6f);}
}
