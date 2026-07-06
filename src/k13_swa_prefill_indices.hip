// k13_swa_prefill_indices.hip - swa_prefill_indices (TileLang)

#include "common.hip"

// ============================================================
// 13. swa_prefill_indices (TileLang)
//   seq_lens_k:[batch], seq_lens_q:[batch], cu_seqlens_q:[batch+1]
//   out:[num_q_tokens, window] int32 (per-query KV index array, -1 pad)
//   grid=ceil(num_q_tokens/4), block=128 (4 warps); one warp per query token.
// ============================================================
__global__ void swa_prefill_indices_kernel(
    int* indices, const int* seq_lens_k, const int* seq_lens_q,
    const int* cu_seqlens_q, int num_q_tokens, int batch, int window
) {
    const int wpb = blockDim.x >> 5;
    const int gwp = (blockIdx.x * wpb) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (gwp >= num_q_tokens) return;
    // find batch_id: largest b with cu_seqlens_q[b] <= gwp
    __shared__ int s_bid[8];
    int bid = -1;
    for (int b = lane; b < batch; b += 32) {
        if (cu_seqlens_q[b] <= gwp && gwp < cu_seqlens_q[b + 1]) bid = b;
    }
    // reduce: any lane that found a bid -> take min valid (unique)
    bid = (bid < 0) ? 0x7fffffff : bid;
    int mbid = bid;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        int v = __shfl_xor(bid, o);
        if (v < mbid) mbid = v;
    }
    if (lane == 0) s_bid[0] = (mbid == 0x7fffffff) ? 0 : mbid;
    __syncthreads();
    int seq_idx = s_bid[0];
    int kv_len = seq_lens_k[seq_idx];
    int qo_len = seq_lens_q[seq_idx];
    int cum_qo_len = cu_seqlens_q[seq_idx];
    int prefix_len = kv_len - qo_len;
    int curr_qo = gwp - cum_qo_len;
    int end_abs = prefix_len + curr_qo + 1;
    int start_abs = max(end_abs - window, 0);
    int old_kv_start = batch * window;            // old KV pool (prefix) per-seq window slots
    int new_kv_start = batch * window + cum_qo_len;
    int* out_row = indices + gwp * window;
    for (int j = lane; j < window; j += 32) {
        int abs_pos = start_abs + j;
        int val;
        if (abs_pos >= end_abs) val = -1;
        else if (abs_pos < prefix_len) val = old_kv_start + (abs_pos % window);
        else val = new_kv_start + (abs_pos - prefix_len);
        out_row[j] = val;
    }
}

extern "C" {
void launch_swa_prefill_indices(void* indices,const void* seq_lens_k,const void* seq_lens_q,
    const void* cu_seqlens_q,int num_q_tokens,int batch,int window,hipStream_t st){
    int warps_per_block = 4;
    int blocks = (num_q_tokens + warps_per_block - 1) / warps_per_block;
    swa_prefill_indices_kernel<<<blocks,warps_per_block*32,0,st>>>((int*)indices,
        (const int*)seq_lens_k,(const int*)seq_lens_q,(const int*)cu_seqlens_q,
        num_q_tokens,batch,window);}
}
