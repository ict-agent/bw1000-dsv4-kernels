#include "common.hip"

// 10. topk_transform_512 (jit_kernel topk.cuh)
//   scores:[batch, seq_len_cap] fp32; seq_lens:[batch] int32;
//   page_table:[batch, page_table_stride] int32; out_indices:[batch, 512] int32;
//   page_size power-of-2. page_to_indices(i) = (page_table[i>>page_bits] << page_bits) | (i & mask).
//   Fast path (seq_len<=512): indices[t<seq_len]=page_to_indices(t); indices[t>=seq_len]=-1.
//   Long path (seq_len>512): radix top-512 by score (descending), then page-transform.
// ============================================================
__device__ __forceinline__ int page_to_indices(const int* page_table, int i, int page_bits) {
    int page = page_table[i >> page_bits];
    int mask = (1 << page_bits) - 1;
    return (page << page_bits) | (i & mask);
}

// float -> order-preserving uint32 key (monotonic) for radix
__device__ __forceinline__ uint32_t f32_to_key(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    uint32_t mask = (u >> 31) ? 0xFFFFFFFFu : 0x80000000u;
    return u ^ mask;
}

// Block-wide top-k via radix select: find the k-th largest key (threshold) in 4
// passes of 8-bit radix (MSB first), then scan once to select all keys > threshold
// plus enough ==threshold to reach k. O(5*sl) work, ~100x faster than k-round argmax.
// shared layout (caller-provided): keys[sl] uint32 + hist[258] int.
__device__ __forceinline__ void topk_radix(
    const float* sc, int sl, int k_top, int* sel,
    uint32_t* smem_keys, int* smem_hist
) {
    int tid = threadIdx.x;
    // load order-preserving keys
    for (int i = tid; i < sl; i += blockDim.x) smem_keys[i] = f32_to_key(sc[i]);
    __syncthreads();

    uint32_t lo = 0u;            // current lower bound of threshold prefix (bytes fixed so far)
    int need = k_top;            // remaining slots
    // smem_hist[256]=t, [257]=cum_gt, [258..]=unused
    for (int pass = 0; pass < 4 && need > 0; pass++) {
        int shift = (3 - pass) * 8;
        // bytes above the current byte (already fixed). pass 0 -> no prefix.
        uint32_t prefix_mask = (pass == 0) ? 0u : (0xFFFFFFFFu << (shift + 8));
        // zero histogram 256 bins
        for (int i = tid; i < 256; i += blockDim.x) smem_hist[i] = 0;
        __syncthreads();
        // histogram current byte over candidates whose fixed prefix matches lo
        for (int i = tid; i < sl; i += blockDim.x) {
            uint32_t key = smem_keys[i];
            if ((key & prefix_mask) == (lo & prefix_mask)) {
                int b = (key >> shift) & 0xFF;
                atomicAdd(&smem_hist[b], 1);
            }
        }
        __syncthreads();
        // find threshold byte t: largest t with count(byte>t) < need
        if (tid == 0) {
            int cum = 0, t = 0, cgt = 0;
            for (int b = 255; b >= 0; b--) {
                cum += smem_hist[b];
                if (cum >= need) { t = b; cgt = cum - smem_hist[b]; break; }
            }
            smem_hist[256] = t;
            smem_hist[257] = cgt;
        }
        __syncthreads();
        int t = smem_hist[256];
        int cgt = smem_hist[257];
        need -= cgt;                        // byte>t candidates confirmed in
        // fix this byte of threshold to t
        uint32_t byte_mask = 0xFFu << shift;
        lo = (lo & ~byte_mask) | ((uint32_t)t << shift);
    }
    uint32_t threshold = lo;

    // stage 2: count key>threshold and key==threshold
    if (tid == 0) { smem_hist[256] = 0; smem_hist[257] = 0; }
    __syncthreads();
    for (int i = tid; i < sl; i += blockDim.x) {
        uint32_t key = smem_keys[i];
        if (key > threshold) atomicAdd(&smem_hist[256], 1);
        else if (key == threshold) atomicAdd(&smem_hist[257], 1);
    }
    __syncthreads();
    int n_gt = smem_hist[256];
    int n_eq = smem_hist[257];
    int need_eq = k_top - n_gt;
    if (need_eq < 0) need_eq = 0;
    if (need_eq > n_eq) need_eq = n_eq;

    // fill sel: positions [0,n_gt) with key>threshold, [n_gt, n_gt+need_eq) with key==threshold
    if (tid == 0) { smem_hist[256] = 0; smem_hist[257] = 0; }  // pos counters
    __syncthreads();
    for (int i = tid; i < sl; i += blockDim.x) {
        uint32_t key = smem_keys[i];
        if (key > threshold) {
            int p = atomicAdd(&smem_hist[256], 1);
            if (p < n_gt) sel[p] = i;
        } else if (need_eq > 0 && key == threshold) {
            int p = atomicAdd(&smem_hist[257], 1);
            if (p < need_eq) sel[n_gt + p] = i;
        }
    }
    __syncthreads();
    int total = n_gt + need_eq;
    for (int i = tid; i < k_top; i += blockDim.x) {
        if (i >= total) sel[i] = -1;   // pad if under-filled
    }
}

// ============================================================
// 10. topk_transform_512  (256 threads/block — gfx936 max 256)
// ============================================================
__global__ void topk_transform_kernel(
    const float* scores, const int* seq_lens, const int* page_tables,
    int* out_indices, int batch, int seq_len_cap, int page_table_stride,
    int page_size, int k_top
) {
    int b = blockIdx.x;
    if (b >= batch) return;
    int tid = threadIdx.x;
    int sl = seq_lens[b];
    if (sl > seq_len_cap) sl = seq_len_cap;
    const float* sc = scores + b * seq_len_cap;
    const int* pt = page_tables + b * page_table_stride;
    int* out = out_indices + b * k_top;
    int page_bits = 0; { int ps = page_size; while (ps > 1) { ps >>= 1; page_bits++; } }

    if (sl <= 0) {
        for (int i = tid; i < k_top; i += blockDim.x) out[i] = -1;
        return;
    }

    if (sl <= k_top) {
        // fast path: take all valid indices in ascending order, pad -1
        for (int i = tid; i < k_top; i += blockDim.x) {
            if (i < sl) out[i] = page_to_indices(pt, i, page_bits);
            else out[i] = -1;
        }
        return;
    }

    // long path: radix select top-k
    extern __shared__ unsigned char smem[];
    uint32_t* s_keys = (uint32_t*)smem;                    // [sl]
    int*      s_hist = (int*)(s_keys + sl);               // [259]
    int*      sel    = s_hist + 259;                        // [k_top]
    topk_radix(sc, sl, k_top, sel, s_keys, s_hist);
    __syncthreads();
    // page-transform each selected logical index
    for (int i = tid; i < k_top; i += blockDim.x) {
        int s = sel[i];
        out[i] = (s < 0) ? -1 : page_to_indices(pt, s, page_bits);
    }
}


extern "C" {
void launch_topk_transform(const void* scores,const void* seq_lens,const void* page_tables,
    void* out,int batch,int seq_len_cap,int page_table_stride,int page_size,int k_top,hipStream_t st){
    // shared: keys[cap] uint32 + hist[259] int + sel[k_top] int
    int smem = seq_len_cap * 4 + 259 * 4 + k_top * 4;
    topk_transform_kernel<<<batch,256,smem,st>>>(
        (const float*)scores,(const int*)seq_lens,(const int*)page_tables,(int*)out,
        batch,seq_len_cap,page_table_stride,page_size,k_top);}
}
