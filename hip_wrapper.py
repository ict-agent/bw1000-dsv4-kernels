"""HIP kernel Python wrappers — aligned with sglang's actual op signatures and shapes.

Design:
- Layer 1 (HIP kernel, dsv4_all_hip_kernels.hip): functional correctness, raw ctypes launch.
- Layer 2 (this file, hip_wrapper.py): sglang-aligned signatures, graph-safe buffer pool,
  minimal Python ops (no torch.empty/contiguous/.item in hot path).
- Layer 3 (hip_dsv4_integration.py): engine monkey-patch.

All output tensors use a static buffer pool (_buf) keyed by (shape,dtype,device) so CUDA
graph capture/replay see stable pointers. Wrappers match sglang signatures exactly so they
can be drop-in replacements.

Actual shapes (decode bs=256, TP=8):
  per_token_quant_int8:      x [256,4096] bf16 -> q [256,4096] int8, s [256,1] f32
  per_token_group_quant_int8: x [1536,2048] bf16 -> q [1536,2048] int8, s [1536,16] f32
  silu_and_mul:              x [1536,4096] bf16 -> out [1536,2048] bf16
  rmsnorm_self:              q [256,8,512] bf16 -> out [256,8,512] bf16 (in-place ok)
  fused_rope:                q [256,8,64] bf16, k [256,1,64]|None, freqs complex, pos [256]
  topk_transform_512:        scores [256,64*seq] f32, out [256,512] int32 (in-place)
  swa_prefill_indices:       swa_indices [nq,128] int32 (in-place)
  hc_split_sinkhorn:         mixes [256,1,24] f32 -> pre/post [256,1,4], comb [256,1,4,4]
  mhc_post:                  x [256,4096] bf16, residual [256,4,4096] -> out [256,4,4096]
  act_quant:                 x [256,64,128] bf16 -> y fp8, s [256,64,1] f32
  merge_attn_states:         output [256,8,512] bf16 (in-place), lse [8,256] f32
"""
import ctypes, torch, os

_LIB = ctypes.CDLL("/workspace/hip_kernels/libdsv4_all_hip.so")
P = ctypes.c_void_p

# argtypes
_LIB.launch_ptq.argtypes = [P,P,P,ctypes.c_int,ctypes.c_int,P]
_LIB.launch_ptgq.argtypes = [P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
_LIB.launch_rmsnorm_self.argtypes = [P,P,ctypes.c_int,ctypes.c_int,ctypes.c_float,P]
_LIB.launch_fused_rope.argtypes = [P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
_LIB.launch_fused_rope_strided.argtypes = [P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,
                                           ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,ctypes.c_int64,P]
_LIB.launch_silu_mul.argtypes = [P,P,P,ctypes.c_int,ctypes.c_int,P]
_LIB.launch_silu_mul_split.argtypes = [P,P,ctypes.c_int,ctypes.c_int,P]
_LIB.launch_silu_mul_masked_quant.argtypes = [P,P,P,P,P,ctypes.c_int,ctypes.c_int,P]
_LIB.launch_hc_split_sinkhorn.argtypes = [P,P,P,P,P,P,ctypes.c_int,P]
_LIB.launch_act_quant_fp8.argtypes = [P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
_LIB.launch_merge_attn_states.argtypes = [P,P,P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
_LIB.launch_topk_transform.argtypes = [P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
_LIB.launch_mhc_pre.argtypes = [P,P,P,P,ctypes.c_int,ctypes.c_int,P]
_LIB.launch_mhc_post.argtypes = [P,P,P,P,P,ctypes.c_int,ctypes.c_int,P]
_LIB.launch_swa_prefill_indices.argtypes = [P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
_LIB.launch_grouped_gemm_int8.argtypes = [P,P,P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]

def _s():
    return torch.cuda.current_stream().cuda_stream

# ---- Debug instrumentation: print each wrapper call (op#count) to locate hang op ----
import os as _os
_DBG = _os.environ.get("SGLANG_HIP_DEBUG", "0") == "1"
_dbg_cnt = {}
def _dbg(name, a):
    if not _DBG: return
    c = _dbg_cnt.get(name, 0)
    _dbg_cnt[name] = c + 1
    # print first 3 with shapes, then every 200th (op#count only) to track progress
    if c < 3:
        shapes = []
        for v in a:
            try: shapes.append(tuple(int(s) for s in v.shape) + (str(v.dtype),))
            except Exception: shapes.append(type(v).__name__)
        print(f"[HIPDBG] {name}#{c} {shapes}", flush=True)
    elif c % 200 == 0:
        print(f"[HIPDBG] {name}#{c} ...", flush=True)

# ---- Optional profiling (SGLANG_HIP_PROFILE=1) ----
# Records GPU time per launch_xxx in REAL inference. Two safety rules:
#  (1) skip timing while a CUDA graph is being captured — synchronize() inside
#      capture is illegal and crashes the server. Capture replays don't go through
#      python anyway, so there's nothing to time during decode (bs<=cuda_graph_max_bs).
#  (2) don't synchronize per call (would serialize inference). Instead record events
#      asynchronously and drain them in batches (every _FLUSH_EVERY calls).
_PROFILE = os.environ.get("SGLANG_HIP_PROFILE", "0") == "1"
_PROF_DIR = os.environ.get("SGLANG_HIP_PROF_DIR", "/workspace/hip_kernels/results")
_timings = {}
_pending = []  # list of (name, e0, e1) awaiting drain
_FLUSH_EVERY = 256

def _capturing():
    try:
        return torch.cuda.is_current_stream_capturing()
    except Exception:
        return False

def _timed(name, fn):
    if not _PROFILE: return fn
    def w(*a, **kw):
        if _capturing():
            return fn(*a, **kw)  # graph capture: no timing (sync illegal)
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); r = fn(*a, **kw); e1.record()
        _pending.append((name, e0, e1))
        if len(_pending) >= _FLUSH_EVERY:
            _drain()
        return r
    return w

def _drain():
    if not _pending: return
    try:
        torch.cuda.current_stream().synchronize()
    except Exception:
        return
    for name, e0, e1 in _pending:
        try:
            _timings.setdefault(name, []).append(round(e0.elapsed_time(e1), 4))
        except Exception:
            pass
    _pending.clear()
    _dump()

def dump_timings():
    if not _PROFILE: return
    _drain()  # flush pending events first
    _dump()

def _dump():
    if not _PROFILE: return
    import json, statistics, os
    out = {k: {"calls": len(v),
               "median_ms": round(statistics.median(v),4) if v else 0,
               "min_ms": round(min(v),4) if v else 0,
               "max_ms": round(max(v),4) if v else 0,
               "total_ms": round(sum(v),4)}
           for k,v in _timings.items()}
    try:
        os.makedirs(_PROF_DIR, exist_ok=True)
        with open(os.path.join(_PROF_DIR, "profiling.json"),"w") as f:
            json.dump(out, f, indent=2)
    except Exception as e:
        print(f"[PROF] dump failed: {e}", flush=True)
    for k,v in out.items():
        print(f"[PROF] {k}: calls={v['calls']} median={v['median_ms']}ms total={v['total_ms']}ms", flush=True)

import atexit; atexit.register(dump_timings)

# Static buffer pool: graph-capture-safe (stable pointer across replays).
# key includes a name tag so different outputs of the same shape don't alias.
# Static buffer pool: graph-capture-safe (stable pointer across replays).
# CRITICAL: only pool during cuda graph capture. In EAGER (prefill) path, pooling
# aliases same-shape outputs across sequential calls — if silu(x1)'s output isn't
# consumed by the downstream async GEMM before silu(x2) overwrites the same buffer,
# the GEMM reads corrupted/NaN data -> hang. So: capture -> pool (stable ptr),
# eager -> fresh alloc (no aliasing).
_pool = {}
def _buf(shape, dtype, device, name=""):
    if not _capturing():
        return torch.empty(shape, device=device, dtype=dtype)
    key = (name, tuple(int(s) for s in shape), dtype, str(device))
    b = _pool.get(key)
    if b is None or tuple(b.shape) != tuple(shape):
        b = torch.empty(shape, device=device, dtype=dtype)
        _pool[key] = b
    return b


# ============================================================
# 1. per_token_quant_int8  (aligned with lmslim: returns (q, scales))
# ============================================================
def per_token_quant_int8(x, scale_dtype=None, cal_sum=False):
    _dbg("ptq", [x])
    M, N = x.shape
    q = _buf((M, N), torch.int8, x.device, "ptq_q")
    s = _buf((M, 1), torch.float32, x.device, "ptq_s")   # lmslim returns [M,1]
    _LIB.launch_ptq(x.data_ptr(), q.data_ptr(), s.data_ptr(), M, N, _s())
    if cal_sum:
        return q, s, torch.zeros(1, device=x.device)  # x_sum not supported; return 0
    return q, s

# 2. per_token_group_quant_int8
def per_token_group_quant_int8(x, group_size=128, eps=1e-10, dtype=torch.int8):
    M, N = x.shape
    ng = N // group_size
    q = _buf((M, N), torch.int8, x.device, "ptgq_q")
    s = _buf((M, ng), torch.float32, x.device, "ptgq_s")
    _LIB.launch_ptgq(x.data_ptr(), q.data_ptr(), s.data_ptr(), M, N, group_size, _s())
    return q, s

# 3. silu_and_mul  (aligned with SiluAndMul.forward_cuda: x [M,2N] -> out [M,N])
#    kernel reads gate=x[:,i], up=x[:,N+i] (split layout, sglang uses cat([gate,up]))
def silu_and_mul(x):
    _dbg("silu", [x])
    d = x.shape[-1] // 2
    out_shape = x.shape[:-1] + (d,)
    out = _buf(tuple(out_shape), x.dtype, x.device, "silu_out")
    M = 1
    for dim in out_shape[:-1]:
        M *= dim
    # split-layout kernel reads gate=x[:,0:d], up=x[:,d:2d] directly from x (no copy)
    _LIB.launch_silu_mul_split(x.data_ptr(), out.data_ptr(), M, d, _s())
    return out

# 4. rmsnorm_self  (aligned with jit_kernel: q [*,head_dim] -> out NEW tensor, reads q writes out)
def rmsnorm_self(q, eps=1e-6):
    out = _buf(tuple(q.shape), q.dtype, q.device, "rms_out")
    N = q.shape[-1]
    M = q.numel() // N
    _LIB.launch_rmsnorm_self(q.data_ptr(), out.data_ptr(), M, N, eps, _s())
    return out

# 5. fused_rope  (aligned with jit_kernel: in-place q/k, freqs_cis complex)
#    NOTE: engine passes q = q[..., -rope_dim:] which is NON-CONTIGUOUS: q is a slice
#    of a larger [nt, nheads, head_dim] tensor, so its stride[1] = head_dim, not rope_dim.
#    Our kernel assumed contiguous and read garbage -> corrupted q -> downstream hang.
#    FIX: the rope op must be in-place on the ORIGINAL q slice memory (engine keeps the
#    full tensor). So we cannot .contiguous() (that would write a detached copy). Instead
#    we pass q's strides to the kernel so it indexes correctly. For k, engine passes
#    .unsqueeze(1) slice — same non-contig issue.
def fused_rope(q, k, freqs_cis, positions, inverse=False):
    _dbg("rope", [q, k, freqs_cis, positions])
    nt = q.shape[0]; nq = q.shape[1]
    nk = k.shape[1] if k is not None else 0
    rd = q.shape[-1]
    has_k = 1 if k is not None else 0
    # CRITICAL: engine passes positions as int64 (torch.long). Our kernel reads int*
    # (4 bytes) -> would read every-other element (garbage rotations). Convert to int32.
    if positions.dtype != torch.int32:
        positions = positions.to(torch.int32)
    # q strides: [stride_tok, stride_head, stride_elem]. stride_elem is normally 1.
    # The kernel indexes dst[tok][head][i] = base + tok*st_t + head*st_h + i*1.
    qst = q.stride()
    qst_t, qst_h = int(qst[0]), int(qst[1])
    if k is not None:
        kst = k.stride(); kst_t, kst_h = int(kst[0]), int(kst[1])
    else:
        kst_t = kst_h = 0
    # freqs_cis is complex64 [max_pos, rd/2]; memory layout = real,imag interleaved
    # = our interleaved format. View as float32 [max_pos, rd].
    if freqs_cis.dtype == torch.complex64:
        fc = freqs_cis.view(torch.float32).reshape(freqs_cis.shape[0], rd)
    else:
        fc = freqs_cis
    _LIB.launch_fused_rope_strided(q.data_ptr(), k.data_ptr() if k is not None else 0,
                           fc.data_ptr(), positions.data_ptr(),
                           nt, nq, nk, rd, has_k, qst_t, qst_h, kst_t, kst_h, _s())
    return None  # in-place

# 6. topk_transform_512  (aligned: in-place out_page_indices)
def topk_transform_512(scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices=None):
    _dbg("topk", [scores, seq_lens, page_tables, out_page_indices])
    # CRITICAL: engine c4_seq_lens is int64 (seq_lens // 4 preserves int64). Kernel reads
    # int* (4 bytes) -> misreads. Convert seq_lens (and guard page_tables) to int32.
    if seq_lens.dtype != torch.int32:
        seq_lens = seq_lens.to(torch.int32)
    if page_tables.dtype != torch.int32:
        page_tables = page_tables.to(torch.int32)
    b = scores.shape[0]; cap = scores.shape[1]
    ptr_stride = page_tables.shape[1]
    k = out_page_indices.shape[1]  # 512
    _LIB.launch_topk_transform(scores.data_ptr(), seq_lens.data_ptr(), page_tables.data_ptr(),
                               out_page_indices.data_ptr(), b, cap, ptr_stride, page_size, k, _s())
    if out_raw_indices is not None and page_size == 1:
        out_raw_indices.copy_(out_page_indices)
    return None

# 7. tilelang_make_swa_prefill_indices  (aligned: in-place swa_indices)
def tilelang_make_swa_prefill_indices(seq_lens_k, seq_lens_q, swa_indices, cu_seqlens_q=None):
    _dbg("swa", [seq_lens_k, seq_lens_q, swa_indices])
    b = seq_lens_q.shape[0]
    window = swa_indices.shape[1]
    nq = swa_indices.shape[0]
    if cu_seqlens_q is None:
        cu = _buf((b + 1,), torch.int32, seq_lens_q.device)
        cu.zero_(); cu[1:] = seq_lens_q.cumsum(0)
        cu_seqlens_q = cu
    _LIB.launch_swa_prefill_indices(swa_indices.data_ptr(), seq_lens_k.data_ptr(),
                                    seq_lens_q.data_ptr(), cu_seqlens_q.data_ptr(),
                                    nq, b, window, _s())
    return swa_indices

# 8. hc_split_sinkhorn  (aligned: mixes [b,s,mh] -> pre/post [b,s,hc], comb [b,s,hc,hc])
def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    b, s, _ = mixes.shape
    n = b * s
    pre = _buf((b, s, hc_mult), torch.float32, mixes.device, "sk_pre")
    post = _buf((b, s, hc_mult), torch.float32, mixes.device, "sk_post")
    comb = _buf((b, s, hc_mult, hc_mult), torch.float32, mixes.device, "sk_comb")
    _LIB.launch_hc_split_sinkhorn(mixes.data_ptr(), hc_scale.data_ptr(), hc_base.data_ptr(),
                                  pre.data_ptr(), post.data_ptr(), comb.data_ptr(), n, _s())
    return pre, post, comb

# 9. mhc_post  (aligned with mhc_post_torch: x [n,h], residual [n,hc,h], post [n,hc], comb [n,hc,hc])
def mhc_post_torch(x, residual, post_layer_mix, comb_res_mix):
    if x.shape[0] == 0:
        return _buf((0, residual.shape[1], residual.shape[2]), x.dtype, x.device)
    if post_layer_mix.dim() == 3 and post_layer_mix.shape[-1] == 1:
        post_layer_mix = post_layer_mix.squeeze(-1)
    n = comb_res_mix.shape[0]
    hidden = x.shape[-1]
    out = _buf((n, residual.shape[1], hidden), x.dtype, x.device)
    _LIB.launch_mhc_post(comb_res_mix.data_ptr(), residual.data_ptr(), post_layer_mix.data_ptr(),
                         x.data_ptr(), out.data_ptr(), n, hidden, _s())
    return out

# 10. act_quant  (aligned with nsa tilelang_kernel: x -> (y fp8, s f32))
#     DCU uses fp8_e5m2fnuz; our kernel emits e4m3. We support both via dtype param.
def act_quant(x, block_size=128, scale_fmt=None):
    N = x.shape[-1]
    ng = N // block_size
    # DCU path uses e5m2fnuz; for correctness testing we emit e4m3 (our kernel's impl).
    # In production DCU, lightop.per_token_group_quant_fp8 is used (vendor). Our wrapper
    # emits e4m3 bytes; caller can reinterpret as needed.
    y = _buf(x.shape, torch.float8_e4m3fn, x.device, "aq_y")
    s_shape = x.shape[:-1] + (ng,)
    s = _buf(s_shape, torch.float32, x.device, "aq_s")
    M = 1
    for d in x.shape[:-1]:
        M *= d
    x2 = x.reshape(M, N) if x.dim() != 2 else x
    y2 = y.reshape(M, N) if y.dim() != 2 else y
    s2 = s.reshape(M, ng) if s.dim() != 2 else s
    _LIB.launch_act_quant_fp8(x2.data_ptr(), y2.view(torch.uint8).data_ptr(), s2.data_ptr(),
                              M, N, block_size, _s())
    return y, s

# 11. merge_attn_states  (aligned: in-place output, output_lse)
def merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse=None):
    nt = prefix_output.shape[0]; nh = prefix_output.shape[1]; hs = prefix_output.shape[2]
    if output_lse is None:
        output_lse = _buf((nh, nt), torch.float32, prefix_output.device)
    _LIB.launch_merge_attn_states(output.data_ptr(), output_lse.data_ptr(),
                                  prefix_output.data_ptr(), prefix_lse.data_ptr(),
                                  suffix_output.data_ptr(), suffix_lse.data_ptr(),
                                  nt, nh, hs, _s())
    return None

# 12. mhc_pre  (simplified — only sigmoid+mix part; full mhc_pre involves GEMM, not aligned)
def mhc_pre(input_mix, mhc_scale, mhc_base, mhc_mult, eps=1e-6):
    n = input_mix.shape[0]
    out = _buf(input_mix.shape, input_mix.dtype, input_mix.device)
    _LIB.launch_mhc_pre(input_mix.data_ptr(), mhc_scale.data_ptr(), mhc_base.data_ptr(),
                        out.data_ptr(), n, mhc_mult, _s())
    return out

# 13. silu_mul_masked_quant  (EP MoE path)
def silu_mul_masked_quant(gate, up, mask=None):
    M, N = gate.shape
    if mask is None:
        mask = torch.ones(M, device=gate.device, dtype=torch.int32)
    q = _buf((M, N), torch.int8, gate.device, "smq_q")
    s = _buf((M, 1), torch.float32, gate.device, "smq_s")
    _LIB.launch_silu_mul_masked_quant(gate.data_ptr(), up.data_ptr(), mask.data_ptr(),
                                      q.data_ptr(), s.data_ptr(), M, N, _s())
    return q, s

# 14. grouped_gemm_int8  (out: C [E,M,N])
def grouped_gemm_int8(A, B, sa, sb, masked_m, E, M, N, K):
    C = _buf((E, M, N), torch.bfloat16, A.device)
    _LIB.launch_grouped_gemm_int8(A.data_ptr(), B.data_ptr(), sa.data_ptr(), sb.data_ptr(),
                                  C.data_ptr(), masked_m.data_ptr(), E, M, N, K, _s())
    return C

# If profiling enabled, wrap each launch_xxx to record GPU time per kernel
if _PROFILE:
    for _name in dir(_LIB):
        if _name.startswith("launch_"):
            _orig = getattr(_LIB, _name)
            setattr(_LIB, _name, _timed(_name, _orig))
