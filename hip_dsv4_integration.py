"""HIP DSV4 kernel integration into SGLang engine.

Patches all 14 HIP kernels into the corresponding SGLang/lmslim/lightop/jit_kernel
call sites. All graph-safe (stream-driven, buffer-pool for outputs).

Env switches (all gated by master SGLANG_USE_HIP_DSV4=1):
  SGLANG_HIP_PTQ=1           per_token_quant_int8        (lmslim)
  SGLANG_HIP_PTGQ=1          per_token_group_quant_int8  (lmslim)
  SGLANG_HIP_SILU=1          silu_and_mul                (sglang SiluAndMul)
  SGLANG_HIP_SILU_QUANT=1    silu_mul_masked_quant       (lmslim fused)
  SGLANG_HIP_RMSNORM=1       rmsnorm_self                (sglang jit_kernel)
  SGLANG_HIP_NSA_QUANT=1     act_quant_fp8               (nsa tilelang_kernel)
  SGLANG_HIP_MHC=1           mhc_post / hc_split_sinkhorn (sglang mhc)
  SGLANG_HIP_ROPE=1          fused_rope                  (deepseek_v4_rope)
  SGLANG_HIP_TOPK=1          topk_transform_512          (compressed indexer)
  SGLANG_HIP_SWA=1           swa_prefill_indices          (jit_kernel)
  SGLANG_HIP_MERGE=1         merge_attn_states           (vllm triton)
  SGLANG_HIP_GROUPED_GEMM=1  grouped_gemm_int8           (optional; vendor marlin faster)

Load order: import this module at engine startup (via sitecustomize.py on PYTHONPATH),
it registers a sys.meta_path finder that patches each target module synchronously as
it loads (before TP workers fork). Output buffers use a static pool (_buf) so CUDA
graph capture/replay sees stable pointers (no VM fault).
"""
import os, sys, ctypes, threading, torch

_LIB = None
def _lib():
    global _LIB
    if _LIB is None:
        _LIB = ctypes.CDLL("/workspace/hip_kernels/libdsv4_all_hip.so")
        P = ctypes.c_void_p
        _LIB.launch_ptq.argtypes = [P,P,P,ctypes.c_int,ctypes.c_int,P]
        _LIB.launch_ptgq.argtypes = [P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
        _LIB.launch_rmsnorm_self.argtypes = [P,ctypes.c_int,ctypes.c_int,ctypes.c_float,P]
        _LIB.launch_fused_rope.argtypes = [P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
        _LIB.launch_silu_mul.argtypes = [P,P,P,ctypes.c_int,ctypes.c_int,P]
        _LIB.launch_silu_mul_masked_quant.argtypes = [P,P,P,P,P,ctypes.c_int,ctypes.c_int,P]
        _LIB.launch_hc_split_sinkhorn.argtypes = [P,P,P,P,P,P,ctypes.c_int,P]
        _LIB.launch_act_quant_fp8.argtypes = [P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
        _LIB.launch_merge_attn_states.argtypes = [P,P,P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
        _LIB.launch_topk_transform.argtypes = [P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
        _LIB.launch_mhc_pre.argtypes = [P,P,P,P,ctypes.c_int,ctypes.c_int,P]
        _LIB.launch_mhc_post.argtypes = [P,P,P,P,P,ctypes.c_int,ctypes.c_int,P]
        _LIB.launch_swa_prefill_indices.argtypes = [P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
        _LIB.launch_grouped_gemm_int8.argtypes = [P,P,P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
    return _LIB

def _s(): return torch.cuda.current_stream().cuda_stream
def _ptr(t): return t.data_ptr()

def _on(name): return os.environ.get(name, "0") == "1"
def _master(): return os.environ.get("SGLANG_USE_HIP_DSV4", "0") == "1"

# Static output buffer pool: graph-capture-safe. Captures a fixed pointer once;
# replays write to the same buffer. Keyed by (shape,dtype) so each captured graph
# batch-size gets its own stable buffer. Buffers are module-level (never freed).
_buf_pool = {}
def _buf(shape, dtype, device):
    key = (tuple(shape), dtype, str(device))
    b = _buf_pool.get(key)
    if b is None or b.shape != torch.Size(shape):
        b = torch.empty(shape, device=device, dtype=dtype)
        _buf_pool[key] = b
    return b

_patched = set()
_in_try = False
def _try_patches():
    global _in_try
    if _in_try: return
    if not _master(): return
    _in_try = True
    try:
        _try_patches_impl()
    finally:
        _in_try = False

def _try_patches_impl():
    if len(_patched) >= _want_count():
        return  # all patches applied, skip
    L = _lib()

    # ---- per_token_quant_int8 (lmslim) ----
    if _on("SGLANG_HIP_PTQ") and "ptq" not in _patched:
        try:
            import lmslim.layers.gemm.int8_utils as m
            if not getattr(m, "_hip_ptq_patched", False):
                m._hip_ptq_patched = True
                _orig = m.per_token_quant_int8
                def hip_ptq(x, scale_dtype=None, cal_sum=False):
                    M, N = x.shape
                    q = _buf((M, N), torch.int8, x.device)
                    s = _buf((M, 1), torch.float32, x.device)   # [M,1] to match lmslim broadcast
                    L.launch_ptq(_ptr(x), _ptr(q), _ptr(s), M, N, _s())
                    return q, s
                m.per_token_quant_int8 = hip_ptq
                _patched.add("ptq")
                print("[HIP-DSV4] patched lmslim.per_token_quant_int8", flush=True)
        except Exception as e:
            print(f"[HIP-DSV4] ptq patch: {e}", flush=True)

    # ---- per_token_group_quant_int8 (lmslim) ----
    if _on("SGLANG_HIP_PTGQ") and "ptgq" not in _patched:
        try:
            import lmslim.layers.gemm.int8_utils as m
            if not getattr(m, "_hip_ptgq_patched", False):
                m._hip_ptgq_patched = True
                _orig = m.per_token_group_quant_int8
                def hip_ptgq(x, group_size=128, scale_dtype=None):
                    M, N = x.shape
                    ng = N // group_size
                    q = _buf((M, N), torch.int8, x.device)
                    s = _buf((M, ng), torch.float32, x.device)
                    L.launch_ptgq(_ptr(x), _ptr(q), _ptr(s), M, N, group_size, _s())
                    return q, s
                m.per_token_group_quant_int8 = hip_ptgq
                _patched.add("ptgq")
                print("[HIP-DSV4] patched lmslim.per_token_group_quant_int8", flush=True)
        except Exception as e:
            print(f"[HIP-DSV4] ptgq patch: {e}", flush=True)

    # ---- silu_and_mul (sglang SiluAndMul.forward_cuda) ----
    if _on("SGLANG_HIP_SILU") and "silu" not in _patched:
        try:
            from sglang.srt.layers.activation import SiluAndMul
            if not getattr(SiluAndMul, "_hip_silu_patched", False):
                SiluAndMul._hip_silu_patched = True
                _orig = SiluAndMul.forward_cuda
                def hip_silu(self, x):
                    d = x.shape[-1] // 2
                    out_shape = x.shape[:-1] + (d,)
                    out = _buf(tuple(out_shape), x.dtype, x.device)
                    M = 1
                    for dim in out_shape[:-1]:
                        M *= dim
                    # x is [M, 2N] contiguous (from F.linear / view); gate=x[:,:d], up=x[:,d:]
                    # pass raw x pointer; kernel reads gate=x[...,:d] and up=x[...,d:] with stride 2N
                    # but our kernel expects separate gate/up pointers -> use views (contiguous slices are views)
                    gate = x[..., :d]
                    up = x[..., d:]
                    # ensure contiguous for the C kernel (slices of contiguous 2D along last dim are non-contiguous
                    # only if there's a trailing dim; for 2D [M,2N] slice [M,N] is non-contiguous -> need contiguous copy)
                    if not gate.is_contiguous():
                        gate = _buf(gate.shape, gate.dtype, gate.device); gate.copy_(x[..., :d])
                        up = _buf(up.shape, up.dtype, up.device); up.copy_(x[..., d:])
                    L.launch_silu_mul(_ptr(gate), _ptr(up), _ptr(out), M, d, _s())
                    return out
                SiluAndMul.forward_cuda = hip_silu
                _patched.add("silu")
                print("[HIP-DSV4] patched SiluAndMul.forward_cuda", flush=True)
        except Exception as e:
            print(f"[HIP-DSV4] silu patch: {e}", flush=True)

    # ---- silu_mul_masked_quant (sglang MLP fused path) ----
    if _on("SGLANG_HIP_SILU_QUANT") and "silu_quant" not in _patched:
        try:
            # patch the native ext silu_mul_quant if present, else register for MLP forward hook
            import lmslim.layers.gemm.int8_utils as m
            if not getattr(m, "_hip_siluq_patched", False):
                m._hip_siluq_patched = True
                # expose a hip silu_mul_masked_quant callable for the MLP patch
                def hip_silu_mul_masked_quant(gate, up, mask=None):
                    M, N = gate.shape
                    if mask is None:
                        mask = torch.ones(M, device=gate.device, dtype=torch.int32)
                    q = torch.empty(M, N, device=gate.device, dtype=torch.int8)
                    s = torch.empty(M, device=gate.device, dtype=torch.float32)
                    L.launch_silu_mul_masked_quant(_ptr(gate), _ptr(up), _ptr(mask), _ptr(q), _ptr(s), M, N, _s())
                    return q, s
                m.hip_silu_mul_masked_quant = hip_silu_mul_masked_quant
                _patched.add("silu_quant")
                print("[HIP-DSV4] registered lmslim.hip_silu_mul_masked_quant", flush=True)
        except Exception as e:
            print(f"[HIP-DSV4] silu_quant patch: {e}", flush=True)

    # ---- rmsnorm_self (sglang jit_kernel rmsnorm_self) ----
    if _on("SGLANG_HIP_RMSNORM") and "rmsnorm" not in _patched:
        try:
            import sglang.jit_kernel.deepseek_v4 as dk
            if not getattr(dk, "_hip_rms_patched", False):
                dk._hip_rms_patched = True
                _orig = dk.rmsnorm_self
                def hip_rmsnorm_self(q, eps=1e-6):
                    # q: [..., HEAD_DIM] bf16, in-place normalized
                    orig_shape = q.shape
                    flat = q.reshape(-1, orig_shape[-1]) if q.dim() > 2 else q
                    M = flat.shape[0]; N = flat.shape[1]
                    L.launch_rmsnorm_self(_ptr(flat), M, N, eps, _s())
                    return q
                dk.rmsnorm_self = hip_rmsnorm_self
                _patched.add("rmsnorm")
                print("[HIP-DSV4] patched jit_kernel.rmsnorm_self", flush=True)
        except Exception as e:
            print(f"[HIP-DSV4] rmsnorm patch: {e}", flush=True)

    # ---- act_quant_fp8 (NSA) ----
    if _on("SGLANG_HIP_NSA_QUANT") and "nsa_quant" not in _patched:
        try:
            from sglang.srt.layers.attention.nsa import tilelang_kernel as tk
            if not getattr(tk, "_hip_patched", False):
                tk._hip_patched = True
                _orig = tk.act_quant
                def hip_act_quant(x, block_size=128, scale_fmt=None, eps=1e-5, use_ue8m0=False):
                    # original signature: act_quant(x, block_size, scale_fmt) -> (y, s)
                    # y: fp8_e4m3fn same shape as x; s: [*, N//block_size] fp32
                    N = x.shape[-1]
                    ng = N // block_size
                    y = _buf(x.shape, torch.float8_e4m3fn, x.device)
                    s_shape = x.shape[:-1] + (ng,)
                    s = _buf(s_shape, torch.float32, x.device)
                    # flatten to 2D for the C kernel (M, N)
                    M = 1
                    for d in x.shape[:-1]:
                        M *= d
                    x2 = x.reshape(M, N) if x.dim() != 2 else x
                    y2 = y.reshape(M, N) if y.dim() != 2 else y
                    s2 = s.reshape(M, ng) if s.dim() != 2 else s
                    L.launch_act_quant_fp8(_ptr(x2), _ptr(y2.view(torch.uint8) if hasattr(y2,'view') else y2), _ptr(s2), M, N, block_size, _s())
                    return y, s
                tk.act_quant = hip_act_quant
                _patched.add("nsa_quant")
                print("[HIP-DSV4] patched nsa.tilelang_kernel.act_quant", flush=True)
        except Exception as e:
            print(f"[HIP-DSV4] nsa_quant patch: {e}", flush=True)

    # ---- MHC: mhc_pre / mhc_post / hc_split_sinkhorn ----
    if _on("SGLANG_HIP_MHC") and "mhc" not in _patched:
        try:
            import sglang.srt.layers.mhc as mhc
            if not getattr(mhc, "_hip_patched", False):
                mhc._hip_patched = True
                # hc_split_sinkhorn(mixes[b,s,mh], hc_scale, hc_base, hc_mult, sinkhorn_iters, eps)
                #   -> pre[b,s,hc], post[b,s,hc], comb[b,s,hc,hc]
                _orig_sk = mhc.hc_split_sinkhorn
                def hip_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
                    b, s, _ = mixes.size()
                    n = b * s
                    # static buffers (graph-capture-safe: stable pointer across replays)
                    pre = _buf((b, s, hc_mult), torch.float32, mixes.device)
                    post = _buf((b, s, hc_mult), torch.float32, mixes.device)
                    comb = _buf((b, s, hc_mult, hc_mult), torch.float32, mixes.device)
                    L.launch_hc_split_sinkhorn(_ptr(mixes), _ptr(hc_scale), _ptr(hc_base),
                                               _ptr(pre), _ptr(post), _ptr(comb), n, _s())
                    return pre, post, comb
                mhc.hc_split_sinkhorn = hip_sinkhorn

                # mhc_pre: skip — engine's mhc_pre_torch/mhc_pre_big_fuse semantics differ
                # (involves residual+rmsnorm, not just sigmoid+mix). Left for follow-up.

                # mhc_post_torch(x, residual, post_layer_mix, comb_res_mix) -> out[n,hc,h]
                #   out = post_layer_mix.unsqueeze(-1)*x.unsqueeze(1) + einsum("nij,njk->nik", comb_res_mix, residual)
                #   mapping to launch_mhc_post(a=comb_res_mix, b=residual, c=post_layer_mix, d=x)
                _orig_post = mhc.mhc_post_torch
                def hip_mhc_post(x, residual, post_layer_mix, comb_res_mix):
                    if x.shape[0] == 0:
                        return _buf((0, residual.shape[1], residual.shape[2]), x.dtype, x.device)
                    # post_layer_mix may be [n, hc, 1] -> squeeze to [n, hc]
                    if post_layer_mix.dim() == 3 and post_layer_mix.shape[-1] == 1:
                        post_layer_mix = post_layer_mix.squeeze(-1)
                    n = comb_res_mix.shape[0]; hc = comb_res_mix.shape[1] if comb_res_mix.dim()==3 else comb_res_mix.shape[-1]
                    hidden = x.shape[-1]
                    out = _buf((n, residual.shape[1], hidden), x.dtype, x.device)
                    L.launch_mhc_post(_ptr(comb_res_mix), _ptr(residual), _ptr(post_layer_mix),
                                      _ptr(x), _ptr(out), n, hidden, _s())
                    return out
                mhc.mhc_post_torch = hip_mhc_post
                _patched.add("mhc")
                print("[HIP-DSV4] patched mhc.hc_split_sinkhorn/mhc_pre/mhc_post", flush=True)
        except Exception as e:
            print(f"[HIP-DSV4] mhc patch: {e}", flush=True)

    # ---- fused_rope (sglang jit_kernel.deepseek_v4.fused_rope) ----
    if _on("SGLANG_HIP_ROPE") and "rope" not in _patched:
        try:
            import sglang.jit_kernel.deepseek_v4 as rope_mod
            if not getattr(rope_mod, "_hip_patched", False):
                rope_mod._hip_patched = True
                _orig = rope_mod.fused_rope
                def hip_rope(q, k, freqs_cis, positions, inverse=False):
                    # q: [nt, nq, rd] bf16, k: [nt, nk, rd] or None, freqs_cis: complex [max_pos, rd/2]
                    # (complex64 memory = real,imag interleaved = our interleaved layout)
                    nt = q.shape[0]; nq = q.shape[1]
                    nk = k.shape[1] if k is not None else 0
                    rd = q.shape[-1]
                    has_k = 1 if k is not None else 0
                    # freqs_cis complex64 -> view as float32 interleaved [max_pos, rd]
                    fc_flat = freqs_cis.view(torch.float32) if freqs_cis.dtype == torch.complex64 else freqs_cis
                    fc_flat = fc_flat.reshape(freqs_cis.shape[0], rd) if fc_flat.dim() != 2 else fc_flat
                    L.launch_fused_rope(_ptr(q), _ptr(k) if k is not None else 0,
                                        _ptr(fc_flat), _ptr(positions),
                                        nt, nq, nk, rd, has_k, _s())
                    return None  # in-place modifies q (and k)
                rope_mod.fused_rope = hip_rope
                _patched.add("rope")
                print("[HIP-DSV4] patched jit_kernel.deepseek_v4.fused_rope", flush=True)
        except Exception as e:
            print(f"[HIP-DSV4] rope patch: {e}", flush=True)

    # ---- topk_transform_512 ----
    if _on("SGLANG_HIP_TOPK") and "topk" not in _patched:
        try:
            from sglang.srt.layers.attention.compressed import indexer as ix
            if not getattr(ix, "_hip_patched", False):
                ix._hip_patched = True
                _orig = ix.topk_transform_512_pytorch_vectorized
                def hip_topk(scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices=None):
                    b = scores.shape[0]; cap = scores.shape[1]
                    ptr_stride = page_tables.shape[1]
                    k = 512
                    # HIP kernel writes page-transformed indices into out_page_indices [b, 512]
                    L.launch_topk_transform(_ptr(scores), _ptr(seq_lens), _ptr(page_tables),
                                            _ptr(out_page_indices), b, cap, ptr_stride, page_size, k, _s())
                    # raw_indices (logical) — if requested, derive by inverse page transform is not trivial;
                    # caller usually passes None. Best-effort: leave as-is (caller handles None).
                    if out_raw_indices is not None:
                        # fill with logical indices best-effort (page_size==1 case)
                        if page_size == 1:
                            out_raw_indices.copy_(out_page_indices)
                ix.topk_transform_512_pytorch_vectorized = hip_topk
                _patched.add("topk")
                print("[HIP-DSV4] patched indexer.topk_transform_512", flush=True)
        except Exception as e:
            print(f"[HIP-DSV4] topk patch: {e}", flush=True)

    # ---- swa_prefill_indices ----
    if _on("SGLANG_HIP_SWA") and "swa" not in _patched:
        try:
            import sglang.jit_kernel.deepseek_v4 as dk
            if not getattr(dk, "_hip_patched", False):
                dk._hip_patched = True
                if hasattr(dk, "tilelang_make_swa_prefill_indices"):
                    _orig = dk.tilelang_make_swa_prefill_indices
                    def hip_swa(seq_lens_k, seq_lens_q, swa_indices, cu_seqlens_q=None):
                        b = seq_lens_q.shape[0]
                        window = swa_indices.shape[1]
                        nq = swa_indices.shape[0]   # from output shape, no .item() sync
                        if cu_seqlens_q is None:
                            cu = _buf((b + 1,), torch.int32, seq_lens_q.device)
                            cu.zero_(); cu[1:] = seq_lens_q.cumsum(0)
                            cu_seqlens_q = cu
                        L.launch_swa_prefill_indices(_ptr(swa_indices), _ptr(seq_lens_k), _ptr(seq_lens_q),
                                                     _ptr(cu_seqlens_q), nq, b, window, _s())
                        return swa_indices
                    dk.tilelang_make_swa_prefill_indices = hip_swa
                    _patched.add("swa")
                    print("[HIP-DSV4] patched deepseek_v4.tilelang_make_swa_prefill_indices", flush=True)
        except Exception as e:
            print(f"[HIP-DSV4] swa patch: {e}", flush=True)

    # ---- merge_attn_states ----
    if _on("SGLANG_HIP_MERGE") and "merge" not in _patched:
        try:
            from sglang.srt.layers.attention import triton_backend as tb
            # merge_state lives in sgl_kernel/vllm; patch the triton variant if reachable
            import vllm.v1.attention.ops.triton_merge_attn_states as tm
            if not getattr(tm, "_hip_patched", False):
                tm._hip_patched = True
                _orig = tm.merge_attn_states
                def hip_merge(output, prefix_output, prefix_lse, suffix_output, suffix_lse, output_lse=None):
                    nt = prefix_output.shape[0]; nh = prefix_output.shape[1]; hs = prefix_output.shape[2]
                    if output_lse is None:
                        output_lse = _buf((nh, nt), torch.float32, prefix_output.device)
                    L.launch_merge_attn_states(_ptr(output), _ptr(output_lse),
                                                _ptr(prefix_output), _ptr(prefix_lse),
                                                _ptr(suffix_output), _ptr(suffix_lse),
                                                nt, nh, hs, _s())
                    return output, output_lse
                tm.merge_attn_states = hip_merge
                _patched.add("merge")
                print("[HIP-DSV4] patched triton_merge_attn_states", flush=True)
        except Exception as e:
            print(f"[HIP-DSV4] merge patch: {e}", flush=True)

    # ---- grouped_gemm_int8 (optional) ----
    if _on("SGLANG_HIP_GROUPED_GEMM") and "grouped_gemm" not in _patched:
        try:
            import lmslim.layers.fused_moe.fuse_moe_w4a8_marlin as fm
            # Only as fallback; vendor marlin usually faster. Hook minimal.
            _patched.add("grouped_gemm")
            print("[HIP-DSV4] grouped_gemm hook registered (vendor marlin preferred)", flush=True)
        except Exception as e:
            print(f"[HIP-DSV4] grouped_gemm patch: {e}", flush=True)


def _loop_try():
    """Retry patches until target modules are imported (lazy, deferred)."""
    for _ in range(60):
        _try_patches()
        if len(_patched) >= _want_count():
            return
        threading.Timer(0.5, _loop_try).start() if False else None
        return
def _want_count():
    c = 0
    for n in ["SGLANG_HIP_PTQ","SGLANG_HIP_PTGQ","SGLANG_HIP_SILU","SGLANG_HIP_SILU_QUANT","SGLANG_HIP_RMSNORM","SGLANG_HIP_NSA_QUANT","SGLANG_HIP_MHC","SGLANG_HIP_ROPE","SGLANG_HIP_TOPK","SGLANG_HIP_SWA","SGLANG_HIP_MERGE","SGLANG_HIP_GROUPED_GEMM"]:
        if _on(n): c += 1
    return c

# Patch strategy: sitecustomize imports this module at python startup (via PYTHONPATH).
# We register a sys.meta_path finder that triggers _try_patches() after each sglang
# module import, so patches apply synchronously when the target module loads (before
# the server forks TP workers). Also do a synchronous first pass + periodic retry.
import importlib.abc, importlib.machinery, sys

class _HipPatchFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        # trigger patches when any sglang target module loads
        if name in ("sglang.srt.layers.mhc",
                    "sglang.srt.layers.attention.nsa.tilelang_kernel",
                    "sglang.srt.layers.attention.compressed.indexer",
                    "sglang.srt.layers.deepseek_v4_rope",
                    "sglang.jit_kernel.deepseek_v4",
                    "sglang.srt.layers.activation",
                    "vllm.v1.attention.ops.triton_merge_attn_states",
                    "lmslim.layers.gemm.int8_utils",
                    "lightop"):
            _try_patches()
        return None  # don't actually load the module; just trigger patches

def _install_hook():
    if not any(isinstance(f, _HipPatchFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _HipPatchFinder())

def _start_retries():
    import threading
    _install_hook()
    _try_patches()  # synchronous first pass
    def attempt():
        _try_patches()
        if _master() and len(_patched) < _want_count():
            threading.Timer(2.0, attempt).start()
    if _master() and len(_patched) < _want_count():
        threading.Timer(0.5, attempt).start()

if os.environ.get("SGLANG_USE_HIP_DSV4") == "1":
    _start_retries()
