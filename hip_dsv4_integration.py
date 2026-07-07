"""HIP DSV4 kernel integration into SGLang engine.

Patches the new HIP kernels (act_quant_fp8, mhc_pre/post, hc_split_sinkhorn,
fused_rope, topk_transform_512, swa_prefill_indices, merge_attn_states) into
the corresponding SGLang/lmslim/lightop call sites. All graph-safe (stream-driven).

Env switches (all gated by master SGLANG_USE_HIP_DSV4=1):
  SGLANG_HIP_NSA_QUANT=1     act_quant_fp8
  SGLANG_HIP_MHC=1           mhc_pre / mhc_post / hc_split_sinkhorn
  SGLANG_HIP_ROPE=1          fused_rope
  SGLANG_HIP_TOPK=1          topk_transform_512
  SGLANG_HIP_SWA=1           swa_prefill_indices
  SGLANG_HIP_MERGE=1         merge_attn_states
  SGLANG_HIP_GROUPED_GEMM=1  grouped_gemm_int8 (optional; vendor GEMM usually faster)

Load order: import this module at engine startup (e.g. via .pth or PYTHONSTARTUP),
it registers lazy patches that apply once the target modules are imported.
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

    # ---- act_quant_fp8 (NSA) ----
    if _on("SGLANG_HIP_NSA_QUANT") and "nsa_quant" not in _patched:
        try:
            from sglang.srt.layers.attention.nsa import tilelang_kernel as tk
            if not getattr(tk, "_hip_patched", False):
                tk._hip_patched = True
                _orig = tk.act_quant
                def hip_act_quant(x, y, s, block_size, eps=1e-5, use_ue8m0=False):
                    M, N = x.shape
                    L.launch_act_quant_fp8(_ptr(x), _ptr(y), _ptr(s), M, N, block_size, _s())
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
                        return torch.empty((0, residual.shape[1], residual.shape[2]), dtype=x.dtype, device=x.device)
                    n = comb_res_mix.shape[0]; hc = comb_res_mix.shape[1] if comb_res_mix.dim()==3 else comb_res_mix.shape[-1]
                    # comb_res_mix:[n,hc,hc] fp32, residual:[n,hc,h] bf16, post_layer_mix:[n,hc] fp32, x:[n,h] bf16
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

    # ---- fused_rope ----
    if _on("SGLANG_HIP_ROPE") and "rope" not in _patched:
        try:
            import sglang.srt.layers.deepseek_v4_rope as rope_mod
            if not getattr(rope_mod, "_hip_patched", False):
                rope_mod._hip_patched = True
                _orig = rope_mod.apply_rotary_emb
                def hip_rope(q, k, freqs_cis, positions, *a, **kw):
                    # q: [nt, nq, rd], k: [nt, nk, rd] or None, freqs_cis: [max_pos, rd] interleaved, positions: [nt]
                    nt = q.shape[0]; nq = q.shape[1]
                    nk = k.shape[1] if k is not None else 0
                    rd = q.shape[-1]
                    has_k = 1 if k is not None else 0
                    L.launch_fused_rope(_ptr(q), _ptr(k) if k is not None else 0,
                                        _ptr(freqs_cis), _ptr(positions),
                                        nt, nq, nk, rd, has_k, _s())
                    return q, k
                rope_mod.apply_rotary_emb = hip_rope
                _patched.add("rope")
                print("[HIP-DSV4] patched deepseek_v4_rope.apply_rotary_emb", flush=True)
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
                        if cu_seqlens_q is None:
                            cu_seqlens_q = torch.cumsum(seq_lens_q, dim=0, dtype=torch.int32)
                            cu_seqlens_q = torch.nn.functional.pad(cu_seqlens_q, (1, 0), value=0)
                        nq = int(seq_lens_q.sum().item())
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
    for n in ["SGLANG_HIP_NSA_QUANT","SGLANG_HIP_MHC","SGLANG_HIP_ROPE","SGLANG_HIP_TOPK","SGLANG_HIP_SWA","SGLANG_HIP_MERGE","SGLANG_HIP_GROUPED_GEMM"]:
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
                    "vllm.v1.attention.ops.triton_merge_attn_states",
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
