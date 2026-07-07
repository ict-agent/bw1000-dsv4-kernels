"""HIP DSV4 kernel integration into SGLang engine.

Patches sglang/lmslim/lightop/jit_kernel call sites to use the HIP wrappers from
hip_wrapper.py (Layer 2). All wrappers are graph-safe (buffer pool, minimal Python ops).

Env switches (all gated by master SGLANG_USE_HIP_DSV4=1):
  SGLANG_HIP_PTQ=1           per_token_quant_int8        (lmslim)
  SGLANG_HIP_PTGQ=1          per_token_group_quant_int8  (lmslim)
  SGLANG_HIP_SILU=1          silu_and_mul                (sglang SiluAndMul)
  SGLANG_HIP_SILU_QUANT=1    silu_mul_masked_quant       (lmslim fused)
  SGLANG_HIP_RMSNORM=1       rmsnorm_self                (sglang jit_kernel)
  SGLANG_HIP_NSA_QUANT=1     act_quant_fp8               (nsa tilelang_kernel)
  SGLANG_HIP_MHC=1           mhc_post / hc_split_sinkhorn (sglang mhc)
  SGLANG_HIP_ROPE=1          fused_rope                  (jit_kernel.fused_rope)
  SGLANG_HIP_TOPK=1          topk_transform_512          (compressed indexer)
  SGLANG_HIP_SWA=1           swa_prefill_indices          (jit_kernel)
  SGLANG_HIP_MERGE=1         merge_attn_states           (vllm triton)

Load: import this module at engine startup (sitecustomize.py on PYTHONPATH).
A sys.meta_path finder patches each target module synchronously as it loads
(before TP workers fork). Output buffers use hip_wrapper._buf static pool.
"""
import os, sys, threading

def _on(name): return os.environ.get(name, "0") == "1"
def _master(): return os.environ.get("SGLANG_USE_HIP_DSV4", "0") == "1"

_patched = set()
_in_try = False

def _try_patches():
    global _in_try
    if _in_try or not _master(): return
    _in_try = True
    try:
        _try_impl()
    finally:
        _in_try = False

def _try_impl():
    if len(_patched) >= _want_count(): return
    import hip_wrapper as W

    # PTQ (lmslim)
    if _on("SGLANG_HIP_PTQ") and "ptq" not in _patched:
        try:
            import lmslim.layers.gemm.int8_utils as m
            if not getattr(m, "_hip_ptq", False):
                m._hip_ptq = True
                m.per_token_quant_int8 = W.per_token_quant_int8
                _patched.add("ptq"); print("[HIP-DSV4] patched lmslim.per_token_quant_int8", flush=True)
        except Exception as e: print(f"[HIP-DSV4] ptq: {e}", flush=True)

    # PTGQ (lmslim)
    if _on("SGLANG_HIP_PTGQ") and "ptgq" not in _patched:
        try:
            import lmslim.layers.gemm.int8_utils as m
            if not getattr(m, "_hip_ptgq", False):
                m._hip_ptgq = True
                m.per_token_group_quant_int8 = W.per_token_group_quant_int8
                _patched.add("ptgq"); print("[HIP-DSV4] patched lmslim.per_token_group_quant_int8", flush=True)
        except Exception as e: print(f"[HIP-DSV4] ptgq: {e}", flush=True)

    # SILU (sglang SiluAndMul.forward_cuda)
    if _on("SGLANG_HIP_SILU") and "silu" not in _patched:
        try:
            from sglang.srt.layers.activation import SiluAndMul
            if not getattr(SiluAndMul, "_hip_silu", False):
                SiluAndMul._hip_silu = True
                _orig = SiluAndMul.forward_cuda
                def hip_silu(self, x):
                    return W.silu_and_mul(x)
                SiluAndMul.forward_cuda = hip_silu
                _patched.add("silu"); print("[HIP-DSV4] patched SiluAndMul.forward_cuda", flush=True)
        except Exception as e: print(f"[HIP-DSV4] silu: {e}", flush=True)

    # RMSNORM (sglang jit_kernel.rmsnorm_self)
    if _on("SGLANG_HIP_RMSNORM") and "rmsnorm" not in _patched:
        try:
            import sglang.jit_kernel.deepseek_v4 as dk
            if not getattr(dk, "_hip_rms", False):
                dk._hip_rms = True
                dk.rmsnorm_self = W.rmsnorm_self
                _patched.add("rmsnorm"); print("[HIP-DSV4] patched jit_kernel.rmsnorm_self", flush=True)
        except Exception as e: print(f"[HIP-DSV4] rmsnorm: {e}", flush=True)

    # NSA act_quant
    if _on("SGLANG_HIP_NSA_QUANT") and "nsa_quant" not in _patched:
        try:
            from sglang.srt.layers.attention.nsa import tilelang_kernel as tk
            if not getattr(tk, "_hip_aq", False):
                tk._hip_aq = True
                _orig = tk.act_quant
                def hip_aq(x, block_size=128, scale_fmt=None):
                    return W.act_quant(x, block_size, scale_fmt)
                tk.act_quant = hip_aq
                _patched.add("nsa_quant"); print("[HIP-DSV4] patched nsa.tilelang_kernel.act_quant", flush=True)
        except Exception as e: print(f"[HIP-DSV4] nsa_quant: {e}", flush=True)

    # MHC (hc_split_sinkhorn + mhc_post_torch)
    if _on("SGLANG_HIP_MHC") and "mhc" not in _patched:
        try:
            import sglang.srt.layers.mhc as mhc
            if not getattr(mhc, "_hip_mhc", False):
                mhc._hip_mhc = True
                mhc.hc_split_sinkhorn = W.hc_split_sinkhorn
                mhc.mhc_post_torch = W.mhc_post_torch
                _patched.add("mhc"); print("[HIP-DSV4] patched mhc.hc_split_sinkhorn/mhc_post_torch", flush=True)
        except Exception as e: print(f"[HIP-DSV4] mhc: {e}", flush=True)

    # FUSED_ROPE (jit_kernel.fused_rope)
    if _on("SGLANG_HIP_ROPE") and "rope" not in _patched:
        try:
            import sglang.jit_kernel.deepseek_v4 as dk
            if not getattr(dk, "_hip_rope", False):
                dk._hip_rope = True
                _orig = dk.fused_rope
                def hip_rope(q, k, freqs_cis, positions, inverse=False):
                    return W.fused_rope(q, k, freqs_cis, positions, inverse)
                dk.fused_rope = hip_rope
                _patched.add("rope"); print("[HIP-DSV4] patched jit_kernel.fused_rope", flush=True)
        except Exception as e: print(f"[HIP-DSV4] rope: {e}", flush=True)

    # TOPK
    if _on("SGLANG_HIP_TOPK") and "topk" not in _patched:
        try:
            from sglang.srt.layers.attention.compressed import indexer as ix
            if not getattr(ix, "_hip_topk", False):
                ix._hip_topk = True
                ix.topk_transform_512_pytorch_vectorized = W.topk_transform_512
                _patched.add("topk"); print("[HIP-DSV4] patched indexer.topk_transform_512", flush=True)
        except Exception as e: print(f"[HIP-DSV4] topk: {e}", flush=True)

    # SWA
    if _on("SGLANG_HIP_SWA") and "swa" not in _patched:
        try:
            import sglang.jit_kernel.deepseek_v4 as dk
            if not getattr(dk, "_hip_swa", False):
                dk._hip_swa = True
                dk.tilelang_make_swa_prefill_indices = W.tilelang_make_swa_prefill_indices
                _patched.add("swa"); print("[HIP-DSV4] patched jit_kernel.tilelang_make_swa_prefill_indices", flush=True)
        except Exception as e: print(f"[HIP-DSV4] swa: {e}", flush=True)

    # MERGE
    if _on("SGLANG_HIP_MERGE") and "merge" not in _patched:
        try:
            import vllm.v1.attention.ops.triton_merge_attn_states as tm
            if not getattr(tm, "_hip_merge", False):
                tm._hip_merge = True
                tm.merge_attn_states = W.merge_attn_states
                _patched.add("merge"); print("[HIP-DSV4] patched triton_merge_attn_states", flush=True)
        except Exception as e: print(f"[HIP-DSV4] merge: {e}", flush=True)

# sys.meta_path finder: trigger patches when target modules load
import importlib.abc, importlib.machinery
class _HipFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name in ("sglang.srt.layers.mhc",
                    "sglang.srt.layers.attention.nsa.tilelang_kernel",
                    "sglang.srt.layers.attention.compressed.indexer",
                    "sglang.srt.layers.activation",
                    "sglang.jit_kernel.deepseek_v4",
                    "vllm.v1.attention.ops.triton_merge_attn_states",
                    "lmslim.layers.gemm.int8_utils",
                    "lightop"):
            _try_patches()
        return None

def _want_count():
    c = 0
    for n in ["SGLANG_HIP_PTQ","SGLANG_HIP_PTGQ","SGLANG_HIP_SILU","SGLANG_HIP_SILU_QUANT",
              "SGLANG_HIP_RMSNORM","SGLANG_HIP_NSA_QUANT","SGLANG_HIP_MHC","SGLANG_HIP_ROPE",
              "SGLANG_HIP_TOPK","SGLANG_HIP_SWA","SGLANG_HIP_MERGE"]:
        if _on(n): c += 1
    return c

def _start():
    import threading
    if not any(isinstance(f, _HipFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _HipFinder())
    _try_patches()
    if _master() and len(_patched) < _want_count():
        def attempt():
            _try_patches()
            if _master() and len(_patched) < _want_count():
                threading.Timer(2.0, attempt).start()
        threading.Timer(0.5, attempt).start()

if _master():
    _start()
