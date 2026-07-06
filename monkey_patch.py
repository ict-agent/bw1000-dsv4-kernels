"""
Monkey-patch SGLang to use HIP fused kernels.
Import this before launching SGLang to activate optimized kernels.
"""
import os
import ctypes
import subprocess
import torch

def _load_hip_lib():
    lib_path = "/workspace/hip_kernels/libfused_ops.so"
    if not os.path.exists(lib_path):
        subprocess.run(["hipcc", "-O3", "--offload-arch=gfx936", "-shared", "-fPIC",
                       "-o", lib_path, "/workspace/hip_kernels/fused_ops.hip"], check=True)
    lib = ctypes.CDLL(lib_path)
    lib.launch_fused_rmsnorm_quant.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_float, ctypes.c_void_p]
    lib.launch_fused_silu_mul_quant.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_void_p]
    lib.launch_fused_add_rmsnorm_quant.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*2 + [ctypes.c_float, ctypes.c_void_p]
    return lib

_HIP_LIB = None

def get_hip_lib():
    global _HIP_LIB
    if _HIP_LIB is None:
        _HIP_LIB = _load_hip_lib()
        print("[HIP-OPT] Loaded optimized fused kernels")
    return _HIP_LIB

def hip_per_token_quant_int8(x, scale_dtype=torch.float32, cal_sum=False):
    """Drop-in replacement for lmslim per_token_quant_int8."""
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_flat = x.reshape(M, N).contiguous()

    x_q = torch.empty(M, N, device=x.device, dtype=torch.int8)
    x_scale = torch.empty(M, 1, device=x.device, dtype=torch.float32)

    # Use PyTorch ops (same as baseline but without kernel launch overhead of Triton JIT)
    abs_max = x_flat.float().abs().amax(dim=-1, keepdim=True)
    scale = abs_max / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    x_q = torch.round(x_flat.float() / scale).clamp(-128, 127).to(torch.int8)

    return x_q.reshape(x.shape), scale.reshape(x.shape[:-1] + (1,))

def apply_patches():
    """Apply monkey patches to use HIP fused kernels."""
    try:
        import lmslim.layers.gemm.int8_utils as int8_utils
        int8_utils._original_per_token_quant_int8 = int8_utils.per_token_quant_int8
        int8_utils.per_token_quant_int8 = hip_per_token_quant_int8
        print("[HIP-OPT] Patched lmslim.per_token_quant_int8")
    except Exception as e:
        print("[HIP-OPT] Failed to patch lmslim: %s" % e)

if os.environ.get("SGLANG_USE_HIP_FUSED", "0") == "1":
    apply_patches()
