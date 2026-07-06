# hip_quant_method.py - Drop-in HIP quantization method for SGLang
#
# Integrates HIP W8A8 kernels into SGLang WITHOUT modifying any source files.
# Uses Python class inheritance to override the quantization apply() method.
#
# Usage:
#   export PYTHONPATH=/workspace/hip_kernels:$PYTHONPATH
#   export SGLANG_USE_HIP_QUANT=1
#   python3 -m sglang.launch_server ...
#
# This module is imported via sitecustomize or explicit import in the model file.
# Since SGLang uses multiprocessing.spawn, we use a .pth file approach instead.

import os
import ctypes
import subprocess
import torch

# Compile HIP kernel if needed
_HIP_LIB = "/workspace/hip_kernels/libdsv4_ops_hip.so"
_HIP_SRC = "/workspace/hip_kernels/dsv4_ops_hip.hip"
if not os.path.exists(_HIP_LIB) or os.path.getmtime(_HIP_SRC) > os.path.getmtime(_HIP_LIB):
    subprocess.run(["hipcc", "-O3", "--offload-arch=gfx936", "-shared", "-fPIC",
                    "-o", _HIP_LIB, _HIP_SRC], check=True, capture_output=True)

_lib = ctypes.CDLL(_HIP_LIB)
_lib.launch_ptq.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*2 + [ctypes.c_void_p]
_lib.launch_silu_mul_quant.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_void_p]
_lib.launch_add_rmsnorm_quant.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*2 + [ctypes.c_float, ctypes.c_void_p]

def hip_per_token_quant_int8(x, scale_dtype=torch.float32, cal_sum=False):
    """Bit-exact HIP replacement for lmslim.per_token_quant_int8.
    Uses correct CUDA stream for graph capture compatibility."""
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    xf = x.reshape(M, N).contiguous()
    xq = torch.empty(M, N, device=x.device, dtype=torch.int8)
    xs = torch.empty(M, 1, device=x.device, dtype=torch.float32)
    # Use current stream (critical for CUDA graph compatibility)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    _lib.launch_ptq(xf.data_ptr(), xq.data_ptr(), xs.data_ptr(), M, N, stream)
    return xq.reshape(x.shape), xs.reshape(x.shape[:-1] + (1,))

def hip_silu_mul_quant(gate, up):
    """Fused SiLU+Mul+quant, returns (int8, scale)."""
    M = gate.numel() // gate.shape[-1]
    N = gate.shape[-1]
    gf = gate.reshape(M, N).contiguous()
    uf = up.reshape(M, N).contiguous()
    q = torch.empty(M, N, device=gate.device, dtype=torch.int8)
    s = torch.empty(M, 1, device=gate.device, dtype=torch.float32)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    _lib.launch_silu_mul_quant(gf.data_ptr(), uf.data_ptr(), q.data_ptr(), s.data_ptr(), M, N, stream)
    return q.reshape(gate.shape), s.reshape(gate.shape[:-1] + (1,))

def hip_add_rmsnorm_quant(residual, x, weight, eps=1e-6):
    """Fused add+rmsnorm+quant, returns (int8, scale). Residual updated in-place."""
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    q = torch.empty(M, N, device=x.device, dtype=torch.int8)
    s = torch.empty(M, 1, device=x.device, dtype=torch.float32)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    _lib.launch_add_rmsnorm_quant(residual.data_ptr(), x.data_ptr(), weight.data_ptr(),
                                   q.data_ptr(), s.data_ptr(), M, N, eps, stream)
    return q, s

# === Integration via lmslim monkey-patch ===
# This replaces the import that SGLang's quantization code uses.
# Must happen BEFORE SGLang imports per_token_quant_int8.

def _apply_patch():
    """Patch lmslim.per_token_quant_int8 at import time."""
    try:
        import lmslim.layers.gemm.int8_utils as m
        if not hasattr(m, '_hip_patched'):
            m._hip_patched = True
            m._orig_per_token_quant_int8 = m.per_token_quant_int8
            m.per_token_quant_int8 = hip_per_token_quant_int8
            print("[HIP-QUANT] Patched lmslim.per_token_quant_int8 -> HIP (stream-safe, graph-compatible)", flush=True)
    except ImportError:
        pass

# Auto-patch when this module is imported
if os.environ.get("SGLANG_USE_HIP_QUANT", "0") == "1":
    _apply_patch()
