"""Register HIP kernels as PyTorch custom ops for CUDA graph compatibility.
Uses torch.library so the op is captured by graph replay.
"""
import torch
import ctypes
import os
import subprocess

_HIP_LIB = "/workspace/hip_kernels/libdsv4_ops_hip.so"
_HIP_SRC = "/workspace/hip_kernels/dsv4_ops_hip.hip"
if not os.path.exists(_HIP_LIB) or os.path.getmtime(_HIP_SRC) > os.path.getmtime(_HIP_LIB):
    subprocess.run(["hipcc", "-O3", "--offload-arch=gfx936", "-shared", "-fPIC",
                    "-o", _HIP_LIB, _HIP_SRC], check=True, capture_output=True)

_lib = ctypes.CDLL(_HIP_LIB)
_lib.launch_ptq.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*2 + [ctypes.c_void_p]

# Define the custom op schema
@torch.library.custom_op("hip::per_token_quant_int8", mutates_args=())
def per_token_quant_int8_custom(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token INT8 quantization using HIP kernel."""
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    xf = x.reshape(M, N).contiguous()
    q = torch.empty(M, N, device=x.device, dtype=torch.int8)
    s = torch.empty(M, 1, device=x.device, dtype=torch.float32)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    _lib.launch_ptq(xf.data_ptr(), q.data_ptr(), s.data_ptr(), M, N, stream)
    return q.reshape(x.shape), s.reshape(x.shape[:-1] + (1,))

@per_token_quant_int8_custom.register_fake
def _(x):
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    return (torch.empty(x.shape, dtype=torch.int8, device=x.device),
            torch.empty(x.shape[:-1] + (1,), dtype=torch.float32, device=x.device))

# Monkey-patch lmslim to use our custom op
def _apply_patch():
    try:
        import lmslim.layers.gemm.int8_utils as m
        if not hasattr(m, '_hip_custom_op_patched'):
            m._hip_custom_op_patched = True
            m._orig_per_token_quant_int8 = m.per_token_quant_int8
            def hip_wrapper(x, scale_dtype=None, cal_sum=False):
                return per_token_quant_int8_custom(x)
            m.per_token_quant_int8 = hip_wrapper
            print("[HIP-CUSTOM-OP] Patched lmslim.per_token_quant_int8 -> torch custom op (graph-safe)", flush=True)
    except ImportError:
        pass

if os.environ.get("SGLANG_USE_HIP_QUANT", "0") == "1":
    _apply_patch()
