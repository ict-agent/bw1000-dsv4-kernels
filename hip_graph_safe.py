# hip_graph_safe.py - Graph-safe HIP kernel integration via PyTorch C++ extension
# Uses per_token_quant_int8_stream with explicit stream from Python
# This is graph-capturable because the kernel launch uses the correct stream
import os, sys, torch

_BUILD_DIR = "/workspace/hip_kernels/torch_ext_build"
if _BUILD_DIR not in sys.path:
    sys.path.insert(0, _BUILD_DIR)

try:
    import dsv4_hip_ext
    _HIP_AVAILABLE = True
except ImportError:
    _HIP_AVAILABLE = False

def _apply_patch():
    if not _HIP_AVAILABLE:
        print("[HIP-GRAPH-SAFE] dsv4_hip_ext not available, skipping patch", flush=True)
        return
    try:
        import lmslim.layers.gemm.int8_utils as m
        if not hasattr(m, '_hip_graph_safe_patched'):
            m._hip_graph_safe_patched = True
            m._orig_per_token_quant_int8 = m.per_token_quant_int8
            def hip_wrapper(x, scale_dtype=None, cal_sum=False):
                # Pass current stream explicitly (graph-safe)
                stream_ptr = torch.cuda.current_stream().cuda_stream
                return dsv4_hip_ext.per_token_quant_int8_stream(x, stream_ptr)
            m.per_token_quant_int8 = hip_wrapper
            print("[HIP-GRAPH-SAFE] Patched lmslim.per_token_quant_int8 -> native ext with stream (graph-safe)", flush=True)
    except ImportError:
        pass

if os.environ.get("SGLANG_USE_HIP_QUANT", "0") == "1":
    _apply_patch()
