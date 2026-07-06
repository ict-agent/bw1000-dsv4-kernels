"""Patch lmslim int8_utils.py to use bit-exact HIP kernel."""
import os, shutil
path = "/usr/local/lib/python3.10/dist-packages/lmslim/layers/gemm/int8_utils.py"
if not os.path.exists(path + ".orig"):
    shutil.copy2(path, path + ".orig")

with open(path) as f: c = f.read()
if "_hip_exact_per_token_quant_int8" in c:
    print("already patched"); exit(0)

patch = '''
# === BIT-EXACT HIP KERNEL PATCH ===
import ctypes as _ct
import subprocess as _sp
import os as _os
_HIP_LIB_PATH = "/workspace/hip_kernels/libfused_ops_exact.so"
_HIP_SRC = "/workspace/hip_kernels/fused_ops_exact.hip"
if not _os.path.exists(_HIP_LIB_PATH) or _os.path.getmtime(_HIP_SRC) > _os.path.getmtime(_HIP_LIB_PATH):
    _sp.run(["hipcc","-O3","--offload-arch=gfx936","-shared","-fPIC","-o",_HIP_LIB_PATH,_HIP_SRC], check=True, capture_output=True)
_hip_lib = _ct.CDLL(_HIP_LIB_PATH)
_hip_lib.launch_per_token_quant_int8_exact.argtypes = [_ct.c_void_p]*3 + [_ct.c_int]*2 + [_ct.c_void_p]

def _hip_exact_per_token_quant_int8(x, scale_dtype=None, cal_sum=False):
    import torch
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    xf = x.reshape(M, N).contiguous()
    xq = torch.empty(M, N, device=x.device, dtype=torch.int8)
    xs = torch.empty(M, device=x.device, dtype=torch.float32)
    _hip_lib.launch_per_token_quant_int8_exact(xf.data_ptr(), xq.data_ptr(), xs.data_ptr(), M, N, None)
    return xq.reshape(x.shape), xs.reshape(x.shape[:-1] + (1,))
# === END PATCH ===
'''
c = patch + "\n" + c.replace("def per_token_quant_int8(x):", "def _triton_per_token_quant_int8_orig(x):")
# add a redirecting public def
c += '\ndef per_token_quant_int8(x):\n    return _hip_exact_per_token_quant_int8(x)\n'
with open(path,"w") as f: f.write(c)
print("patched")
