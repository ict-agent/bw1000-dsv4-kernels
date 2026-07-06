"""Patch lmslim per_token_quant_int8 to use HIP kernel."""
import os

path = "/usr/local/lib/python3.10/dist-packages/lmslim/layers/gemm/int8_utils.py"

# Backup
if not os.path.exists(path + ".bak"):
    import shutil
    shutil.copy2(path, path + ".bak")

with open(path) as f:
    content = f.read()

# Check if already patched
if "_hip_per_token_quant_int8" in content:
    print("Already patched, skipping")
else:
    patch_code = '''
# === HIP FUSED KERNEL PATCH (auto-injected) ===
import ctypes as _hip_ctypes
import subprocess as _hip_subprocess
import os as _hip_os
_hip_lib_path = "/workspace/hip_kernels/libfused_ops_v2.so"
if not _hip_os.path.exists(_hip_lib_path):
    _hip_subprocess.run(["hipcc", "-O3", "--offload-arch=gfx936", "-shared", "-fPIC",
                        "-o", _hip_lib_path, "/workspace/hip_kernels/fused_ops_v2.hip"],
                       check=True, capture_output=True)
_hip_lib2 = _hip_ctypes.CDLL(_hip_lib_path)
_hip_lib2.launch_per_token_quant_int8.argtypes = [_hip_ctypes.c_void_p]*3 + [_hip_ctypes.c_int]*2 + [_hip_ctypes.c_void_p]

def _hip_per_token_quant_int8(x, scale_dtype=None, cal_sum=False):
    import torch
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    x_flat = x.reshape(M, N).contiguous()
    x_q = torch.empty(M, N, device=x.device, dtype=torch.int8)
    x_s = torch.empty(M, device=x.device, dtype=torch.float32)
    _hip_lib2.launch_per_token_quant_int8(x_flat.data_ptr(), x_q.data_ptr(), x_s.data_ptr(), M, N, None)
    return x_q.reshape(x.shape), x_s.unsqueeze(-1)  # MUST be [M, 1] not [M]
# === END HIP PATCH ===
'''

    # Find and replace the function
    old_def = "def per_token_quant_int8(x):"
    new_def = "def per_token_quant_int8(x):\n    return _hip_per_token_quant_int8(x)\n\ndef _original_per_token_quant_int8(x):"

    if old_def in content:
        content = patch_code + "\n" + content.replace(old_def, new_def)
        with open(path, "w") as f:
            f.write(content)
        print("Patched successfully: %s" % path)
    else:
        print("ERROR: Could not find function definition to patch")
