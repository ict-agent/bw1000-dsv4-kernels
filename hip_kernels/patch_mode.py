"""Patch lmslim per_token_quant_int8 to one of three configs.
Usage: python3 patch_mode.py <orig|github|hip>
"""
import sys, os, shutil
PATH = "/usr/local/lib/python3.10/dist-packages/lmslim/layers/gemm/int8_utils.py"
ORIG = PATH + ".orig"
if not os.path.exists(ORIG): shutil.copy2(PATH, ORIG)
mode = sys.argv[1]

# Always start from original
shutil.copy2(ORIG, PATH)
with open(PATH) as f: c = f.read()

if mode == "orig":
    print("MODE=orig (original lmslim Triton)")
    # nothing to do, original restored
elif mode == "github":
    # Replace per_token_quant_int8 with the repo's naive PyTorch _quantize_input ref
    inject = '''
# === GITHUB REF KERNEL (mmt-at/dsv4_ops_unit_tests _quantize_input) ===
def _github_per_token_quant_int8(x, scale_dtype=None, cal_sum=False):
    import torch
    absmax = x.abs().amax(dim=-1, keepdim=True).float()
    scale_x = absmax / 127.0
    qx = (x.float() / scale_x).round().clamp(-128, 127).char()
    return qx, scale_x
# === END ===

def _disabled_per_token_quant_int8(x):'''
    c = inject + "\n" + c.replace("def per_token_quant_int8(x):", "def _triton_per_token_quant_int8(x):", 1)
    c += '\ndef per_token_quant_int8(x):\n    return _github_per_token_quant_int8(x)\n'
    with open(PATH,"w") as f: f.write(c)
    print("MODE=github (naive PyTorch ref from dsv4_ops_unit_tests)")
elif mode == "hip":
    inject = '''
# === HIP EXACT KERNEL ===
import ctypes as _ct, subprocess as _sp, os as _os
_L="/workspace/hip_kernels/libdsv4_ops_hip.so"
if not _os.path.exists(_L):
    _sp.run(["hipcc","-O3","--offload-arch=gfx936","-shared","-fPIC","-o",_L,"/workspace/hip_kernels/dsv4_ops_hip.hip"],check=True,capture_output=True)
_hl=_ct.CDLL(_L)
_hl.launch_ptq.argtypes=[_ct.c_void_p]*3+[_ct.c_int]*2+[_ct.c_void_p]
def _hip_per_token_quant_int8(x, scale_dtype=None, cal_sum=False):
    import torch
    M=x.numel()//x.shape[-1]; N=x.shape[-1]
    xf=x.reshape(M,N).contiguous()
    q=torch.empty(M,N,device=x.device,dtype=torch.int8)
    s=torch.empty(M,1,device=x.device,dtype=torch.float32)
    _hl.launch_ptq(xf.data_ptr(),q.data_ptr(),s.data_ptr(),M,N,None)
    return q.reshape(x.shape), s.reshape(x.shape[:-1]+(1,))
# === END ===

def _disabled_per_token_quant_int8(x):'''
    c = inject + "\n" + c.replace("def per_token_quant_int8(x):", "def _triton_per_token_quant_int8(x):", 1)
    c += '\ndef per_token_quant_int8(x):\n    return _hip_per_token_quant_int8(x)\n'
    with open(PATH,"w") as f: f.write(c)
    print("MODE=hip (HIP C++ exact kernel)")
else:
    print("unknown mode"); sys.exit(1)
