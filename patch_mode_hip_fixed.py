"""Patch lmslim to use HIP kernel with CORRECT stream (fixes the slowdown).
Key fix: pass torch.cuda.current_stream().cuda_stream instead of None.
This eliminates implicit stream synchronization.
"""
import sys, os, shutil
PATH = "/usr/local/lib/python3.10/dist-packages/lmslim/layers/gemm/int8_utils.py"
ORIG = PATH + ".orig"
if not os.path.exists(ORIG): shutil.copy2(PATH, ORIG)
shutil.copy2(ORIG, PATH)
with open(PATH) as f: c = f.read()

inject = '''
# === HIP EXACT KERNEL WITH PROPER STREAM ===
import ctypes as _ct, subprocess as _sp, os as _os, torch as _torch
_L="/workspace/hip_kernels/libdsv4_ops_hip.so"
if not _os.path.exists(_L):
    _sp.run(["hipcc","-O3","--offload-arch=gfx936","-shared","-fPIC","-o",_L,"/workspace/hip_kernels/dsv4_ops_hip.hip"],check=True,capture_output=True)
_hl=_ct.CDLL(_L)
_hl.launch_ptq.argtypes=[_ct.c_void_p]*3+[_ct.c_int]*2+[_ct.c_void_p]
def _hip_per_token_quant_int8(x, scale_dtype=None, cal_sum=False):
    M=x.numel()//x.shape[-1]; N=x.shape[-1]
    xf=x.reshape(M,N).contiguous()
    q=_torch.empty(M,N,device=x.device,dtype=_torch.int8)
    s=_torch.empty(M,1,device=x.device,dtype=_torch.float32)
    # KEY FIX: use the actual current CUDA stream, not None (default stream 0)
    _stream = _ct.c_void_p(_torch.cuda.current_stream().cuda_stream)
    _hl.launch_ptq(xf.data_ptr(),q.data_ptr(),s.data_ptr(),M,N,_stream)
    return q.reshape(x.shape), s.reshape(x.shape[:-1]+(1,))
# === END ===

def _disabled_per_token_quant_int8(x):'''
c = inject + "\n" + c.replace("def per_token_quant_int8(x):", "def _triton_per_token_quant_int8(x):", 1)
c += '\ndef per_token_quant_int8(x):\n    return _hip_per_token_quant_int8(x)\n'
with open(PATH,"w") as f: f.write(c)
print("MODE=hip_stream_fixed (HIP kernel with correct stream)")
