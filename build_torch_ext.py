"""Build PyTorch C++ extension linking the HIP kernels.
This gives us proper stream handling + zero FFI overhead.
"""
import torch, os, subprocess
from torch.utils.cpp_extension import load

HIP_SRC = "/workspace/hip_kernels/dsv4_ops_hip.hip"
CPP_SRC = "/workspace/hip_kernels/dsv4_torch_ext.cpp"
BUILD_DIR = "/workspace/hip_kernels/torch_ext_build"
os.makedirs(BUILD_DIR, exist_ok=True)

# Compile HIP kernels to object file first
obj = os.path.join(BUILD_DIR, "dsv4_ops_hip.o")
if not os.path.exists(obj) or os.path.getmtime(HIP_SRC) > os.path.getmtime(obj):
    subprocess.run(["hipcc", "-O3", "--offload-arch=gfx936", "-c", "-fPIC",
                    "-o", obj, HIP_SRC], check=True)

# Build the extension linking the .o
ext = load(
    name="dsv4_hip_ext",
    sources=[CPP_SRC],
    extra_ldflags=[obj],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "--offload-arch=gfx936"],
    build_directory=BUILD_DIR,
    verbose=True,
)
print("Extension loaded:", ext)
print("Functions:", [x for x in dir(ext) if not x.startswith('_')])

# Quick test
import torch
x = torch.randn(64, 4096, device='cuda', dtype=torch.bfloat16)
result = ext.per_token_quant_int8(x)
print("per_token_quant_int8 test:", result[0].shape, result[1].shape)

# Verify bit-exact
from lmslim.layers.gemm.int8_utils import per_token_quant_int8 as sota
rq, rs = sota(x)
be = (rq.reshape(64, 4096) == result[0].reshape(64, 4096)).all().item()
print("bit-exact vs SOTA:", be)

# Benchmark: torch ext vs ctypes vs SOTA
import time, ctypes
def bench(fn, w=50, r=500):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(r): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/r*1000

ms_sota = bench(lambda: sota(x))
ms_ext = bench(lambda: ext.per_token_quant_int8(x))
print(f"SOTA Triton: {ms_sota:.4f}ms")
print(f"Torch ext:   {ms_ext:.4f}ms  speedup={ms_sota/ms_ext:.2f}x")

# Also test ctypes for comparison
lib = ctypes.CDLL("/workspace/hip_kernels/libdsv4_ops_hip.so")
lib.launch_ptq.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*2 + [ctypes.c_void_p]
q2 = torch.empty(64, 4096, device='cuda', dtype=torch.int8)
s2 = torch.empty(64, 1, device='cuda', dtype=torch.float32)
ms_ctypes = bench(lambda: lib.launch_ptq(x.data_ptr(), q2.data_ptr(), s2.data_ptr(), 64, 4096, None))
print(f"ctypes:      {ms_ctypes:.4f}ms  speedup={ms_sota/ms_ctypes:.2f}x")
print(f"\nTorch ext eliminates ctypes FFI overhead + uses correct stream")
