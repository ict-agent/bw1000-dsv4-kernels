"""Reduce native ext overhead: inline kernel launch without PyTorch dispatch.

The current native ext has 2.75x speedup vs ctypes 7.2x.
The gap is from PyTorch's Python→C++ dispatch overhead.

Solution: use torch.library.custom_op with a direct C function that
minimizes wrapper overhead, and also try direct ctypes with graph-safe
stream capture via torch.cuda.graph().
"""
import torch, ctypes, time, json

# Method 1: ctypes with torch.cuda.graph capture (graph-safe!)
# Key insight: if we capture the ctypes call inside a graph, it becomes graph-safe
# because the graph replays the actual kernel, not the Python call

lib = ctypes.CDLL("/workspace/hip_kernels/libdsv4_ops_hip.so")
lib.launch_ptq.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*2 + [ctypes.c_void_p]

# Method 2: native ext (current)
import sys
sys.path.insert(0, "/workspace/hip_kernels/torch_ext_build")
import dsv4_native_ext

# Method 3: Try using torch.cuda.graph for ctypes (capture once, replay)
class GraphQuantWrapper:
    def __init__(self, M, N, device="cuda"):
        self.M = M
        self.N = N
        self.x = torch.randn(M, N, device=device, dtype=torch.bfloat16)
        self.q = torch.empty(M, N, device=device, dtype=torch.int8)
        self.s = torch.empty(M, device=device, dtype=torch.float32)
        self.graph = None
        self.static_x = torch.randn(M, N, device=device, dtype=torch.bfloat16)

    def capture(self):
        # Warmup
        for _ in range(3):
            lib.launch_ptq(self.static_x.data_ptr(), self.q.data_ptr(), self.s.data_ptr(), self.M, self.N, None)
        torch.cuda.synchronize()

        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            lib.launch_ptq(self.static_x.data_ptr(), self.q.data_ptr(), self.s.data_ptr(), self.M, self.N, None)

    def run(self, x):
        self.static_x.copy_(x)
        self.graph.replay()
        return self.q.clone(), self.s.clone()

def bench(fn, w=30, r=500):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(r): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/r*1000

print("="*72)
print("NATIVE EXT OVERHEAD REDUCTION")
print("="*72)

from lmslim.layers.gemm.int8_utils import per_token_quant_int8 as sota

for M in [1, 64, 256]:
    N = 4096
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    q = torch.empty(M, N, device="cuda", dtype=torch.int8)
    s = torch.empty(M, device="cuda", dtype=torch.float32)

    # SOTA
    ms_sota = bench(lambda: sota(x))

    # Native ext (current, has overhead)
    stream = torch.cuda.current_stream().cuda_stream
    ms_native = bench(lambda: dsv4_native_ext.per_token_quant_int8_stream(x, stream))

    # ctypes direct (minimal overhead, not graph-safe normally)
    ms_ctypes = bench(lambda: lib.launch_ptq(x.data_ptr(), q.data_ptr(), s.data_ptr(), M, N, None))

    # Graph-captured ctypes (graph-safe!)
    gw = GraphQuantWrapper(M, N)
    gw.capture()
    ms_graph = bench(lambda: gw.run(x))

    print(f"\n  M={M}:")
    print(f"    SOTA Triton:      {ms_sota:.4f}ms  (1.00x)")
    print(f"    Native ext:       {ms_native:.4f}ms  ({ms_sota/ms_native:.2f}x)")
    print(f"    ctypes direct:    {ms_ctypes:.4f}ms  ({ms_sota/ms_ctypes:.2f}x)")
    print(f"    Graph-captured:   {ms_graph:.4f}ms  ({ms_sota/ms_graph:.2f}x)")

    # Check graph-captured correctness
    gq, gs = gw.run(x)
    rq, rs = sota(x)
    be = (gq.reshape(M,N) == rq.reshape(M,N)).all().item()
    print(f"    Graph bit-exact:  {be}")

# Also test: can we reduce native ext overhead by caching the stream lookup?
print("\n--- Overhead breakdown ---")
M = 64; N = 4096
x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

# Measure stream lookup overhead
def stream_lookup_only():
    return torch.cuda.current_stream().cuda_stream
ms_lookup = bench(stream_lookup_only)

# Measure data_ptr overhead
def data_ptr_only():
    return x.data_ptr()
ms_dataptr = bench(data_ptr_only)

# Measure empty alloc
def alloc_only():
    return torch.empty(M, N, device="cuda", dtype=torch.int8)
ms_alloc = bench(alloc_only)

print(f"  Stream lookup:  {ms_lookup:.4f}ms")
print(f"  data_ptr:       {ms_dataptr:.4f}ms")
print(f"  torch.empty:    {ms_alloc:.4f}ms")
print(f"  Total overhead: {ms_lookup+ms_dataptr+ms_alloc:.4f}ms")
print(f"  Native ext:     {ms_native:.4f}ms")
print(f"  ctypes:         {ms_ctypes:.4f}ms")
print(f"  Gap (native-ctypes): {ms_native-ms_ctypes:.4f}ms = overhead")

# Try: pre-allocate buffers + cache stream
q_pre = torch.empty(M, N, device="cuda", dtype=torch.int8)
s_pre = torch.empty(M, device="cuda", dtype=torch.float32)
cached_stream = torch.cuda.current_stream().cuda_stream

def ctypes_prealloc():
    lib.launch_ptq(x.data_ptr(), q_pre.data_ptr(), s_pre.data_ptr(), M, N, cached_stream)
ms_prealloc = bench(ctypes_prealloc)
print(f"\n  ctypes (prealloc+cached stream): {ms_prealloc:.4f}ms ({ms_sota/ms_prealloc:.2f}x)")
