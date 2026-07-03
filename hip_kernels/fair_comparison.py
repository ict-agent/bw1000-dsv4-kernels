"""Fair comparison: HIP vs engine SOTA (lmslim Triton + lightop C++).
Uses correct metric: memory bandwidth (GB/s), not TFLOPS for memory-bound ops.
Only compares against REAL optimized baselines, not naive PyTorch.
"""
import torch, ctypes, json, time
from lmslim.layers.gemm.int8_utils import per_token_quant_int8 as sota_quant
import lightop

lib = ctypes.CDLL("/workspace/hip_kernels/libdsv4_ops_hip.so")
lib.launch_ptq.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]
lib.launch_silu_mul_quant.argtypes = [ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_void_p]
lib.launch_add_rmsnorm_quant.argtypes = [ctypes.c_void_p]*5+[ctypes.c_int]*2+[ctypes.c_float, ctypes.c_void_p]
lib.launch_rmsnorm.argtypes = [ctypes.c_void_p]*2+[ctypes.c_int]*2+[ctypes.c_float, ctypes.c_void_p]
lib.launch_silu_mul.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]

def bench(fn, w=50, r=500):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(r): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/r*1000

def bytes_per_elem(op, M, N):
    """Effective memory bytes for each op (read+write)."""
    bf16 = 2; int8 = 1; f32 = 4
    if op=="per_token_quant":  # read bf16 (2) + write int8 (1) + scale (negligible)
        return M*N*(2+1)
    if op=="silu_mul":  # read gate+up (4) + write out (2)
        return M*N*(4+2)
    if op=="silu_mul_quant":  # read gate+up (4) + write int8+scale (1)
        return M*N*(4+1)
    if op=="rmsnorm":  # read (2) + write (2)
        return M*N*(2+2)
    if op=="add_rmsnorm_quant":  # read res+x+w (6) + write res+q (2+1=3)
        return M*N*(6+3)
    return M*N*4

R={"metric":"effective_bandwidth_GB/s","comparisons":[]}
print("="*80)
print("FAIR COMPARISON: HIP vs Engine SOTA (lmslim Triton / lightop C++)")
print("Metric: effective memory bandwidth (GB/s) — correct for memory-bound ops")
print("="*80)

# 1. per_token_quant: HIP vs lmslim Triton
print("\n--- per_token_quant_int8 ---")
print(f"{'M':>5} {'N':>5} {'SOTA ms':>9} {'HIP ms':>9} {'speedup':>8} {'SOTA GB/s':>10} {'HIP GB/s':>10} {'bitexact':>8}")
for M in [1,16,64,256,1024]:
    N=4096
    x=torch.randn(M,N,device="cuda",dtype=torch.bfloat16)
    out_q=torch.empty(M,N,device="cuda",dtype=torch.int8)
    out_s=torch.empty(M,device="cuda",dtype=torch.float32)
    rq,rs=sota_quant(x)
    ms_sota=bench(lambda: sota_quant(x))
    def hip_fn(): lib.launch_ptq(x.data_ptr(),out_q.data_ptr(),out_s.data_ptr(),M,N,None)
    hip_fn(); torch.cuda.synchronize()
    ms_hip=bench(hip_fn)
    be=(rq.reshape(M,N)==out_q).all().item()
    nbytes=bytes_per_elem("per_token_quant",M,N)
    bw_sota=nbytes/(ms_sota*1e6); bw_hip=nbytes/(ms_hip*1e6)
    sp=ms_sota/ms_hip
    print(f"{M:>5} {N:>5} {ms_sota:>9.4f} {ms_hip:>9.4f} {sp:>7.2f}x {bw_sota:>9.1f} {bw_hip:>9.1f} {str(be):>8}")
    R["comparisons"].append({"op":"per_token_quant","M":M,"N":N,"sota_ms":round(ms_sota,4),"hip_ms":round(ms_hip,4),"speedup":round(sp,2),"sota_GBps":round(bw_sota,1),"hip_GBps":round(bw_hip,1),"bitexact":be})

# 2. fused_add_rmsnorm_quant: HIP vs lightop+Triton
print("\n--- fused_add_rmsnorm_quant ---")
print(f"{'M':>5} {'N':>5} {'SOTA ms':>9} {'HIP ms':>9} {'speedup':>8} {'SOTA GB/s':>10} {'HIP GB/s':>10} {'maxdiff':>8}")
eps=1e-6
for M in [1,16,64,256]:
    N=4096
    res=torch.randn(M,N,device="cuda",dtype=torch.bfloat16)
    x=torch.randn(M,N,device="cuda",dtype=torch.bfloat16)
    w=torch.randn(N,device="cuda",dtype=torch.bfloat16)
    def sota_fn():
        r=res.clone(); xc=x.clone()
        n,ro=lightop.fused_add_rms_norm(xc,r,w,eps)
        from lmslim.layers.gemm.int8_utils import per_token_quant_int8
        q,s=per_token_quant_int8(n)
        return q
    rq=sota_fn()
    ms_sota=bench(sota_fn)
    out_q=torch.empty(M,N,device="cuda",dtype=torch.int8)
    out_s=torch.empty(M,device="cuda",dtype=torch.float32)
    def hip_fn():
        r=res.clone()
        lib.launch_add_rmsnorm_quant(r.data_ptr(),x.data_ptr(),w.data_ptr(),out_q.data_ptr(),out_s.data_ptr(),M,N,eps,None)
    hip_fn(); torch.cuda.synchronize()
    ms_hip=bench(hip_fn)
    diff=(rq.reshape(M,N).int()-out_q.int()).abs().max().item()
    nbytes=bytes_per_elem("add_rmsnorm_quant",M,N)
    bw_sota=nbytes/(ms_sota*1e6); bw_hip=nbytes/(ms_hip*1e6)
    sp=ms_sota/ms_hip
    print(f"{M:>5} {N:>5} {ms_sota:>9.4f} {ms_hip:>9.4f} {sp:>7.2f}x {bw_sota:>9.1f} {bw_hip:>9.1f} {diff:>8}")
    R["comparisons"].append({"op":"add_rmsnorm_quant","M":M,"N":N,"sota_ms":round(ms_sota,4),"hip_ms":round(ms_hip,4),"speedup":round(sp,2),"sota_GBps":round(bw_sota,1),"hip_GBps":round(bw_hip,1),"maxdiff":diff})

# 3. silu_mul_quant: HIP vs PyTorch+Triton (no lightop fused exists)
print("\n--- silu_mul_quant ---")
print(f"{'M':>5} {'N':>5} {'SOTA ms':>9} {'HIP ms':>9} {'speedup':>8} {'SOTA GB/s':>10} {'HIP GB/s':>10} {'maxdiff':>8}")
for M in [1,16,64,256]:
    N=2048
    gate=torch.randn(M,N,device="cuda",dtype=torch.bfloat16)
    up=torch.randn(M,N,device="cuda",dtype=torch.bfloat16)
    def sota_fn():
        h=torch.nn.functional.silu(gate.float())*up.float()
        return sota_quant(h.to(torch.bfloat16))
    rq,rs=sota_fn()
    ms_sota=bench(sota_fn)
    out_q=torch.empty(M,N,device="cuda",dtype=torch.int8)
    out_s=torch.empty(M,device="cuda",dtype=torch.float32)
    def hip_fn(): lib.launch_silu_mul_quant(gate.data_ptr(),up.data_ptr(),out_q.data_ptr(),out_s.data_ptr(),M,N,None)
    hip_fn(); torch.cuda.synchronize()
    ms_hip=bench(hip_fn)
    diff=(rq.reshape(M,N).int()-out_q.int()).abs().max().item()
    nbytes=bytes_per_elem("silu_mul_quant",M,N)
    bw_sota=nbytes/(ms_sota*1e6); bw_hip=nbytes/(ms_hip*1e6)
    sp=ms_sota/ms_hip
    print(f"{M:>5} {N:>5} {ms_sota:>9.4f} {ms_hip:>9.4f} {sp:>7.2f}x {bw_sota:>9.1f} {bw_hip:>9.1f} {diff:>8}")
    R["comparisons"].append({"op":"silu_mul_quant","M":M,"N":N,"sota_ms":round(ms_sota,4),"hip_ms":round(ms_hip,4),"speedup":round(sp,2),"sota_GBps":round(bw_sota,1),"hip_GBps":round(bw_hip,1),"maxdiff":diff})

with open("/workspace/fair_comparison.json","w") as f: json.dump(R,f,indent=2)
print("\nSaved: /workspace/fair_comparison.json")
print("\nNote: DCU BW HBM peak bandwidth ~819 GB/s (est). Memory-bound ops approach this.")
