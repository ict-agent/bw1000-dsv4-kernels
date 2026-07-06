"""Comprehensive multi-baseline kernel comparison.
Compares per_token_quant_int8 across:
  A. Default PyTorch (torch round/clamp)        — naive baseline
  B. SOTA: lmslim Triton (the engine's actual kernel) — current SOTA
  C. Our HIP exact (fused_ops_exact)             — our optimized
Also fused variants where applicable.
Produces one JSON with all numbers + correctness vs B as golden reference.
"""
import torch, ctypes, subprocess, os, json, time

OUT = "/workspace/multi_baseline_comparison.json"
def compile_exact():
    lib="/workspace/hip_kernels/libfused_ops_exact.so"
    src="/workspace/hip_kernels/fused_ops_exact.hip"
    if not os.path.exists(lib) or os.path.getmtime(src)>os.path.getmtime(lib):
        subprocess.run(["hipcc","-O3","--offload-arch=gfx936","-shared","-fPIC","-o",lib,src],check=True,capture_output=True)
    l=ctypes.CDLL(lib)
    l.launch_per_token_quant_int8_exact.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]
    l.launch_fused_silu_mul_quant_exact.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_void_p]
    l.launch_fused_add_rmsnorm_quant_exact.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*2+[ctypes.c_float,ctypes.c_void_p]
    return l

def bench(fn,w=30,r=400):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(r): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/r*1000

def py_default_quant(x):
    """A: default PyTorch naive per-token int8 quant."""
    M,N=x.shape
    xf=x.float()
    am=xf.abs().amax(-1,keepdim=True)
    s=am/127.0
    s=torch.where(s>0,s,torch.ones_like(s))
    q=(xf/s).round().clamp(-128,127).to(torch.int8)
    return q, s

def sota_quant(x):
    """B: SOTA — lmslim Triton (engine's kernel)."""
    from lmslim.layers.gemm.int8_utils import per_token_quant_int8
    return per_token_quant_int8(x)

def hip_quant(lib,x):
    M=x.numel()//x.shape[-1]; N=x.shape[-1]
    xf=x.reshape(M,N).contiguous()
    q=torch.empty(M,N,device=x.device,dtype=torch.int8)
    s=torch.empty(M,device=x.device,dtype=torch.float32)
    lib.launch_per_token_quant_int8_exact(xf.data_ptr(),q.data_ptr(),s.data_ptr(),M,N,None)
    return q.reshape(x.shape), s.reshape(x.shape[:-1]+(1,))

def main():
    lib=compile_exact()
    R={"device":torch.cuda.get_device_name(0),"kernels":[]}
    print("="*72); print("MULTI-BASELINE per_token_quant_int8"); print("="*72)
    for M in [1,16,64,256,1024]:
        for N in [4096,2048]:
            x=torch.randn(M,N,device="cuda",dtype=torch.bfloat16)
            # golden = SOTA (B)
            gq,gs=sota_quant(x); gq=gq.reshape(M,N); gs=gs.reshape(M)
            # A
            aq,as_=py_default_quant(x)
            ms_a=bench(lambda: py_default_quant(x))
            # B
            ms_b=bench(lambda: sota_quant(x))
            # C
            cq,cs=hip_quant(lib,x)
            ms_c=bench(lambda: hip_quant(lib,x))
            row={
                "M":M,"N":N,
                "A_default_pytorch_ms":round(ms_a,4),
                "B_sota_triton_ms":round(ms_b,4),
                "C_hip_exact_ms":round(ms_c,4),
                "C_vs_B_speedup":round(ms_b/ms_c,2),
                "C_vs_A_speedup":round(ms_a/ms_c,2),
                "C_bitexact_vs_B":(gq==cq.reshape(M,N)).all().item(),
                "A_maxdiff_vs_B":(gq.int()-aq.reshape(M,N).int()).abs().max().item(),
            }
            R["kernels"].append(row)
            print("M=%4d N=%4d: A=%.4f B=%.4f C=%.4f | C/B=%.2fx C/A=%.2fx bitexact=%s"%(
                M,N,ms_a,ms_b,ms_c,ms_b/ms_c,ms_a/ms_c,row["C_bitexact_vs_B"]))
    with open(OUT,"w") as f: json.dump(R,f,indent=2)
    print("Saved:",OUT)
if __name__=="__main__": main()
