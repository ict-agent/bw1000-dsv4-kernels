"""Engine-integration-style A/B: simulate the EXACT call pattern the engine
makes during a decode step, measuring SOTA kernel chain vs HIP kernel chain.
This is the real integration performance delta (no multi-process fragility).

Reproduces: post-attn residual+add -> rmsnorm -> quant -> [MoE] -> silu_mul -> quant
"""
import torch, ctypes, json, time
from lmslim.layers.gemm.int8_utils import per_token_quant_int8 as sota_q
import lightop

lib=ctypes.CDLL("/workspace/hip_kernels/libdsv4_ops_hip.so")
lib.launch_ptq.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]
lib.launch_add_rmsnorm_quant.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int]*2+[ctypes.c_float,ctypes.c_void_p]
lib.launch_silu_mul_quant.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_void_p]

def bench(fn,w=50,r=500):
    for _ in range(w):fn()
    torch.cuda.synchronize();t0=time.time()
    for _ in range(r):fn()
    torch.cuda.synchronize();return (time.time()-t0)/r*1000

eps=1e-6
R={"label":"engine_integration_AB","tests":[]}
print("="*80); print("ENGINE-INTEGRATION A/B (real call pattern)"); print("="*80)

for M in [1,8,32,64]:
    N=4096; I=2048
    res=torch.randn(M,N,device="cuda",dtype=torch.bfloat16)
    x=torch.randn(M,N,device="cuda",dtype=torch.bfloat16)
    w=torch.randn(N,device="cuda",dtype=torch.bfloat16)
    gate=torch.randn(M,I,device="cuda",dtype=torch.bfloat16)
    up=torch.randn(M,I,device="cuda",dtype=torch.bfloat16)

    # Config A: SOTA chain (what engine does now)
    def configA():
        r=res.clone();xc=x.clone()
        n,ro=lightop.fused_add_rms_norm(xc,r,w,eps)      # C++ add+rmsnorm
        qa,sa=sota_q(n)                                    # Triton quant
        # (GEMM here is identical for both - skip, it's rocBLAS)
        h=torch.nn.functional.silu(gate.float())*up.float()# silu*mul
        hbf=h.to(torch.bfloat16)
        qb,sb=sota_q(hbf)                                  # Triton quant
        return qa,qb

    # Config B: HIP chain (our kernels)
    qa_h=torch.empty(M,N,device="cuda",dtype=torch.int8)
    sa_h=torch.empty(M,device="cuda",dtype=torch.float32)
    qb_h=torch.empty(M,I,device="cuda",dtype=torch.int8)
    sb_h=torch.empty(M,device="cuda",dtype=torch.float32)
    def configB():
        r=res.clone()
        lib.launch_add_rmsnorm_quant(r.data_ptr(),x.data_ptr(),w.data_ptr(),qa_h.data_ptr(),sa_h.data_ptr(),M,N,eps,None)
        lib.launch_silu_mul_quant(gate.data_ptr(),up.data_ptr(),qb_h.data_ptr(),sb_h.data_ptr(),M,I,None)
        return qa_h,qb_h

    msA=bench(configA); msB=bench(configB)
    sp=msA/msB
    print("M=%3d: SOTA=%.4fms HIP=%.4fms speedup=%.2fx"%(M,msA,msB,sp))
    R["tests"].append({"M":M,"configA_ms":round(msA,4),"configB_ms":round(msB,4),"speedup":round(sp,2)})

# Correctness: both should produce same quant (bit-exact chain)
M=64;N=4096;I=2048
res=torch.randn(M,N,device="cuda",dtype=torch.bfloat16)
x=torch.randn(M,N,device="cuda",dtype=torch.bfloat16)
w=torch.randn(N,device="cuda",dtype=torch.bfloat16)
gate=torch.randn(M,I,device="cuda",dtype=torch.bfloat16)
up=torch.randn(M,I,device="cuda",dtype=torch.bfloat16)
qaA,qbA=configA() if False else (None,None)
# do fresh
r=res.clone();xc=x.clone()
nA,_=lightop.fused_add_rms_norm(xc,r,w,eps)
qaA,saA=sota_q(nA)
hA=torch.nn.functional.silu(gate.float())*up.float()
qbA,sbA=sota_q(hA.to(torch.bfloat16))
r2=res.clone()
lib.launch_add_rmsnorm_quant(r2.data_ptr(),x.data_ptr(),w.data_ptr(),qa_h.data_ptr(),sa_h.data_ptr(),M,N,eps,None)
lib.launch_silu_mul_quant(gate.data_ptr(),up.data_ptr(),qb_h.data_ptr(),sb_h.data_ptr(),M,I,None)
torch.cuda.synchronize()
da=(qaA.reshape(M,N).int()-qa_h.int()).abs().max().item()
db=(qbA.reshape(M,I).int()-qb_h.int()).abs().max().item()
print("\nCorrectness: add_rmsnorm_quant maxdiff=%d, silu_mul_quant maxdiff=%d"%(da,db))
R["correctness"]={"add_rmsnorm_quant_maxdiff":da,"silu_mul_quant_maxdiff":db}
with open("/workspace/engine_integration_ab.json","w") as f:json.dump(R,f,indent=2)
print("Saved: /workspace/engine_integration_ab.json")
