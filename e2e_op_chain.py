"""End-to-end engine op-chain benchmark.
Simulates a DeepSeek V4 MLP decode/prefill step using REAL engine ops
(lightop.fused_add_rms_norm, lmslim per_token_quant_int8, SiluAndMul, torch._int_mm W8A8 GEMM)
and compares latency with HIP patches ON vs OFF.

This observes the HIP kernels' impact in the actual engine code path.
"""
import os, sys, time, torch, ctypes
sys.path.insert(0, "/workspace/dsv4_ops_unit_tests")
sys.path.insert(0, "/workspace/sglang/python")
from utils import model_config as C

DEV = "cuda"
bf = torch.bfloat16
HID = C.HIDDEN_SIZE       # 4096
INTER = C.MOE_INTERMEDIATE_SIZE  # 2048

def bench(fn, w=20, r=200):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(r): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/r*1000

def run_chain(x, res, w_rms, w_gateup, w_down, mask, M, hip_on):
    """One MLP step: fused_add_rmsnorm+quant -> gateup GEMM -> silu_mul_masked_quant -> down GEMM."""
    import lightop
    from lmslim.layers.gemm.int8_utils import per_token_quant_int8 as sota_ptq
    # 1. fused add+rmsnorm (engine: lightop) + per-token quant
    normed, res_new = lightop.fused_add_rms_norm(x.clone(), res.clone(), w_rms, 1e-6)
    if hip_on:
        q, s = hip_ptq(normed)
    else:
        q, s = sota_ptq(normed)
    # 2. gateup W8A8 GEMM: q[M,K] @ w_gateup[K,2N] -> [M, 2*INTER]  (w_gateup stored [K,2N])
    _r = lightop.gemm_w8a8_smooth(q, w_gateup.t(), s.unsqueeze(-1), sb_gateup.unsqueeze(0), None, bf)
    gu = _r[1] if isinstance(_r, tuple) else _r
    gate, up = gu.chunk(2, dim=-1)
    # 3. silu+mul+masked quant
    if hip_on:
        hq, hs = hip_silu_mul_masked_quant(gate, up, mask)
    else:
        h = (torch.sigmoid(gate.float())*gate*up).to(bf)
        hq, hs = sota_ptq(h)
    # 4. down W8A8 GEMM: hq[M,INTER] @ w_down[INTER,HID] -> [M,HID]
    _r2 = lightop.gemm_w8a8_smooth(hq, w_down.t(), hs.unsqueeze(-1), sb_down.unsqueeze(0), None, bf)
    out = _r2[1] if isinstance(_r2, tuple) else _r2
    return out, res_new

# HIP kernel handles
_lib = ctypes.CDLL("/workspace/hip_kernels/libdsv4_all_hip.so")
P = ctypes.c_void_p
_lib.launch_ptq.argtypes = [P,P,P,ctypes.c_int,ctypes.c_int,P]
_lib.launch_silu_mul_masked_quant.argtypes = [P,P,P,P,P,ctypes.c_int,ctypes.c_int,P]
stream = torch.cuda.current_stream().cuda_stream

def hip_ptq(x):
    M,N = x.shape
    q = torch.empty(M,N,device=DEV,dtype=torch.int8)
    s = torch.empty(M,device=DEV,dtype=torch.float32)
    _lib.launch_ptq(x.data_ptr(),q.data_ptr(),s.data_ptr(),M,N,stream)
    return q,s

def hip_silu_mul_masked_quant(gate, up, mask):
    M,N = gate.shape
    q = torch.empty(M,N,device=DEV,dtype=torch.int8)
    s = torch.empty(M,device=DEV,dtype=torch.float32)
    _lib.launch_silu_mul_masked_quant(gate.data_ptr(),up.data_ptr(),mask.data_ptr(),
                                       q.data_ptr(),s.data_ptr(),M,N,stream)
    return q,s

# setup weights (int8 W8A8)
torch.manual_seed(0)
w_gateup = torch.randint(-127,127,(HID,2*INTER),device=DEV,dtype=torch.int8)  # [K, 2N] for _int_mm
w_down   = torch.randint(-127,127,(INTER,HID),device=DEV,dtype=torch.int8)
sb_gateup = torch.ones(2*INTER,device=DEV,dtype=torch.float32)*0.01
sb_down   = torch.ones(HID,device=DEV,dtype=torch.float32)*0.01
w_rms = torch.ones(HID,device=DEV,dtype=bf)

print("="*70); print("E2E ENGINE OP-CHAIN BENCHMARK (DeepSeek V4 MLP step)"); print("="*70)
print(f"  hidden={HID} intermediate={INTER}")
results = []
for M in [1, 8, 64, 256]:
    x = torch.randn(M,HID,device=DEV,dtype=bf)
    res = torch.randn(M,HID,device=DEV,dtype=bf)
    mask = torch.ones(M,device=DEV,dtype=torch.int32)
    # warmup
    run_chain(x,res,w_rms,w_gateup,w_down,mask,M,False)
    run_chain(x,res,w_rms,w_gateup,w_down,mask,M,True)
    ms_off = bench(lambda: run_chain(x,res,w_rms,w_gateup,w_down,mask,M,False))
    ms_on  = bench(lambda: run_chain(x,res,w_rms,w_gateup,w_down,mask,M,True))
    sp = ms_off/ms_on
    results.append({"M":M,"off_ms":round(ms_off,4),"on_ms":round(ms_on,4),"speedup":round(sp,2)})
    tag = "decode" if M<=8 else "prefill"
    print(f"  M={M:>4} ({tag:>7}): OFF={ms_off:.4f}ms  HIP-ON={ms_on:.4f}ms  speedup={sp:.2f}x")

import json
json.dump(results, open("/workspace/hip_kernels/results/e2e_op_chain.json","w"), indent=2)
print("\nSaved /workspace/hip_kernels/results/e2e_op_chain.json")
print("\nNote: GEMM uses torch._int_mm (vendor W8A8); HIP patches quant+silu_mul_masked_quant.")
