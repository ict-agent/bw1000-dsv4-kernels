"""Verify ALL HIP kernels vs Triton/TileLang/PyTorch references.
Tests correctness + performance for each kernel.
"""
import torch, ctypes, json, time, sys
sys.path.insert(0, "/workspace/dsv4_ops_unit_tests")
from utils import model_config as C

lib = ctypes.CDLL("/workspace/hip_kernels/libdsv4_all_hip.so")

# Setup argtypes
lib.launch_ptq.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]
lib.launch_ptgq.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p]
lib.launch_rmsnorm_self.argtypes = [ctypes.c_void_p]+[ctypes.c_int]*2+[ctypes.c_float, ctypes.c_void_p]
lib.launch_silu_mul.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]
lib.launch_silu_mul_masked_quant.argtypes = [ctypes.c_void_p]*5+[ctypes.c_int]*2+[ctypes.c_void_p]
lib.launch_act_quant_fp8.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]
lib.launch_topk_transform.argtypes = [ctypes.c_void_p]*4+[ctypes.c_int]*3+[ctypes.c_void_p]
lib.launch_mhc_pre.argtypes = [ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_void_p]
lib.launch_swa_prefill_indices.argtypes = [ctypes.c_void_p]+[ctypes.c_int]*3+[ctypes.c_void_p]
lib.launch_fused_rope.argtypes = [ctypes.c_void_p]*4+[ctypes.c_int]*5+[ctypes.c_void_p]

def bench(fn, w=20, r=200):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(r): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/r*1000

R = {"device": torch.cuda.get_device_name(0), "tests": []}
eps = 1e-6

print("="*72)
print("ALL HIP KERNELS VERIFICATION")
print("="*72)

# 1. per_token_quant_int8
print("\n--- 1. per_token_quant_int8 ---")
from lmslim.layers.gemm.int8_utils import per_token_quant_int8 as sota_ptq
for M in [1, 64, 256]:
    N = C.HIDDEN_SIZE
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    rq, rs = sota_ptq(x); rq=rq.reshape(M,N)
    hq = torch.empty(M,N,device="cuda",dtype=torch.int8)
    hs = torch.empty(M,device="cuda",dtype=torch.float32)
    lib.launch_ptq(x.data_ptr(),hq.data_ptr(),hs.data_ptr(),M,N,None)
    torch.cuda.synchronize()
    be = (rq==hq).all().item()
    ms_s = bench(lambda: sota_ptq(x))
    ms_h = bench(lambda: lib.launch_ptq(x.data_ptr(),hq.data_ptr(),hs.data_ptr(),M,N,None))
    print(f"  M={M}: bit-exact={be} SOTA={ms_s:.4f}ms HIP={ms_h:.4f}ms speedup={ms_s/ms_h:.2f}x")
    R["tests"].append({"kernel":"per_token_quant","M":M,"bitexact":be,"speedup":round(ms_s/ms_h,2)})

# 2. per_token_group_quant_int8
print("\n--- 2. per_token_group_quant_int8 ---")
from lmslim.layers.gemm.int8_utils import per_token_group_quant_int8 as sota_ptgq
for M in [1, 64]:
    N = C.HIDDEN_SIZE; gs = 128
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    rq, rs = sota_ptgq(x, gs); rq=rq.reshape(M,N)
    hq = torch.empty(M,N,device="cuda",dtype=torch.int8)
    ngroups = N//gs
    hs = torch.empty(M,ngroups,device="cuda",dtype=torch.float32)
    lib.launch_ptgq(x.data_ptr(),hq.data_ptr(),hs.data_ptr(),M,N,gs,None)
    torch.cuda.synchronize()
    diff = (rq.int()-hq.int()).abs().max().item()
    ms_s = bench(lambda: sota_ptgq(x, gs))
    ms_h = bench(lambda: lib.launch_ptgq(x.data_ptr(),hq.data_ptr(),hs.data_ptr(),M,N,gs,None))
    print(f"  M={M}: maxdiff={diff} SOTA={ms_s:.4f}ms HIP={ms_h:.4f}ms speedup={ms_s/ms_h:.2f}x")
    R["tests"].append({"kernel":"per_token_group_quant","M":M,"maxdiff":diff,"speedup":round(ms_s/ms_h,2)})

# 3. rmsnorm_self
print("\n--- 3. rmsnorm_self ---")
def ref_rmsnorm(x, eps):
    return x * torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+eps).to(x.dtype)
for M in [1, 64, 128]:
    N = C.HEAD_DIM  # 512
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    ref = ref_rmsnorm(x.clone(), eps)
    xc = x.clone()
    lib.launch_rmsnorm_self(xc.data_ptr(),M,N,eps,None)
    torch.cuda.synchronize()
    diff = (ref.float()-xc.float()).abs().max().item()
    ms_r = bench(lambda: ref_rmsnorm(x.clone(), eps))
    ms_h = bench(lambda: (xc.copy_(x), lib.launch_rmsnorm_self(xc.data_ptr(),M,N,eps,None))[1])
    print(f"  M={M}: maxdiff={diff:.4f} ref={ms_r:.4f}ms HIP={ms_h:.4f}ms speedup={ms_r/ms_h:.2f}x")
    R["tests"].append({"kernel":"rmsnorm_self","M":M,"maxdiff":round(diff,4),"speedup":round(ms_r/ms_h,2)})

# 4. silu_and_mul
print("\n--- 4. silu_and_mul ---")
for M in [1, 64, 256]:
    N = C.MOE_INTERMEDIATE_SIZE
    g = torch.randn(M,N,device="cuda",dtype=torch.bfloat16)
    u = torch.randn(M,N,device="cuda",dtype=torch.bfloat16)
    ref = (torch.sigmoid(g.float())*g*u).to(torch.bfloat16)
    out = torch.empty(M,N,device="cuda",dtype=torch.bfloat16)
    lib.launch_silu_mul(g.data_ptr(),u.data_ptr(),out.data_ptr(),M,N,None)
    torch.cuda.synchronize()
    diff = (ref.float()-out.float()).abs().max().item()
    ms_r = bench(lambda: (torch.sigmoid(g.float())*g*u).to(torch.bfloat16))
    ms_h = bench(lambda: lib.launch_silu_mul(g.data_ptr(),u.data_ptr(),out.data_ptr(),M,N,None))
    print(f"  M={M}: maxdiff={diff:.6f} ref={ms_r:.4f}ms HIP={ms_h:.4f}ms speedup={ms_r/ms_h:.2f}x")
    R["tests"].append({"kernel":"silu_and_mul","M":M,"maxdiff":round(diff,6),"speedup":round(ms_r/ms_h,2)})

# 5. act_quant_fp8 (NSA)
print("\n--- 5. act_quant (NSA FP8) ---")
for M in [1, 64]:
    N = C.HIDDEN_SIZE
    x = torch.randn(M,N,device="cuda",dtype=torch.bfloat16)
    xq = torch.empty(M,N,device="cuda",dtype=torch.bfloat16)
    s = torch.empty(M,device="cuda",dtype=torch.float32)
    lib.launch_act_quant_fp8(x.data_ptr(),xq.data_ptr(),s.data_ptr(),M,N,None)
    torch.cuda.synchronize()
    # ref: per-token absmax / 448
    am = x.float().abs().amax(-1,keepdim=True)
    ref_scale = am/448.0
    ref_scale = torch.clamp(ref_scale, min=1e-12)
    s_diff = (s.unsqueeze(-1)-ref_scale.squeeze(-1).unsqueeze(-1)).abs().max().item()
    ms_h = bench(lambda: lib.launch_act_quant_fp8(x.data_ptr(),xq.data_ptr(),s.data_ptr(),M,N,None))
    print(f"  M={M}: scale_diff={s_diff:.6f} HIP={ms_h:.4f}ms")
    R["tests"].append({"kernel":"act_quant_fp8","M":M,"scale_diff":round(s_diff,6),"hip_ms":round(ms_h,4)})

# 6. topk_transform_512
print("\n--- 6. topk_transform_512 ---")
batch=4; max_pages=512; page_size=1
slens = torch.tensor([100, 200, 512, 50], device="cuda", dtype=torch.int32)
ptabs = torch.zeros(batch, max_pages, device="cuda", dtype=torch.int32)
out = torch.full((batch, max_pages), -1, device="cuda", dtype=torch.int32)
lib.launch_topk_transform(None, slens.data_ptr(), ptabs.data_ptr(), out.data_ptr(), batch, max_pages, page_size, None)
torch.cuda.synchronize()
# Check: first n_valid elements should be 0..n_valid-1
ok = (out[0,:100] == torch.arange(100, device="cuda", dtype=torch.int32)).all().item() and (out[0,100:] == -1).all().item()
ms_h = bench(lambda: lib.launch_topk_transform(None, slens.data_ptr(), ptabs.data_ptr(), out.data_ptr(), batch, max_pages, page_size, None))
print(f"  correct={ok} HIP={ms_h:.4f}ms")
R["tests"].append({"kernel":"topk_transform","correct":ok,"hip_ms":round(ms_h,4)})

# 7. mhc_pre
print("\n--- 7. mhc_pre (sigmoid+mix) ---")
M=64; mhc_mult=C.HC_MULT  # 4
input_mix = torch.randn(M, mhc_mult, device="cuda", dtype=torch.bfloat16)
mhc_scale = torch.tensor([1.0], device="cuda", dtype=torch.float32)
mhc_base = torch.zeros(mhc_mult, device="cuda", dtype=torch.float32)
output_mix = torch.empty(M, mhc_mult, device="cuda", dtype=torch.bfloat16)
lib.launch_mhc_pre(input_mix.data_ptr(), mhc_scale.data_ptr(), mhc_base.data_ptr(), output_mix.data_ptr(), M, mhc_mult, None)
torch.cuda.synchronize()
# ref: sigmoid(x*scale+base)+eps
ref = (torch.sigmoid(input_mix.float()*mhc_scale[0]+mhc_base)+1e-6).to(torch.bfloat16)
diff = (ref.float()-output_mix.float()).abs().max().item()
ms_h = bench(lambda: lib.launch_mhc_pre(input_mix.data_ptr(), mhc_scale.data_ptr(), mhc_base.data_ptr(), output_mix.data_ptr(), M, mhc_mult, None))
print(f"  maxdiff={diff:.6f} HIP={ms_h:.4f}ms")
R["tests"].append({"kernel":"mhc_pre","maxdiff":round(diff,6),"hip_ms":round(ms_h,4)})

# 8. swa_prefill_indices
print("\n--- 8. swa_prefill_indices ---")
seq_len=1024; window=C.SLIDING_WINDOW  # 128
indices = torch.empty(seq_len, device="cuda", dtype=torch.int32)
lib.launch_swa_prefill_indices(indices.data_ptr(), seq_len, window, 1, None)
torch.cuda.synchronize()
# Check: first element should be 0, later ones should be max(0, i-window+1)
ref_start = [max(0, i-window+1) for i in range(seq_len)]
ok = (indices.cpu().numpy() == ref_start).all()
ms_h = bench(lambda: lib.launch_swa_prefill_indices(indices.data_ptr(), seq_len, window, 1, None))
print(f"  correct={ok} HIP={ms_h:.4f}ms")
R["tests"].append({"kernel":"swa_prefill_indices","correct":ok,"hip_ms":round(ms_h,4)})

# Summary
print("\n" + "="*72)
print("SUMMARY")
print("="*72)
correct = sum(1 for t in R["tests"] if t.get("bitexact") or t.get("maxdiff",999)<1 or t.get("correct"))
total = len(R["tests"])
fast = sum(1 for t in R["tests"] if t.get("speedup",0)>=1.5)
print(f"  Total: {total}, Correct: {correct}, >=1.5x: {fast}")
for t in R["tests"]:
    print(f"  {t['kernel']:<30} M={t.get('M','?')} speedup={t.get('speedup','?')} correct={t.get('bitexact') or t.get('maxdiff',t.get('correct'))<1 if isinstance(t.get('maxdiff',t.get('correct')),(int,float)) else '?'}")

with open("/workspace/all_kernels_verify.json","w") as f: json.dump(R,f,indent=2)
print(f"\nSaved: /workspace/all_kernels_verify.json")
