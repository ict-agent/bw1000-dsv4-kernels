"""Compare our HIP kernels against mmt-at/dsv4_ops_unit_tests reference impls.
Bit-exactness + performance, using the repo's own model_config shapes.
"""
import torch, ctypes, json, time, sys
sys.path.insert(0, "/workspace/dsv4_ops_unit_tests")
from utils import model_config as C

lib = ctypes.CDLL("/workspace/hip_kernels/libdsv4_ops_hip.so")
lib.launch_ptq.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*2 + [ctypes.c_void_p]
lib.launch_silu_mul.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*2 + [ctypes.c_void_p]
lib.launch_silu_mul_quant.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_void_p]
lib.launch_rmsnorm.argtypes = [ctypes.c_void_p]*2 + [ctypes.c_int]*2 + [ctypes.c_float, ctypes.c_void_p]
lib.launch_add_rmsnorm_quant.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*2 + [ctypes.c_float, ctypes.c_void_p]

def bench(fn, w=30, r=400):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(r): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/r*1000

def ref_rmsnorm(x, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)

def ref_quant(x):
    absmax = x.abs().amax(dim=-1, keepdim=True).float()
    scale = absmax / 127.0
    qx = (x.float() / scale).round().clamp(-128, 127).char()
    return qx, scale

def ref_silu_mul(gate, up):
    return (torch.sigmoid(gate.float()) * gate * up).to(torch.bfloat16)

R = {"device": torch.cuda.get_device_name(0), "tests": []}

print("="*72); print("HIP vs mmt-at/dsv4_ops_unit_tests reference"); print("="*72)

# 1. per_token_quant
print("\n--- per_token_quant_int8 ---")
for M in [1, 16, 64, 256, 1024]:
    N = C.HIDDEN_SIZE
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    rq, rs = ref_quant(x)
    hq = torch.empty(M, N, device="cuda", dtype=torch.int8)
    hs = torch.empty(M, 1, device="cuda", dtype=torch.float32)
    lib.launch_ptq(x.data_ptr(), hq.data_ptr(), hs.data_ptr(), M, N, None)
    torch.cuda.synchronize()
    be = (rq == hq).all().item()
    ms_r = bench(lambda: ref_quant(x))
    ms_h = bench(lambda: lib.launch_ptq(x.data_ptr(), hq.data_ptr(), hs.data_ptr(), M, N, None))
    sp = ms_r / ms_h
    print("  M=%4d N=%4d bitexact=%s ref=%.4fms hip=%.4fms speedup=%.2fx" % (M, N, be, ms_r, ms_h, sp))
    R["tests"].append({"op":"per_token_quant","M":M,"N":N,"bitexact":be,"ref_ms":round(ms_r,4),"hip_ms":round(ms_h,4),"speedup":round(sp,2)})

# 2. silu_mul (no quant)
print("\n--- silu_and_mul ---")
for M in [1, 16, 64, 256]:
    N = C.MOE_INTERMEDIATE_SIZE
    gate = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    rh = ref_silu_mul(gate, up)
    hh = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    lib.launch_silu_mul(gate.data_ptr(), up.data_ptr(), hh.data_ptr(), M, N, None)
    torch.cuda.synchronize()
    diff = (rh.float()-hh.float()).abs().max().item()
    ms_r = bench(lambda: ref_silu_mul(gate, up))
    ms_h = bench(lambda: lib.launch_silu_mul(gate.data_ptr(), up.data_ptr(), hh.data_ptr(), M, N, None))
    sp = ms_r / ms_h
    print("  M=%4d N=%4d maxdiff=%.6f ref=%.4fms hip=%.4fms speedup=%.2fx" % (M, N, diff, ms_r, ms_h, sp))
    R["tests"].append({"op":"silu_mul","M":M,"N":N,"maxdiff":round(diff,6),"ref_ms":round(ms_r,4),"hip_ms":round(ms_h,4),"speedup":round(sp,2)})

# 3. silu_mul_quant
print("\n--- silu_and_mul + quant (fused) ---")
for M in [1, 16, 64, 256]:
    N = C.MOE_INTERMEDIATE_SIZE
    gate = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    rh = ref_silu_mul(gate, up)
    rq, rs = ref_quant(rh)
    hq = torch.empty(M, N, device="cuda", dtype=torch.int8)
    hs = torch.empty(M, 1, device="cuda", dtype=torch.float32)
    lib.launch_silu_mul_quant(gate.data_ptr(), up.data_ptr(), hq.data_ptr(), hs.data_ptr(), M, N, None)
    torch.cuda.synchronize()
    diff = (rq.int()-hq.int()).abs().max().item()
    def ref_fused():
        h = ref_silu_mul(gate, up); return ref_quant(h)
    ms_r = bench(ref_fused)
    ms_h = bench(lambda: lib.launch_silu_mul_quant(gate.data_ptr(), up.data_ptr(), hq.data_ptr(), hs.data_ptr(), M, N, None))
    sp = ms_r / ms_h
    print("  M=%4d N=%4d maxdiff=%d ref=%.4fms hip=%.4fms speedup=%.2fx" % (M, N, diff, ms_r, ms_h, sp))
    R["tests"].append({"op":"silu_mul_quant","M":M,"N":N,"maxdiff":diff,"ref_ms":round(ms_r,4),"hip_ms":round(ms_h,4),"speedup":round(sp,2)})

# 4. rmsnorm
print("\n--- rmsnorm ---")
for M in [1, 64, 128]:
    N = C.HEAD_DIM
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    ro = ref_rmsnorm(x)
    ho = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    lib.launch_rmsnorm(x.data_ptr(), ho.data_ptr(), M, N, 1e-6, None)
    torch.cuda.synchronize()
    diff = (ro.float()-ho.float()).abs().max().item()
    ms_r = bench(lambda: ref_rmsnorm(x))
    ms_h = bench(lambda: lib.launch_rmsnorm(x.data_ptr(), ho.data_ptr(), M, N, 1e-6, None))
    sp = ms_r / ms_h
    print("  M=%4d N=%4d maxdiff=%.6f ref=%.4fms hip=%.4fms speedup=%.2fx" % (M, N, diff, ms_r, ms_h, sp))
    R["tests"].append({"op":"rmsnorm","M":M,"N":N,"maxdiff":round(diff,6),"ref_ms":round(ms_r,4),"hip_ms":round(ms_h,4),"speedup":round(sp,2)})

# 5. fused_add_rmsnorm_quant (engine path)
print("\n--- fused_add_rmsnorm_quant ---")
for M in [1, 16, 64, 256]:
    N = C.HIDDEN_SIZE
    res = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    # ref: res+x, rmsnorm(*w), quant
    added = res.float() + x.float()
    normed = added * torch.rsqrt(added.pow(2).mean(-1,keepdim=True)+1e-6) * w.float()
    rq, rs = ref_quant(normed.to(torch.bfloat16))
    hq = torch.empty(M, N, device="cuda", dtype=torch.int8)
    hs = torch.empty(M, 1, device="cuda", dtype=torch.float32)
    rc = res.clone()
    lib.launch_add_rmsnorm_quant(rc.data_ptr(), x.data_ptr(), w.data_ptr(), hq.data_ptr(), hs.data_ptr(), M, N, 1e-6, None)
    torch.cuda.synchronize()
    diff = (rq.int()-hq.int()).abs().max().item()
    def ref_fused2():
        a = res.float()+x.float()
        n = a*torch.rsqrt(a.pow(2).mean(-1,keepdim=True)+1e-6)*w.float()
        return ref_quant(n.to(torch.bfloat16))
    ms_r = bench(ref_fused2)
    ms_h = bench(lambda: (rc.copy_(res), lib.launch_add_rmsnorm_quant(rc.data_ptr(), x.data_ptr(), w.data_ptr(), hq.data_ptr(), hs.data_ptr(), M, N, 1e-6, None))[1])
    sp = ms_r / ms_h
    print("  M=%4d N=%4d maxdiff=%d ref=%.4fms hip=%.4fms speedup=%.2fx" % (M, N, diff, ms_r, ms_h, sp))
    R["tests"].append({"op":"add_rmsnorm_quant","M":M,"N":N,"maxdiff":diff,"ref_ms":round(ms_r,4),"hip_ms":round(ms_h,4),"speedup":round(sp,2)})

with open("/workspace/hip_vs_dsv4_ref.json", "w") as f: json.dump(R, f, indent=2)
print("\nSaved: /workspace/hip_vs_dsv4_ref.json")
print("\n=== SUMMARY ===")
for t in R["tests"]:
    op = t["op"]; M = t["M"]
    be = t.get("bitexact", t.get("maxdiff", "?") <= 1 if isinstance(t.get("maxdiff"), (int,float)) else "?")
    print("  %-20s M=%-4d speedup=%.2fx correct=%s" % (op, M, t["speedup"], be))
