"""Build native ext with fused _int_mm + scale_convert for graph-safe W8A8 GEMM.
Also includes all elementwise kernels.
This is the final v2 native ext that wraps _int_mm for graph capture.
"""
import torch, sys, time, json, ctypes

sys.path.insert(0, "/workspace/hip_kernels/torch_ext_build")
import dsv4_native_ext as ext

# Load scale_convert library
sc_lib = ctypes.CDLL("/workspace/hip_kernels/libscale_convert.so")
sc_lib.launch_scale_convert.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_void_p]

from lmslim.layers.gemm.int8_utils import per_token_quant_int8 as sota_q
from lmslim import quant_ops
import lightop

def bench(fn, w=30, r=400):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(r): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/r*1000

R = {"tests": []}
eps = 1e-6

print("="*72)
print("FINAL ALL-KERNEL VERIFICATION (native ext + fused GEMM)")
print("="*72)

# 1. per_token_quant (graph-safe native ext)
print("\n--- 1. per_token_quant_int8 ---")
for M in [1, 64, 256]:
    N = 4096; x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    rq, rs = sota_q(x); rq = rq.reshape(M, N)
    stream = torch.cuda.current_stream().cuda_stream
    q, s = ext.per_token_quant_int8_stream(x, stream)
    be = (rq == q.reshape(M, N)).all().item()
    ms_sota = bench(lambda: sota_q(x))
    ms_hip = bench(lambda: ext.per_token_quant_int8_stream(x, stream))
    print(f"  M={M}: bit-exact={be} SOTA={ms_sota:.4f}ms HIP={ms_hip:.4f}ms speedup={ms_sota/ms_hip:.2f}x")
    R["tests"].append({"kernel": "per_token_quant", "M": M, "bitexact": bool(be), "speedup": round(ms_sota/ms_hip, 2)})

# 2. rmsnorm
print("\n--- 2. rmsnorm ---")
for M in [1, 64]:
    N = 512; x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    ref = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True)+eps)
    out = ext.rmsnorm_stream(x.clone(), eps, stream)
    diff = (ref.float()-out.float()).abs().max().item()
    ms_r = bench(lambda: x*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+eps))
    ms_h = bench(lambda: ext.rmsnorm_stream(x.clone(), eps, stream))
    print(f"  M={M}: maxdiff={diff:.4f} ref={ms_r:.4f}ms HIP={ms_h:.4f}ms speedup={ms_r/ms_h:.2f}x")
    R["tests"].append({"kernel": "rmsnorm", "M": M, "maxdiff": round(diff, 4), "speedup": round(ms_r/ms_h, 2)})

# 3. silu_and_mul
print("\n--- 3. silu_and_mul ---")
for M in [1, 64]:
    N = 2048; g = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    u = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    ref = (torch.sigmoid(g.float())*g*u).to(torch.bfloat16)
    out = ext.silu_and_mul_stream(g, u, stream)
    diff = (ref.float()-out.float()).abs().max().item()
    ms_r = bench(lambda: (torch.sigmoid(g.float())*g*u).to(torch.bfloat16))
    ms_h = bench(lambda: ext.silu_and_mul_stream(g, u, stream))
    print(f"  M={M}: maxdiff={diff:.6f} ref={ms_r:.4f}ms HIP={ms_h:.4f}ms speedup={ms_r/ms_h:.2f}x")
    R["tests"].append({"kernel": "silu_and_mul", "M": M, "maxdiff": round(diff, 6), "speedup": round(ms_r/ms_h, 2)})

# 4. silu_mul_quant
print("\n--- 4. silu_mul_quant ---")
for M in [1, 64]:
    N = 2048; g = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    u = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    h = (torch.sigmoid(g.float())*g*u).to(torch.bfloat16)
    rq, rs = sota_q(h); rq = rq.reshape(M, N)
    q, s = ext.silu_mul_quant_stream(g, u, stream)
    diff = (rq.int()-q.reshape(M, N).int()).abs().max().item()
    ms_r = bench(lambda: sota_q((torch.sigmoid(g.float())*g*u).to(torch.bfloat16)))
    ms_h = bench(lambda: ext.silu_mul_quant_stream(g, u, stream))
    print(f"  M={M}: maxdiff={diff} ref={ms_r:.4f}ms HIP={ms_h:.4f}ms speedup={ms_r/ms_h:.2f}x")
    R["tests"].append({"kernel": "silu_mul_quant", "M": M, "maxdiff": diff, "speedup": round(ms_r/ms_h, 2)})

# 5. add_rmsnorm_quant
print("\n--- 5. add_rmsnorm_quant ---")
for M in [1, 64]:
    N = 4096; res = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, device="cuda", dtype=torch.bfloat16).abs()+0.1
    r_ref = res.clone(); xc = x.clone()
    n, ro = lightop.fused_add_rms_norm(xc, r_ref, w, eps)
    rq, rs = sota_q(n); rq = rq.reshape(M, N)
    r_hip = res.clone()
    q, s = ext.add_rmsnorm_quant_stream(r_hip, x, w, eps, stream)
    diff = (rq.int()-q.reshape(M, N).int()).abs().max().item()
    def ref_fn():
        r = res.clone(); xc = x.clone()
        n, ro = lightop.fused_add_rms_norm(xc, r, w, eps)
        return sota_q(n)
    ms_r = bench(ref_fn)
    ms_h = bench(lambda: ext.add_rmsnorm_quant_stream(res.clone(), x, w, eps, stream))
    print(f"  M={M}: maxdiff={diff} SOTA={ms_r:.4f}ms HIP={ms_h:.4f}ms speedup={ms_r/ms_h:.2f}x")
    R["tests"].append({"kernel": "add_rmsnorm_quant", "M": M, "maxdiff": diff, "speedup": round(ms_r/ms_h, 2)})

# 6. W8A8 GEMM: _int_mm + HIP fused scale
print("\n--- 6. W8A8 GEMM (_int_mm + HIP fused scale) ---")
for M in [32, 64, 256]:
    N, K = 4096, 4096
    A = torch.randint(-128, 127, (M, K), device="cuda", dtype=torch.int8).contiguous()
    B = torch.randint(-128, 127, (K, N), device="cuda", dtype=torch.int8).contiguous()
    sa = torch.ones(M, device="cuda", dtype=torch.float32)
    sb = torch.ones(N, device="cuda", dtype=torch.float32)
    sa2d = sa.unsqueeze(-1); sb2d = sb.unsqueeze(0)
    B_nc = B.t()
    int_out = torch.empty(M, N, device="cuda", dtype=torch.int32)
    bf_out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)

    def fused_gemm():
        torch._int_mm(A, B, out=int_out)
        sc_lib.launch_scale_convert(int_out.data_ptr(), sa.data_ptr(), sb.data_ptr(), bf_out.data_ptr(), M, N, None)
        return bf_out

    ref = (A.float() @ B.float().T * sa2d * sb2d).to(torch.bfloat16)
    out = fused_gemm()
    diff = (ref.float() - out.float()).abs().max().item()

    ms_fused = bench(fused_gemm)
    ms_sota = bench(lambda: lightop.gemm_w8a8_smooth(A, B_nc, sa2d, sb2d, None, torch.bfloat16))
    ms_triton = bench(lambda: quant_ops.triton_scaled_mm(A, B_nc, scale_a=sa2d, scale_b=sb2d, out_dtype=torch.bfloat16))

    tops_fused = 2*M*N*K/ms_fused/1e9
    tops_sota = 2*M*N*K/ms_sota/1e9
    ratio = ms_sota/ms_fused
    print(f"  M={M}: diff={diff:.0f} fused={ms_fused:.4f}ms({tops_fused:.0f}T) sota={ms_sota:.4f}ms({tops_sota:.0f}T) triton={ms_triton:.4f}ms ratio={ratio:.2f}x")
    R["tests"].append({"kernel": "w8a8_gemm_fused", "M": M, "diff": diff, "fused_ms": round(ms_fused, 4), "sota_ms": round(ms_sota, 4), "ratio": round(ratio, 2)})

# Summary
print("\n" + "="*72)
print("SUMMARY")
print("="*72)
correct = sum(1 for t in R["tests"] if t.get("bitexact") or t.get("maxdiff", 999) < 2 or t.get("diff", 999) < 100)
fast = sum(1 for t in R["tests"] if t.get("speedup", 0) >= 1.5)
total = len(R["tests"])
print(f"  Total: {total}, Correct: {correct}, >=1.5x: {fast}")
for t in R["tests"]:
    print(f"  {t['kernel']:<25} M={t.get('M','?')} speedup={t.get('speedup', t.get('ratio','?'))} correct={t.get('bitexact') or t.get('maxdiff', t.get('diff', '?')) < 2 if isinstance(t.get('maxdiff', t.get('diff')), (int, float)) else '?'}")

with open("/workspace/final_all_kernel_verify.json", "w") as f:
    json.dump(R, f, indent=2, default=str)
print(f"\nSaved: /workspace/final_all_kernel_verify.json")
