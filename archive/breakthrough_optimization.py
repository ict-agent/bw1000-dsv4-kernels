"""Try multiple GEMM backends + 1-pass quant + fused chain optimization."""
import torch, ctypes, time, json
import lightop
from lmslim.layers.gemm.int8_utils import per_token_quant_int8 as sota_q
from lmslim import quant_ops

def bench(fn, w=30, r=400):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(r): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/r*1000

R = {"tests": []}

# ============================================================
# 1. Try lightop gemm_w8a8_smooth directly (understand its API)
# ============================================================
print("="*72)
print("1. lightop gemm_w8a8_smooth API exploration")
print("="*72)
M,N,K = 64,4096,4096
a = torch.randint(-128,127,(M,K),device="cuda",dtype=torch.int8).contiguous()
b = torch.randint(-128,127,(K,N),device="cuda",dtype=torch.int8).t()  # [N,K] non-contig
sa = torch.ones(M,1,device="cuda",dtype=torch.float32)
sb = torch.ones(1,N,device="cuda",dtype=torch.float32)
result = lightop.gemm_w8a8_smooth(a, b, sa, sb, None, torch.bfloat16)
print(f"  result type: {type(result)}")
if isinstance(result, tuple):
    print(f"  tuple len: {len(result)}")
    for i, r in enumerate(result):
        if isinstance(r, torch.Tensor):
            print(f"    [{i}] shape={r.shape} dtype={r.dtype}")
else:
    print(f"  shape: {result.shape}")

# ============================================================
# 2. Compare all W8A8 GEMM backends
# ============================================================
print("\n" + "="*72)
print("2. W8A8 GEMM backend comparison")
print("="*72)
print(f"{'M':>5} {'N':>5} {'K':>5} {'lightop_smooth':>16} {'lightop_asm':>14} {'triton_scaled':>14} {'hipBLASLt':>10}")

for M,N,K in [(1,4096,4096),(6,4096,2048),(64,4096,4096),(256,4096,4096)]:
    a = torch.randint(-128,127,(M,K),device="cuda",dtype=torch.int8).contiguous()
    b_nc = torch.randint(-128,127,(K,N),device="cuda",dtype=torch.int8).t()
    b_asm = torch.randint(-128,127,(N,K),device="cuda",dtype=torch.int8)  # contig for asm? need to check
    sa = torch.ones(M,1,device="cuda",dtype=torch.float32)
    sb = torch.ones(1,N,device="cuda",dtype=torch.float32)

    # lightop_smooth
    try:
        ms_smooth = bench(lambda: lightop.gemm_w8a8_smooth(a, b_nc, sa, sb, None, torch.bfloat16))
    except: ms_smooth = -1

    # lightop_asm
    try:
        out_asm = lightop.gemm_w8a8_asm(a, b_asm, sa.squeeze(-1), sb.squeeze(0), 128, torch.bfloat16)
        ms_asm = bench(lambda: lightop.gemm_w8a8_asm(a, b_asm, sa.squeeze(-1), sb.squeeze(0), 128, torch.bfloat16))
    except Exception as e:
        ms_asm = -1

    # triton_scaled_mm (lmslim)
    try:
        ms_triton = bench(lambda: quant_ops.triton_scaled_mm(a, b_nc, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
    except: ms_triton = -1

    # hipBLASLt (via torch._int_mm + manual scale)
    try:
        # torch._int_mm does INT8 GEMM: C = A @ B^T, output int32
        # Then apply scales manually
        b_for_intmm = torch.randint(-128,127,(N,K),device="cuda",dtype=torch.int8).contiguous()
        def hipblaslt_fn():
            c_int = torch._int_mm(a, b_for_intmm.T)
            return (c_int.float() * sa * sb.T).to(torch.bfloat16)
        ms_hipblaslt = bench(hipblaslt_fn)
    except: ms_hipblaslt = -1

    print(f"{M:>5} {N:>5} {K:>5} {ms_smooth:>14.4f}ms {ms_asm:>12.4f}ms {ms_triton:>12.4f}ms {ms_hipblaslt:>8.4f}ms")
    R["tests"].append({"M":M,"N":N,"K":K,
        "lightop_smooth":round(ms_smooth,4) if ms_smooth>0 else None,
        "lightop_asm":round(ms_asm,4) if ms_asm>0 else None,
        "triton_scaled":round(ms_triton,4) if ms_triton>0 else None,
        "hipblaslt_intmm":round(ms_hipblaslt,4) if ms_hipblaslt>0 else None})

# ============================================================
# 3. 1-pass online quantization (breakthrough attempt)
# ============================================================
print("\n" + "="*72)
print("3. 1-pass online quantization vs 2-pass HIP")
print("="*72)

# Load our HIP native ext
import sys
sys.path.insert(0, "/workspace/hip_kernels/torch_ext_build")
import dsv4_native_ext as ext

for M in [1, 64, 256]:
    N = 4096
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)

    # SOTA: lmslim Triton (2-pass)
    ms_sota = bench(lambda: sota_q(x))

    # Our HIP native ext (2-pass)
    stream = torch.cuda.current_stream().cuda_stream
    ms_hip = bench(lambda: ext.per_token_quant_int8_stream(x, stream))

    # PyTorch fused (1-pass via torch.compile or custom)
    # Online approach: estimate scale from first chunk, refine
    def online_quant(x):
        # Process in chunks of 128, track running max
        M, N = x.shape
        chunk = 128
        xq = torch.empty_like(x, dtype=torch.int8)
        xs = torch.empty(M, 1, device=x.device, dtype=torch.float32)
        for i in range(0, N, chunk):
            c = x[:, i:i+chunk].float()
            am = c.abs().amax(-1, keepdim=True)
            if i == 0:
                running_max = am
            else:
                running_max = torch.maximum(running_max, am)
        # Final quant with running max
        scale = running_max / 127.0
        scale = torch.clamp(scale, min=1e-10)
        xq = (x.float() / scale).round().clamp(-128, 127).to(torch.int8)
        xs = scale
        return xq, xs

    ms_online = bench(lambda: online_quant(x))

    print(f"  M={M}: SOTA={ms_sota:.4f}ms HIP_2pass={ms_hip:.4f}ms ({ms_sota/ms_hip:.2f}x) Online_1pass={ms_online:.4f}ms")
    R["tests"].append({"kernel":"quant_comparison","M":M,
        "sota_ms":round(ms_sota,4),"hip_2pass_ms":round(ms_hip,4),
        "online_1pass_ms":round(ms_online,4)})

# ============================================================
# 4. Fused MoE elementwise chain (breakthrough: batch 2 quants)
# ============================================================
print("\n" + "="*72)
print("4. Fused MoE chain: quant→[GEMM]→silu_mul_quant (batch optimization)")
print("="*72)

for M in [1, 64, 256]:
    N = 4096; I = 2048
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    gate = torch.randn(M, I, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(M, I, device="cuda", dtype=torch.bfloat16)

    # Current: separate calls
    def separate_chain():
        q1, s1 = sota_q(x)
        # GEMM would go here (skip, just measure elementwise)
        q2, s2 = sota_q(torch.randn(M, I, device="cuda", dtype=torch.bfloat16))
        h = torch.nn.functional.silu(gate.float()) * up.float()
        q3, s3 = sota_q(h.to(torch.bfloat16))
        return q1, q2, q3

    # Optimized: fused silu_mul_quant + separate quant
    def fused_chain():
        q1, s1 = ext.per_token_quant_int8_stream(x, torch.cuda.current_stream().cuda_stream)
        q2, s2 = ext.silu_mul_quant_stream(gate, up, torch.cuda.current_stream().cuda_stream)
        return q1, q2

    ms_sep = bench(separate_chain)
    ms_fused = bench(fused_chain)
    print(f"  M={M}: separate={ms_sep:.4f}ms fused={ms_fused:.4f}ms speedup={ms_sep/ms_fused:.2f}x")
    R["tests"].append({"kernel":"moe_chain","M":M,
        "separate_ms":round(ms_sep,4),"fused_ms":round(ms_fused,4),
        "speedup":round(ms_sep/ms_fused,2)})

# ============================================================
# 5. Elementwise kernel block size optimization
# ============================================================
print("\n" + "="*72)
print("5. Block size optimization for per_token_quant")
print("="*72)

# Current: BS=256, EPT=16
# Try: load ctypes version with different configs
lib = ctypes.CDLL("/workspace/hip_kernels/libdsv4_ops_hip.so")
lib.launch_ptq.argtypes = [ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]

for M in [1, 64, 256]:
    N = 4096
    x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    q = torch.empty(M, N, device="cuda", dtype=torch.int8)
    s = torch.empty(M, device="cuda", dtype=torch.float32)

    ms_sota = bench(lambda: sota_q(x))
    ms_native = bench(lambda: ext.per_token_quant_int8_stream(x, torch.cuda.current_stream().cuda_stream))
    ms_ctypes = bench(lambda: lib.launch_ptq(x.data_ptr(), q.data_ptr(), s.data_ptr(), M, N, None))

    print(f"  M={M}: SOTA={ms_sota:.4f}ms native_ext={ms_native:.4f}ms ({ms_sota/ms_native:.2f}x) ctypes={ms_ctypes:.4f}ms ({ms_sota/ms_ctypes:.2f}x)")
    R["tests"].append({"kernel":"quant_backend_comparison","M":M,
        "sota_ms":round(ms_sota,4),"native_ms":round(ms_native,4),"ctypes_ms":round(ms_ctypes,4),
        "native_speedup":round(ms_sota/ms_native,2),"ctypes_speedup":round(ms_sota/ms_ctypes,2)})

with open("/workspace/breakthrough_attempts.json","w") as f:
    json.dump(R, f, indent=2, default=str)
print(f"\nSaved: /workspace/breakthrough_attempts.json")
