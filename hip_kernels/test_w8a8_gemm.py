"""Test W8A8 GEMM kernel: correctness + performance vs SOTA."""
import torch, ctypes, time, json

lib = ctypes.CDLL("/workspace/hip_kernels/libw8a8_gemm.so")
lib.launch_w8a8_scaled_gemm.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*3 + [ctypes.c_void_p]

def w8a8_gemm_hip(A, B, scale_a, scale_b, M, N, K):
    """A: [M,K] int8, B: [N,K] int8, returns [M,N] bf16"""
    C = torch.empty(M, N, device=A.device, dtype=torch.bfloat16)
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    lib.launch_w8a8_scaled_gemm(
        A.contiguous().data_ptr(), B.contiguous().data_ptr(),
        scale_a.contiguous().data_ptr(), scale_b.contiguous().data_ptr(),
        C.data_ptr(), M, N, K, stream)
    return C

def bench(fn, w=20, r=200):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(r): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/r*1000

R = {"tests": []}
print("="*72)
print("W8A8 SCALED GEMM: HIP vs SOTA (lightop gemm_w8a8_smooth)")
print("="*72)

for M, N, K in [(1, 4096, 4096), (6, 4096, 2048), (16, 4096, 4096), (64, 4096, 4096)]:
    torch.manual_seed(42)
    A = torch.randint(-128, 127, (M, K), device="cuda", dtype=torch.int8).contiguous()
    B = torch.randint(-128, 127, (N, K), device="cuda", dtype=torch.int8).contiguous()
    sa = torch.ones(M, 1, device="cuda", dtype=torch.float32)
    sb = torch.ones(1, N, device="cuda", dtype=torch.float32)

    # Reference: dequant then matmul
    A_f = A.float() * sa
    B_f = B.float() * sb.T
    ref = (A_f @ B_f.T).to(torch.bfloat16)

    # HIP kernel
    hip_out = w8a8_gemm_hip(A, B, sa.squeeze(-1), sb.squeeze(0), M, N, K)
    torch.cuda.synchronize()
    diff = (hip_out.float() - ref.float()).abs().max().item()
    rel_err = diff / (ref.float().abs().max().item() + 1e-6)

    # Benchmark
    ms_hip = bench(lambda: w8a8_gemm_hip(A, B, sa.squeeze(-1), sb.squeeze(0), M, N, K))

    # SOTA: triton_scaled_mm (if available)
    try:
        from lmslim import quant_ops
        # B needs to be non-contiguous (transposed) for triton_scaled_mm
        B_nc = B.clone().t().t()  # trick to make non-contig? Actually need [N,K] non-contig
        B_t = B.clone()  # [N, K] contiguous
        # triton_scaled_mm expects b as [N, K] non-contiguous (i.e., [K, N].t())
        B_for_sota = torch.randint(-128, 127, (K, N), device="cuda", dtype=torch.int8).t()  # [N,K] non-contig
        sota_out = quant_ops.triton_scaled_mm(A, B_for_sota, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16)
        ms_sota = bench(lambda: quant_ops.triton_scaled_mm(A, B_for_sota, scale_a=sa, scale_b=sb, out_dtype=torch.bfloat16))
        sota_tops = 2*M*N*K/ms_sota/1e9
    except Exception as e:
        ms_sota = -1
        sota_tops = 0
        sota_out = None

    hip_tops = 2*M*N*K/ms_hip/1e9
    sp = ms_sota/ms_hip if ms_sota > 0 else 0

    print(f"  M={M:3d} N={N} K={K}: HIP={ms_hip:.4f}ms ({hip_tops:.1f} TOPS) SOTA={ms_sota:.4f}ms ({sota_tops:.1f} TOPS) speedup={sp:.2f}x diff={diff:.1f} rel={rel_err:.4f}")
    R["tests"].append({"M":M,"N":N,"K":K,"hip_ms":round(ms_hip,4),"sota_ms":round(ms_sota,4) if ms_sota>0 else None,
                       "hip_tops":round(hip_tops,1),"sota_tops":round(sota_tops,1),"speedup":round(sp,2),
                       "max_diff":round(diff,1),"rel_err":round(rel_err,4)})

with open("/workspace/w8a8_gemm_test.json","w") as f: json.dump(R,f,indent=2)
print(f"\nSaved: /workspace/w8a8_gemm_test.json")
