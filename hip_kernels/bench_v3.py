"""V1 vs V3 HIP kernel comparison benchmark."""
import torch, ctypes, time

def bench(fn, warmup=30, repeat=500):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / repeat * 1000

lib1 = ctypes.CDLL("/workspace/hip_kernels/libfused_ops.so")
lib3 = ctypes.CDLL("/workspace/hip_kernels/libfused_ops_v3.so")
lib1.launch_fused_add_rmsnorm_quant.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*2 + [ctypes.c_float, ctypes.c_void_p]
lib1.launch_fused_silu_mul_quant.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_void_p]
lib3.launch_fused_add_rmsnorm_quant_v3.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*2 + [ctypes.c_float, ctypes.c_void_p]
lib3.launch_fused_silu_mul_quant_v3.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_void_p]
lib3.launch_per_token_quant_int8_v3.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*2 + [ctypes.c_void_p]

from lmslim.layers.gemm.int8_utils import per_token_quant_int8
import lightop

print("=" * 70)
print("V1 vs V3 HIP Kernel Comparison (2-pass optimized)")
print("=" * 70)
eps = 1e-6

print("\n--- fused_add_rmsnorm_quant ---")
for M in [1, 16, 64, 256, 1024]:
    N = 4096
    residual = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
    x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(N, device='cuda', dtype=torch.bfloat16).abs() + 0.1
    out_q = torch.empty(M, N, device='cuda', dtype=torch.int8)
    out_s = torch.empty(M, device='cuda', dtype=torch.float32)

    def base():
        r = residual.clone(); xc = x.clone()
        lightop.fused_add_rms_norm(xc, r, w, eps)
        per_token_quant_int8(xc)
    ms_base = bench(base)

    def v1():
        r = residual.clone()
        lib1.launch_fused_add_rmsnorm_quant(r.data_ptr(), x.data_ptr(), w.data_ptr(), out_q.data_ptr(), out_s.data_ptr(), M, N, eps, None)
    ms_v1 = bench(v1)

    def v3():
        r = residual.clone()
        lib3.launch_fused_add_rmsnorm_quant_v3(r.data_ptr(), x.data_ptr(), w.data_ptr(), out_q.data_ptr(), out_s.data_ptr(), M, N, eps, None)
    ms_v3 = bench(v3)

    sp_v1 = ms_base / ms_v1
    sp_v3 = ms_base / ms_v3
    v3_vs_v1 = ms_v1 / ms_v3
    print("  M=%4d: base=%.4fms v1=%.4fms(%.1fx) v3=%.4fms(%.1fx) v3_vs_v1=%.2fx" % (M, ms_base, ms_v1, sp_v1, ms_v3, sp_v3, v3_vs_v1))

print("\n--- fused_silu_mul_quant ---")
for M in [1, 16, 64, 256, 1024]:
    N = 2048
    gate = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
    up = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
    out_q = torch.empty(M, N, device='cuda', dtype=torch.int8)
    out_s = torch.empty(M, device='cuda', dtype=torch.float32)

    def base_s():
        h = torch.nn.functional.silu(gate.float()) * up.float()
        per_token_quant_int8(h.to(torch.bfloat16))
    ms_base = bench(base_s)

    def v1_s():
        lib1.launch_fused_silu_mul_quant(gate.data_ptr(), up.data_ptr(), out_q.data_ptr(), out_s.data_ptr(), M, N, None)
    ms_v1 = bench(v1_s)

    def v3_s():
        lib3.launch_fused_silu_mul_quant_v3(gate.data_ptr(), up.data_ptr(), out_q.data_ptr(), out_s.data_ptr(), M, N, None)
    ms_v3 = bench(v3_s)

    sp_v1 = ms_base / ms_v1
    sp_v3 = ms_base / ms_v3
    v3_vs_v1 = ms_v1 / ms_v3
    print("  M=%4d: base=%.4fms v1=%.4fms(%.1fx) v3=%.4fms(%.1fx) v3_vs_v1=%.2fx" % (M, ms_base, ms_v1, sp_v1, ms_v3, sp_v3, v3_vs_v1))

# Correctness
print("\n--- V3 Correctness Check ---")
M, N = 64, 4096
residual = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
w = torch.randn(N, device='cuda', dtype=torch.bfloat16).abs() + 0.1
r_ref = residual.clone(); xc = x.clone()
n_ref, _ = lightop.fused_add_rms_norm(xc, r_ref, w, eps)
ref_q, ref_s = per_token_quant_int8(n_ref)
ref_q = ref_q.reshape(M, N)

out_q = torch.empty(M, N, device='cuda', dtype=torch.int8)
out_s = torch.empty(M, device='cuda', dtype=torch.float32)
r3 = residual.clone()
lib3.launch_fused_add_rmsnorm_quant_v3(r3.data_ptr(), x.data_ptr(), w.data_ptr(), out_q.data_ptr(), out_s.data_ptr(), M, N, eps, None)
torch.cuda.synchronize()
diff = (out_q.int() - ref_q.int()).abs().max().item()
match = (out_q == ref_q).float().mean().item()
print("  fused_add_rmsnorm_quant_v3: max_diff=%d match=%.1f%% [%s]" % (diff, match*100, "PASS" if diff <= 1 else "FAIL"))

M, N = 64, 2048
gate = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
up = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
ref_h = torch.nn.functional.silu(gate.float()) * up.float()
ref_q2, _ = per_token_quant_int8(ref_h.to(torch.bfloat16))
ref_q2 = ref_q2.reshape(M, N)
out_q2 = torch.empty(M, N, device='cuda', dtype=torch.int8)
out_s2 = torch.empty(M, device='cuda', dtype=torch.float32)
lib3.launch_fused_silu_mul_quant_v3(gate.data_ptr(), up.data_ptr(), out_q2.data_ptr(), out_s2.data_ptr(), M, N, None)
torch.cuda.synchronize()
diff2 = (out_q2.int() - ref_q2.int()).abs().max().item()
print("  fused_silu_mul_quant_v3: max_diff=%d [%s]" % (diff2, "PASS" if diff2 <= 1 else "FAIL"))
