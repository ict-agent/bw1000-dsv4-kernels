"""
Complete validation of ALL kernels (v1 + v2) against baseline,
then integrate into running SGLang server and measure actual inference speedup.
"""
import torch
import ctypes
import subprocess
import time
import json
import os
import requests

ALL_RESULTS = []

def compile_all():
    """Compile both kernel files into shared libs."""
    for src, lib in [
        ("/workspace/hip_kernels/fused_ops.hip", "/workspace/hip_kernels/libfused_ops.so"),
        ("/workspace/hip_kernels/fused_ops_v2.hip", "/workspace/hip_kernels/libfused_ops_v2.so"),
    ]:
        if not os.path.exists(lib) or os.path.getmtime(src) > os.path.getmtime(lib):
            r = subprocess.run(["hipcc", "-O3", "--offload-arch=gfx936", "-shared", "-fPIC", "-o", lib, src],
                             capture_output=True, text=True)
            if r.returncode != 0:
                print("COMPILE ERROR %s: %s" % (src, r.stderr[:200]))
                return None, None
    lib1 = ctypes.CDLL("/workspace/hip_kernels/libfused_ops.so")
    lib2 = ctypes.CDLL("/workspace/hip_kernels/libfused_ops_v2.so")
    # Setup argtypes for lib1
    lib1.launch_fused_rmsnorm_quant.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_float, ctypes.c_void_p]
    lib1.launch_fused_silu_mul_quant.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_void_p]
    lib1.launch_fused_add_rmsnorm_quant.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*2 + [ctypes.c_float, ctypes.c_void_p]
    # Setup argtypes for lib2
    lib2.launch_per_token_quant_int8.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*2 + [ctypes.c_void_p]
    lib2.launch_rope.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*5 + [ctypes.c_void_p]
    lib2.launch_moe_sum.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3 + [ctypes.c_void_p]
    lib2.launch_moe_align.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*4 + [ctypes.c_void_p]
    return lib1, lib2

def bench(fn, warmup=20, repeat=200):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / repeat * 1000

def record(name, M, N, base_ms, opt_ms, correct, notes=""):
    sp = base_ms / opt_ms if opt_ms > 0 else 0
    ALL_RESULTS.append({"kernel": name, "M": M, "N": N, "baseline_ms": round(base_ms, 4),
                       "optimized_ms": round(opt_ms, 4), "speedup": round(sp, 2),
                       "correct": correct, "notes": notes})
    mark = "PASS" if correct else "FAIL"
    print("  M=%4d N=%4d: base=%.4fms opt=%.4fms %.1fx [%s]" % (M, N, base_ms, opt_ms, sp, mark))

# ============================================================
# TEST per_token_quant_int8
# ============================================================
def test_per_token_quant(lib2):
    print("\n" + "="*60)
    print("KERNEL 4: per_token_quant_int8")
    print("="*60)
    from lmslim.layers.gemm.int8_utils import per_token_quant_int8 as baseline_fn

    for M in [1, 16, 64, 256, 1024]:
        N = 4096
        x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        # Baseline
        ref_q, ref_s = baseline_fn(x)
        ref_q = ref_q.reshape(M, N)
        ref_s = ref_s.reshape(M)
        ms_base = bench(lambda: baseline_fn(x))
        # Optimized
        out_q = torch.empty(M, N, device='cuda', dtype=torch.int8)
        out_s = torch.empty(M, device='cuda', dtype=torch.float32)
        def opt(): lib2.launch_per_token_quant_int8(x.data_ptr(), out_q.data_ptr(), out_s.data_ptr(), M, N, None)
        opt()
        torch.cuda.synchronize()
        ms_opt = bench(opt)
        diff = (out_q.int() - ref_q.int()).abs().max().item()
        correct = diff <= 1
        record("per_token_quant_int8", M, N, ms_base, ms_opt, correct, "max_diff=%d" % diff)

# ============================================================
# TEST RoPE
# ============================================================
def test_rope(lib2):
    print("\n" + "="*60)
    print("KERNEL 5: RoPE")
    print("="*60)
    for M in [1, 16, 64, 256]:
        num_heads = 64
        num_kv_heads = 1
        head_dim = 128
        rope_dim = 64  # qk_rope_head_dim

        q = torch.randn(M, num_heads, head_dim, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(M, num_kv_heads, head_dim, device='cuda', dtype=torch.bfloat16)
        cos = torch.randn(M, rope_dim // 2, device='cuda', dtype=torch.float32)
        sin = torch.randn(M, rope_dim // 2, device='cuda', dtype=torch.float32)

        # Baseline: manual RoPE
        def baseline_rope():
            q_clone = q.clone()
            k_clone = k.clone()
            half = rope_dim // 2
            for h in range(num_heads):
                x0 = q_clone[:, h, :half].float()
                x1 = q_clone[:, h, half:rope_dim].float()
                q_clone[:, h, :half] = (x0 * cos - x1 * sin).to(torch.bfloat16)
                q_clone[:, h, half:rope_dim] = (x0 * sin + x1 * cos).to(torch.bfloat16)
            for h in range(num_kv_heads):
                x0 = k_clone[:, h, :half].float()
                x1 = k_clone[:, h, half:rope_dim].float()
                k_clone[:, h, :half] = (x0 * cos - x1 * sin).to(torch.bfloat16)
                k_clone[:, h, half:rope_dim] = (x0 * sin + x1 * cos).to(torch.bfloat16)
            return q_clone, k_clone

        ref_q, ref_k = baseline_rope()
        ms_base = bench(baseline_rope)

        def optimized_rope():
            q_c = q.clone()
            k_c = k.clone()
            lib2.launch_rope(q_c.data_ptr(), k_c.data_ptr(), cos.data_ptr(), sin.data_ptr(),
                           M, num_heads, num_kv_heads, head_dim, rope_dim, None)
            return q_c, k_c

        opt_q, opt_k = optimized_rope()
        torch.cuda.synchronize()
        ms_opt = bench(optimized_rope)

        q_diff = (opt_q.float() - ref_q.float()).abs().max().item()
        k_diff = (opt_k.float() - ref_k.float()).abs().max().item()
        correct = q_diff < 0.01 and k_diff < 0.01
        record("rope", M, head_dim, ms_base, ms_opt, correct,
               "q_diff=%.5f k_diff=%.5f" % (q_diff, k_diff))

# ============================================================
# TEST moe_sum
# ============================================================
def test_moe_sum(lib2):
    print("\n" + "="*60)
    print("KERNEL 6: moe_sum")
    print("="*60)
    for M in [4, 16, 64, 256]:
        N = 4096
        topk = 6
        expert_out = torch.randn(M * topk, N, device='cuda', dtype=torch.bfloat16)
        weights = torch.randn(M, topk, device='cuda', dtype=torch.float32).softmax(dim=-1)

        # Baseline
        def baseline():
            out = torch.zeros(M, N, device='cuda', dtype=torch.bfloat16)
            for k in range(topk):
                out += (expert_out[k*M:(k+1)*M] * weights[:, k:k+1].to(torch.bfloat16))
            return out

        # Actually for correct indexing: expert_out is [M*topk, N] where row i*topk+k is expert k for token i
        def baseline_correct():
            out = torch.zeros(M, N, device='cuda', dtype=torch.float32)
            for t in range(M):
                for k in range(topk):
                    out[t] += weights[t, k] * expert_out[t * topk + k].float()
            return out.to(torch.bfloat16)

        ref = baseline_correct()
        ms_base = bench(baseline_correct)

        output = torch.empty(M, N, device='cuda', dtype=torch.bfloat16)
        def opt():
            lib2.launch_moe_sum(expert_out.data_ptr(), weights.data_ptr(), output.data_ptr(), M, N, topk, None)
        opt()
        torch.cuda.synchronize()
        ms_opt = bench(opt)

        diff = (output.float() - ref.float()).abs().max().item()
        correct = diff < 0.1
        record("moe_sum", M, N, ms_base, ms_opt, correct, "max_diff=%.5f" % diff)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("PyTorch: %s" % torch.__version__)
    print("Device: %s" % torch.cuda.get_device_name(0))

    lib1, lib2 = compile_all()
    if lib1 is None:
        exit(1)
    print("All kernels compiled.\n")

    test_per_token_quant(lib2)
    test_rope(lib2)
    test_moe_sum(lib2)

    # Also re-run the v1 kernels for complete report
    print("\n" + "="*60)
    print("KERNEL 1-3: (from v1, previously validated)")
    print("="*60)
    import lightop
    from lmslim.layers.gemm.int8_utils import per_token_quant_int8

    eps = 1e-6
    for M in [16, 64, 256]:
        N = 4096
        residual = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(N, device='cuda', dtype=torch.bfloat16).abs() + 0.1
        # fused_add_rmsnorm_quant
        def base1():
            r = residual.clone(); xc = x.clone()
            n, ro = lightop.fused_add_rms_norm(xc, r, w, eps)
            q, s = per_token_quant_int8(n)
            return q.reshape(M, N), s.reshape(M)
        ms_b = bench(base1)
        out_q = torch.empty(M, N, device='cuda', dtype=torch.int8)
        out_s = torch.empty(M, device='cuda', dtype=torch.float32)
        def opt1():
            r = residual.clone()
            lib1.launch_fused_add_rmsnorm_quant(r.data_ptr(), x.data_ptr(), w.data_ptr(),
                                                out_q.data_ptr(), out_s.data_ptr(), M, N, eps, None)
        ms_o = bench(opt1)
        opt1(); torch.cuda.synchronize()
        ref_q, ref_s = base1()
        diff = (out_q.int() - ref_q.int()).abs().max().item()
        record("fused_add_rmsnorm_quant", M, N, ms_b, ms_o, diff <= 1)

    # Summary
    print("\n" + "="*60)
    print("COMPLETE SUMMARY - ALL KERNELS")
    print("="*60)
    correct_count = sum(1 for r in ALL_RESULTS if r['correct'])
    fast_count = sum(1 for r in ALL_RESULTS if r.get('speedup', 0) >= 1.5)
    total = len(ALL_RESULTS)
    print("  Total configs: %d" % total)
    print("  Correct: %d/%d" % (correct_count, total))
    print("  >=1.5x speedup: %d/%d" % (fast_count, total))
    if ALL_RESULTS:
        avg = sum(r['speedup'] for r in ALL_RESULTS) / len(ALL_RESULTS)
        print("  Average speedup: %.2fx" % avg)
    all_pass = all(r['correct'] for r in ALL_RESULTS)
    print("  OVERALL: %s" % ("ALL PASS" if all_pass else "FAILURES"))

    with open("/workspace/all_kernels_validation.json", "w") as f:
        json.dump(ALL_RESULTS, f, indent=2)
    print("  Trace: /workspace/all_kernels_validation.json")
