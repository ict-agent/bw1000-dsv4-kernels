"""
Complete kernel validation: compare HIP optimized kernels against
actual baseline implementations (lightop + lmslim Triton).
Fixed for Python 3.10 f-string limitations.
"""
import torch
import ctypes
import subprocess
import time
import json
import os
import sys

def load_hip_lib():
    lib_path = "/workspace/hip_kernels/libfused_ops.so"
    src_path = "/workspace/hip_kernels/fused_ops.hip"
    if not os.path.exists(lib_path) or os.path.getmtime(src_path) > os.path.getmtime(lib_path):
        subprocess.run(["hipcc", "-O3", "--offload-arch=gfx936", "-shared", "-fPIC",
                       "-o", lib_path, src_path], check=True, capture_output=True)
    lib = ctypes.CDLL(lib_path)
    lib.launch_fused_rmsnorm_quant.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_float, ctypes.c_void_p]
    lib.launch_fused_silu_mul_quant.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_void_p]
    lib.launch_fused_add_rmsnorm_quant.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*2 + [ctypes.c_float, ctypes.c_void_p]
    return lib

def bench(fn, warmup=30, repeat=300):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / repeat * 1000

ALL_RESULTS = []

def validate_add_rmsnorm_quant(hip_lib):
    """
    Baseline: lightop.fused_add_rms_norm (returns normed, residual_updated)
              + lmslim.per_token_quant_int8
    Optimized: HIP fused_add_rmsnorm_quant (one kernel does all)
    """
    print("\n" + "=" * 70)
    print("KERNEL: fused_add_rmsnorm_quant")
    print("  Baseline: lightop.fused_add_rms_norm + per_token_quant_int8")
    print("  Optimized: HIP C++ single fused kernel")
    print("=" * 70)

    import lightop
    from lmslim.layers.gemm.int8_utils import per_token_quant_int8

    eps = 1e-6
    for M in [1, 4, 16, 64, 256, 1024]:
        N = 4096
        residual_orig = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        # lightop.fused_add_rms_norm expects weight as first param difference
        # API: fused_add_rms_norm(x, residual, weight, eps) -> (normed, updated_residual)
        weight = torch.randn(N, device='cuda', dtype=torch.bfloat16).abs() + 0.1

        # === BASELINE ===
        def run_baseline():
            res = residual_orig.clone()
            x_clone = x.clone()
            # lightop: fused_add_rms_norm(x, residual, weight, eps)
            # modifies x in-place to normed, residual in-place to residual+x
            normed, res_out = lightop.fused_add_rms_norm(x_clone, res, weight, eps)
            # normed = x_clone (modified in-place), res_out = res (modified in-place)
            q, s = per_token_quant_int8(normed)
            return res_out, q.reshape(M, N), s.reshape(M)

        ref_res, ref_q, ref_s = run_baseline()
        ms_base = bench(run_baseline)

        # === OPTIMIZED (HIP) ===
        # Our kernel: residual += x, then norm(residual), then quant
        # To match lightop: we need residual_out = residual + x, normed = rmsnorm(residual_out)
        out_int8 = torch.empty(M, N, device='cuda', dtype=torch.int8)
        out_scale = torch.empty(M, device='cuda', dtype=torch.float32)

        def run_optimized():
            res = residual_orig.clone()
            hip_lib.launch_fused_add_rmsnorm_quant(
                res.data_ptr(), x.data_ptr(), weight.data_ptr(),
                out_int8.data_ptr(), out_scale.data_ptr(),
                M, N, eps, None)
            return res, out_int8, out_scale

        opt_res, opt_q, opt_s = run_optimized()
        torch.cuda.synchronize()
        ms_opt = bench(run_optimized)

        # === COMPARE ===
        q_diff = (opt_q.int() - ref_q.int()).abs()
        q_max = q_diff.max().item()
        q_rate = (q_diff == 0).float().mean().item()
        s_diff = (opt_s - ref_s).abs().max().item()
        r_diff = (opt_res.float() - ref_res.float()).abs().max().item()
        speedup = ms_base / ms_opt
        correct = q_max <= 1 and s_diff < 0.01 and r_diff < 0.01

        status = "PASS" if correct else "FAIL"
        print("  M=%4d: base=%.4fms opt=%.4fms speedup=%.1fx q_match=%.0f%% q_max=%d [%s]" %
              (M, ms_base, ms_opt, speedup, q_rate*100, q_max, status))

        ALL_RESULTS.append({
            "kernel": "fused_add_rmsnorm_quant", "M": M, "N": N,
            "baseline_ms": round(ms_base, 4), "optimized_ms": round(ms_opt, 4),
            "speedup": round(speedup, 2), "correct": correct,
            "q_match_rate": round(q_rate, 4), "q_max_diff": q_max,
            "s_max_diff": round(s_diff, 6), "r_max_diff": round(r_diff, 6)
        })


def validate_silu_mul_quant(hip_lib):
    """
    Baseline: silu + mul + per_token_quant_int8
    Optimized: HIP fused_silu_mul_quant
    """
    print("\n" + "=" * 70)
    print("KERNEL: fused_silu_mul_quant")
    print("  Baseline: torch.silu * up + per_token_quant_int8")
    print("  Optimized: HIP C++ single fused kernel")
    print("=" * 70)

    from lmslim.layers.gemm.int8_utils import per_token_quant_int8

    for M in [1, 4, 16, 64, 256, 1024]:
        N = 2048
        gate = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        up = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)

        # === BASELINE ===
        def run_baseline():
            hidden = torch.nn.functional.silu(gate.float()) * up.float()
            hidden_bf = hidden.to(torch.bfloat16)
            q, s = per_token_quant_int8(hidden_bf)
            return q.reshape(M, N), s.reshape(M)

        ref_q, ref_s = run_baseline()
        ms_base = bench(run_baseline)

        # === OPTIMIZED ===
        out_int8 = torch.empty(M, N, device='cuda', dtype=torch.int8)
        out_scale = torch.empty(M, device='cuda', dtype=torch.float32)

        def run_optimized():
            hip_lib.launch_fused_silu_mul_quant(
                gate.data_ptr(), up.data_ptr(),
                out_int8.data_ptr(), out_scale.data_ptr(),
                M, N, None)
            return out_int8, out_scale

        opt_q, opt_s = run_optimized()
        torch.cuda.synchronize()
        ms_opt = bench(run_optimized)

        # === COMPARE ===
        q_diff = (opt_q.int() - ref_q.int()).abs()
        q_max = q_diff.max().item()
        q_rate = (q_diff == 0).float().mean().item()
        s_diff = (opt_s - ref_s).abs().max().item()
        speedup = ms_base / ms_opt
        correct = q_max <= 2 and s_diff < 0.05

        status = "PASS" if correct else "FAIL"
        print("  M=%4d: base=%.4fms opt=%.4fms speedup=%.1fx q_match=%.0f%% q_max=%d [%s]" %
              (M, ms_base, ms_opt, speedup, q_rate*100, q_max, status))

        ALL_RESULTS.append({
            "kernel": "fused_silu_mul_quant", "M": M, "N": N,
            "baseline_ms": round(ms_base, 4), "optimized_ms": round(ms_opt, 4),
            "speedup": round(speedup, 2), "correct": correct,
            "q_match_rate": round(q_rate, 4), "q_max_diff": q_max,
        })


def validate_flash_mla():
    """FlashMLA baseline benchmark (already optimized HIP/ASM by Hygon)."""
    print("\n" + "=" * 70)
    print("KERNEL: flash_mla_decode (baseline reference, no optimization needed)")
    print("=" * 70)

    from flash_mla import get_mla_metadata, flash_mla_with_kvcache

    batch, heads_q, heads_k = 64, 128, 1
    head_dim, head_dim_v, block_size = 576, 512, 64

    for seqlen in [1024, 4096, 8192]:
        nb = seqlen // block_size
        tb = batch * nb
        q = torch.randn(batch, 1, heads_q, head_dim, dtype=torch.float16, device='cuda')
        kc = torch.randn(tb, block_size, heads_k, head_dim, dtype=torch.float16, device='cuda')
        bt = torch.arange(tb, dtype=torch.int32, device='cuda').reshape(batch, nb)
        cs = torch.full((batch,), seqlen, dtype=torch.int32, device='cuda')
        tsm, ns = get_mla_metadata(cs, heads_q // heads_k, heads_k)

        ms = bench(lambda: flash_mla_with_kvcache(q, kc, bt, cs, head_dim_v, tsm, ns,
                                                   softmax_scale=head_dim**-0.5))
        flops = 2 * batch * heads_q * seqlen * (head_dim + head_dim_v)
        tflops = flops / ms / 1e9
        print("  seqlen=%5d: %.4fms  %.1f TFLOPS" % (seqlen, ms, tflops))
        ALL_RESULTS.append({"kernel": "flash_mla_decode", "seqlen": seqlen,
                           "ms": round(ms, 4), "tflops": round(tflops, 1)})


if __name__ == "__main__":
    print("PyTorch: %s" % torch.__version__)
    print("Device: %s" % torch.cuda.get_device_name(0))
    print()

    hip_lib = load_hip_lib()
    print("HIP kernels loaded.\n")

    validate_add_rmsnorm_quant(hip_lib)
    validate_silu_mul_quant(hip_lib)
    validate_flash_mla()

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    opt_results = [r for r in ALL_RESULTS if 'speedup' in r]
    correct_count = sum(1 for r in opt_results if r.get('correct', True))
    fast_count = sum(1 for r in opt_results if r.get('speedup', 0) >= 1.5)

    print("  Optimized kernels tested: %d" % len(opt_results))
    print("  Correct vs baseline: %d/%d" % (correct_count, len(opt_results)))
    print("  With >=1.5x speedup: %d/%d" % (fast_count, len(opt_results)))
    if opt_results:
        avg = sum(r['speedup'] for r in opt_results) / len(opt_results)
        print("  Average speedup: %.2fx" % avg)

    all_ok = all(r.get('correct', True) for r in ALL_RESULTS)
    print("\n  OVERALL: %s" % ("ALL PASS" if all_ok else "FAILURES DETECTED"))

    with open("/workspace/validation_vs_baseline.json", "w") as f:
        json.dump(ALL_RESULTS, f, indent=2)
    print("  Trace: /workspace/validation_vs_baseline.json")
