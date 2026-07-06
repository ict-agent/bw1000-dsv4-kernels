"""
Test and benchmark HIP fused kernels directly using hipcc + ctypes.
No PyTorch extension needed - use raw hipcc shared library.
"""
import torch
import ctypes
import os
import subprocess
import time
import json

RESULTS = []

def record(name, config, base_ms, opt_ms, correct, notes=""):
    sp = base_ms / opt_ms if opt_ms > 0 else 0
    RESULTS.append({
        "kernel": name, "config": config,
        "baseline_ms": round(base_ms, 4), "optimized_ms": round(opt_ms, 4),
        "speedup": round(sp, 2), "correct": correct, "notes": notes
    })
    mark = "PASS" if correct else "FAIL"
    arrow = "✓" if sp >= 1.5 else "△"
    print(f"  [{name}] {config}")
    print(f"    Base: {base_ms:.4f}ms  Opt: {opt_ms:.4f}ms  Speedup: {sp:.2f}x {arrow}  Correct: {mark}")

def bench_fn(fn, warmup=20, repeat=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / repeat * 1000


def compile_hip_lib():
    """Compile fused_ops.hip into a shared library."""
    src = "/workspace/hip_kernels/fused_ops.hip"
    lib = "/workspace/hip_kernels/libfused_ops.so"

    cmd = [
        "hipcc", "-O3", "--offload-arch=gfx936",
        "-shared", "-fPIC",
        "-o", lib, src
    ]
    print(f"Compiling: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"COMPILE ERROR:\n{result.stderr}")
        return None
    print(f"Compiled: {lib} ({os.path.getsize(lib)} bytes)")
    return lib


def load_hip_lib(lib_path):
    """Load the compiled HIP shared library."""
    lib = ctypes.CDLL(lib_path)

    # Define function signatures
    # void launch_fused_add_rmsnorm_quant(void*, const void*, const void*, void*, void*, int, int, float, hipStream_t)
    lib.launch_fused_add_rmsnorm_quant.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p
    ]
    lib.launch_fused_add_rmsnorm_quant.restype = None

    lib.launch_fused_silu_mul_quant.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p
    ]
    lib.launch_fused_silu_mul_quant.restype = None

    lib.launch_fused_rmsnorm_quant.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p
    ]
    lib.launch_fused_rmsnorm_quant.restype = None

    return lib


def test_fused_rmsnorm_quant(lib):
    """Test and benchmark fused_rmsnorm_quant."""
    print("\n" + "=" * 70)
    print("KERNEL: fused_rmsnorm_quant (HIP C++)")
    print("=" * 70)

    eps = 1e-6
    for M in [16, 64, 256, 1024, 4096]:
        N = 4096
        x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        weight = torch.ones(N, device='cuda', dtype=torch.bfloat16)

        # Baseline: PyTorch separate ops
        def baseline():
            xf = x.float()
            variance = xf.pow(2).mean(-1, keepdim=True)
            normed = (xf * torch.rsqrt(variance + eps) * weight.float()).to(torch.bfloat16)
            abs_max = normed.abs().amax(dim=-1, keepdim=True).float()
            scale = abs_max / 127.0
            quant = (normed.float() / scale).round().clamp(-128, 127).to(torch.int8)
            return quant, scale.squeeze(-1)

        ms_base = bench_fn(baseline)

        # Optimized: HIP kernel
        out_int8 = torch.empty(M, N, device='cuda', dtype=torch.int8)
        out_scale = torch.empty(M, device='cuda', dtype=torch.float32)

        def optimized():
            lib.launch_fused_rmsnorm_quant(
                x.data_ptr(), weight.data_ptr(),
                out_int8.data_ptr(), out_scale.data_ptr(),
                M, N, eps, None  # default stream
            )

        ms_opt = bench_fn(optimized)

        # Correctness check
        ref_q, ref_s = baseline()
        optimized()
        torch.cuda.synchronize()

        # Allow +-1 tolerance for INT8 quantization
        diff = (out_int8.int() - ref_q.int()).abs()
        correct = (diff <= 1).float().mean().item() > 0.95

        record("fused_rmsnorm_quant", f"M={M},N={N}", ms_base, ms_opt, correct,
               f"match_rate={(diff==0).float().mean().item():.3f}")


def test_fused_silu_mul_quant(lib):
    """Test and benchmark fused_silu_mul_quant."""
    print("\n" + "=" * 70)
    print("KERNEL: fused_silu_mul_quant (HIP C++)")
    print("=" * 70)

    for M in [16, 64, 256, 1024, 4096]:
        N = 2048  # MoE intermediate
        gate = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        up = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)

        def baseline():
            g = gate.float()
            u = up.float()
            hidden = torch.nn.functional.silu(g) * u
            hidden_bf = hidden.to(torch.bfloat16)
            abs_max = hidden_bf.abs().amax(dim=-1, keepdim=True).float()
            scale = abs_max / 127.0
            quant = (hidden / scale).round().clamp(-128, 127).to(torch.int8)
            return quant, scale.squeeze(-1)

        ms_base = bench_fn(baseline)

        out_int8 = torch.empty(M, N, device='cuda', dtype=torch.int8)
        out_scale = torch.empty(M, device='cuda', dtype=torch.float32)

        def optimized():
            lib.launch_fused_silu_mul_quant(
                gate.data_ptr(), up.data_ptr(),
                out_int8.data_ptr(), out_scale.data_ptr(),
                M, N, None
            )

        ms_opt = bench_fn(optimized)

        ref_q, ref_s = baseline()
        optimized()
        torch.cuda.synchronize()

        diff = (out_int8.int() - ref_q.int()).abs()
        correct = (diff <= 1).float().mean().item() > 0.95

        record("fused_silu_mul_quant", f"M={M},N={N}", ms_base, ms_opt, correct,
               f"match_rate={(diff==0).float().mean().item():.3f}")


def test_fused_add_rmsnorm_quant(lib):
    """Test and benchmark fused_add_rmsnorm_quant."""
    print("\n" + "=" * 70)
    print("KERNEL: fused_add_rmsnorm_quant (HIP C++)")
    print("=" * 70)

    eps = 1e-6
    for M in [16, 64, 256, 1024, 4096]:
        N = 4096
        residual = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        weight = torch.ones(N, device='cuda', dtype=torch.bfloat16)

        def baseline():
            r = (residual + x)
            rf = r.float()
            variance = rf.pow(2).mean(-1, keepdim=True)
            normed = (rf * torch.rsqrt(variance + eps) * weight.float())
            normed_bf = normed.to(torch.bfloat16)
            abs_max = normed_bf.abs().amax(dim=-1, keepdim=True).float()
            scale = abs_max / 127.0
            quant = (normed / scale).round().clamp(-128, 127).to(torch.int8)
            return quant, scale.squeeze(-1)

        ms_base = bench_fn(baseline)

        residual_copy = residual.clone()
        out_int8 = torch.empty(M, N, device='cuda', dtype=torch.int8)
        out_scale = torch.empty(M, device='cuda', dtype=torch.float32)

        def optimized():
            residual_copy.copy_(residual)
            lib.launch_fused_add_rmsnorm_quant(
                residual_copy.data_ptr(), x.data_ptr(), weight.data_ptr(),
                out_int8.data_ptr(), out_scale.data_ptr(),
                M, N, eps, None
            )

        ms_opt = bench_fn(optimized)

        ref_q, ref_s = baseline()
        optimized()
        torch.cuda.synchronize()

        diff = (out_int8.int() - ref_q.int()).abs()
        correct = (diff <= 1).float().mean().item() > 0.95

        record("fused_add_rmsnorm_quant", f"M={M},N={N}", ms_base, ms_opt, correct,
               f"match_rate={(diff==0).float().mean().item():.3f}")


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}\n")

    # Compile
    lib_path = compile_hip_lib()
    if lib_path is None:
        exit(1)

    # Load
    lib = load_hip_lib(lib_path)
    print(f"Library loaded: {lib}\n")

    # Test all kernels
    test_fused_rmsnorm_quant(lib)
    test_fused_silu_mul_quant(lib)
    test_fused_add_rmsnorm_quant(lib)

    # Summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - HIP C++ Fused Kernels")
    print("=" * 70)
    print(f"{'Kernel':<30} {'Config':<20} {'Base':<10} {'Opt':<10} {'Speedup':<8} {'Correct'}")
    print("-" * 88)
    for r in RESULTS:
        print(f"{r['kernel']:<30} {r['config']:<20} {r['baseline_ms']:<10.4f} {r['optimized_ms']:<10.4f} {r['speedup']:<8.2f}x {'✓' if r['correct'] else '✗'}")

    wins = sum(1 for r in RESULTS if r['speedup'] >= 1.5 and r['correct'])
    total = len(RESULTS)
    print(f"\nKernels with >=1.5x AND correct: {wins}/{total}")

    with open("/workspace/hip_kernel_results.json", 'w') as f:
        json.dump(RESULTS, f, indent=2)
    print(f"Trace: /workspace/hip_kernel_results.json")
