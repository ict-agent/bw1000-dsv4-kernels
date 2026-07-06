"""
Strict correctness tests for HIP fused kernels.
Requirements:
- Numerical correctness within fp32 rounding tolerance
- Edge cases (zeros, large values, negative values)
- Various shapes that appear in real DeepSeek V4 inference
"""
import torch
import ctypes
import os
import subprocess
import sys

def compile_and_load():
    src = "/workspace/hip_kernels/fused_ops.hip"
    lib_path = "/workspace/hip_kernels/libfused_ops.so"
    cmd = ["hipcc", "-O3", "--offload-arch=gfx936", "-shared", "-fPIC", "-o", lib_path, src]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"COMPILE ERROR: {result.stderr}")
        sys.exit(1)
    lib = ctypes.CDLL(lib_path)
    lib.launch_fused_add_rmsnorm_quant.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
    lib.launch_fused_silu_mul_quant.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.launch_fused_rmsnorm_quant.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
    return lib


def reference_rmsnorm_quant(x, weight, eps):
    """Reference implementation: exact same math as HIP kernel."""
    xf = x.float()
    variance = xf.pow(2).mean(-1, keepdim=True)
    rrms = torch.rsqrt(variance + eps)
    normed = xf * rrms * weight.float()
    # Per-token abs max quantization
    abs_max = normed.abs().amax(dim=-1, keepdim=True)
    scale = abs_max / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    # Use round-to-nearest-even (same as __float2int_rn in HIP)
    quantized = torch.round(normed / scale).clamp(-128, 127).to(torch.int8)
    return quantized, scale.squeeze(-1)


def reference_silu_mul_quant(gate, up):
    """Reference: SiLU(gate) * up then quantize."""
    gf = gate.float()
    uf = up.float()
    hidden = torch.nn.functional.silu(gf) * uf
    abs_max = hidden.abs().amax(dim=-1, keepdim=True)
    scale = abs_max / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.round(hidden / scale).clamp(-128, 127).to(torch.int8)
    return quantized, scale.squeeze(-1)


def reference_add_rmsnorm_quant(residual, x, weight, eps):
    """Reference: residual += x; rmsnorm; quantize."""
    added = (residual.float() + x.float())
    variance = added.pow(2).mean(-1, keepdim=True)
    rrms = torch.rsqrt(variance + eps)
    normed = added * rrms * weight.float()
    abs_max = normed.abs().amax(dim=-1, keepdim=True)
    scale = abs_max / 127.0
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    quantized = torch.round(normed / scale).clamp(-128, 127).to(torch.int8)
    return quantized, scale.squeeze(-1), added.to(torch.bfloat16)


def test_kernel(lib, name, test_fn, shapes, tolerance_int8=0, tolerance_scale=1e-3):
    """Run strict correctness tests."""
    print(f"\n{'='*60}")
    print(f"STRICT TEST: {name}")
    print(f"  INT8 tolerance: ±{tolerance_int8}")
    print(f"  Scale tolerance: {tolerance_scale}")
    print(f"{'='*60}")

    all_pass = True
    for shape_desc, (M, N) in shapes:
        # Normal random data
        results = test_fn(lib, M, N, "normal")

        # Edge cases
        results_zeros = test_fn(lib, M, N, "zeros")
        results_large = test_fn(lib, M, N, "large")

        for case_name, (q_match, s_match, max_q_diff, max_s_diff) in [
            ("normal", results), ("zeros", results_zeros), ("large", results_large)
        ]:
            q_ok = max_q_diff <= tolerance_int8
            s_ok = max_s_diff <= tolerance_scale
            status = "PASS" if (q_ok and s_ok) else "FAIL"
            if not (q_ok and s_ok):
                all_pass = False
            print(f"  {shape_desc:<20} {case_name:<8} Q_diff={max_q_diff:<4} S_diff={max_s_diff:<10.6f} [{status}]")

    return all_pass


def test_rmsnorm_quant(lib, M, N, mode):
    eps = 1e-6
    if mode == "zeros":
        x = torch.zeros(M, N, device='cuda', dtype=torch.bfloat16)
    elif mode == "large":
        x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16) * 100
    else:
        x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(N, device='cuda', dtype=torch.bfloat16).abs() + 0.5

    # Reference
    ref_q, ref_s = reference_rmsnorm_quant(x, weight, eps)

    # HIP kernel
    out_int8 = torch.empty(M, N, device='cuda', dtype=torch.int8)
    out_scale = torch.empty(M, device='cuda', dtype=torch.float32)
    lib.launch_fused_rmsnorm_quant(
        x.data_ptr(), weight.data_ptr(),
        out_int8.data_ptr(), out_scale.data_ptr(),
        M, N, eps, None)
    torch.cuda.synchronize()

    q_diff = (out_int8.int() - ref_q.int()).abs().max().item()
    s_diff = (out_scale - ref_s).abs().max().item()
    q_match = (out_int8 == ref_q).float().mean().item()
    s_match = (out_scale - ref_s).abs().max().item()

    return (q_match, s_match, q_diff, s_diff)


def test_silu_mul_quant(lib, M, N, mode):
    if mode == "zeros":
        gate = torch.zeros(M, N, device='cuda', dtype=torch.bfloat16)
        up = torch.zeros(M, N, device='cuda', dtype=torch.bfloat16)
    elif mode == "large":
        gate = torch.randn(M, N, device='cuda', dtype=torch.bfloat16) * 10
        up = torch.randn(M, N, device='cuda', dtype=torch.bfloat16) * 10
    else:
        gate = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        up = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)

    ref_q, ref_s = reference_silu_mul_quant(gate, up)

    out_int8 = torch.empty(M, N, device='cuda', dtype=torch.int8)
    out_scale = torch.empty(M, device='cuda', dtype=torch.float32)
    lib.launch_fused_silu_mul_quant(
        gate.data_ptr(), up.data_ptr(),
        out_int8.data_ptr(), out_scale.data_ptr(),
        M, N, None)
    torch.cuda.synchronize()

    q_diff = (out_int8.int() - ref_q.int()).abs().max().item()
    s_diff = (out_scale - ref_s).abs().max().item()
    return ((out_int8 == ref_q).float().mean().item(), s_diff, q_diff, s_diff)


def test_add_rmsnorm_quant(lib, M, N, mode):
    eps = 1e-6
    if mode == "zeros":
        residual = torch.zeros(M, N, device='cuda', dtype=torch.bfloat16)
        x = torch.zeros(M, N, device='cuda', dtype=torch.bfloat16)
    elif mode == "large":
        residual = torch.randn(M, N, device='cuda', dtype=torch.bfloat16) * 50
        x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16) * 50
    else:
        residual = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(N, device='cuda', dtype=torch.bfloat16).abs() + 0.5

    ref_q, ref_s, ref_residual = reference_add_rmsnorm_quant(residual.clone(), x, weight, eps)

    residual_copy = residual.clone()
    out_int8 = torch.empty(M, N, device='cuda', dtype=torch.int8)
    out_scale = torch.empty(M, device='cuda', dtype=torch.float32)
    lib.launch_fused_add_rmsnorm_quant(
        residual_copy.data_ptr(), x.data_ptr(), weight.data_ptr(),
        out_int8.data_ptr(), out_scale.data_ptr(),
        M, N, eps, None)
    torch.cuda.synchronize()

    q_diff = (out_int8.int() - ref_q.int()).abs().max().item()
    s_diff = (out_scale - ref_s).abs().max().item()
    # Also check residual was updated correctly
    r_diff = (residual_copy - ref_residual).abs().max().item()

    return ((out_int8 == ref_q).float().mean().item(), s_diff, q_diff, max(s_diff, r_diff))


if __name__ == "__main__":
    print("Compiling HIP kernels...")
    lib = compile_and_load()
    print("OK\n")

    # DeepSeek V4 shapes
    shapes = [
        ("1x4096 (single)", (1, 4096)),
        ("16x4096 (small batch)", (16, 4096)),
        ("64x4096 (medium)", (64, 4096)),
        ("256x4096 (large)", (256, 4096)),
        ("1024x4096 (prefill)", (1024, 4096)),
        ("4096x4096 (max prefill)", (4096, 4096)),
    ]
    moe_shapes = [
        ("1x2048 (single)", (1, 2048)),
        ("16x2048", (16, 2048)),
        ("64x2048", (64, 2048)),
        ("256x2048", (256, 2048)),
        ("1024x2048", (1024, 2048)),
    ]

    pass1 = test_kernel(lib, "fused_rmsnorm_quant", test_rmsnorm_quant, shapes,
                        tolerance_int8=1, tolerance_scale=1e-3)
    pass2 = test_kernel(lib, "fused_silu_mul_quant", test_silu_mul_quant, moe_shapes,
                        tolerance_int8=1, tolerance_scale=1e-3)
    pass3 = test_kernel(lib, "fused_add_rmsnorm_quant", test_add_rmsnorm_quant, shapes,
                        tolerance_int8=1, tolerance_scale=1e-3)

    print(f"\n{'='*60}")
    print("OVERALL RESULT")
    print(f"{'='*60}")
    all_ok = pass1 and pass2 and pass3
    print(f"  fused_rmsnorm_quant:     {'PASS' if pass1 else 'FAIL'}")
    print(f"  fused_silu_mul_quant:    {'PASS' if pass2 else 'FAIL'}")
    print(f"  fused_add_rmsnorm_quant: {'PASS' if pass3 else 'FAIL'}")
    print(f"\n  ALL TESTS: {'PASS' if all_ok else 'FAIL'}")
