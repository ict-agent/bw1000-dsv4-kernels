"""
Complete kernel optimization exploration across DSLs:
- Triton: iterative tuning with different configs
- HIP C++: handwritten optimized
- Compare and record best for each

Outputs detailed JSON trace with all iterations.
"""
import torch
import triton
import triton.language as tl
import ctypes
import subprocess
import time
import json
import os

TRACE = {"metadata": {"device": "", "timestamp": "", "pytorch": ""},
         "kernels": {}}

def init_trace():
    TRACE["metadata"]["device"] = torch.cuda.get_device_name(0)
    TRACE["metadata"]["pytorch"] = torch.__version__
    TRACE["metadata"]["triton"] = triton.__version__
    TRACE["metadata"]["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")

def bench(fn, warmup=30, repeat=500):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / repeat * 1000

# ============================================================
# TRITON per_token_quant_int8 - Explore block sizes
# ============================================================

@triton.jit
def triton_per_token_quant_kernel(
    X, OUT_Q, OUT_S,
    stride_x, stride_q,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    abs_max = tl.max(tl.abs(x), axis=0)
    scale = abs_max / 127.0
    scale = tl.where(scale > 0, scale, 1.0)
    inv_scale = 1.0 / scale
    quantized = tl.where(x >= 0, (x * inv_scale + 0.5).to(tl.int32), (x * inv_scale - 0.5).to(tl.int32))
    quantized = tl.maximum(tl.minimum(quantized, 127), -128)
    tl.store(OUT_Q + row * stride_q + cols, quantized.to(tl.int8), mask=mask)
    tl.store(OUT_S + row, scale.to(tl.float32))

@triton.jit
def triton_fused_silu_mul_quant_kernel(
    GATE, UP, OUT_Q, OUT_S,
    stride_g, stride_u, stride_q,
    N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    g = tl.load(GATE + row * stride_g + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(UP + row * stride_u + cols, mask=mask, other=0.0).to(tl.float32)
    silu_g = g * tl.sigmoid(g)
    val = silu_g * u
    abs_max = tl.max(tl.abs(val), axis=0)
    scale = abs_max / 127.0
    scale = tl.where(scale > 0, scale, 1.0)
    inv_scale = 1.0 / scale
    quantized = tl.where(val >= 0, (val * inv_scale + 0.5).to(tl.int32), (val * inv_scale - 0.5).to(tl.int32))
    quantized = tl.maximum(tl.minimum(quantized, 127), -128)
    tl.store(OUT_Q + row * stride_q + cols, quantized.to(tl.int8), mask=mask)
    tl.store(OUT_S + row, scale.to(tl.float32))

@triton.jit
def triton_fused_add_rmsnorm_quant_kernel(
    RESIDUAL, X, WEIGHT, OUT_Q, OUT_S,
    stride_r, stride_x, stride_q,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    r = tl.load(RESIDUAL + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(WEIGHT + cols, mask=mask, other=0.0).to(tl.float32)
    added = r + x
    # Store updated residual
    tl.store(RESIDUAL + row * stride_r + cols, added.to(tl.bfloat16), mask=mask)
    # RMSNorm
    var = tl.sum(added * added, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + eps)
    normed = added * rrms * w
    # Quantize
    abs_max = tl.max(tl.abs(normed), axis=0)
    scale = abs_max / 127.0
    scale = tl.where(scale > 0, scale, 1.0)
    inv_scale = 1.0 / scale
    quantized = tl.where(normed >= 0, (normed * inv_scale + 0.5).to(tl.int32), (normed * inv_scale - 0.5).to(tl.int32))
    quantized = tl.maximum(tl.minimum(quantized, 127), -128)
    tl.store(OUT_Q + row * stride_q + cols, quantized.to(tl.int8), mask=mask)
    tl.store(OUT_S + row, scale.to(tl.float32))


def explore_triton_per_token_quant():
    """Explore Triton kernel configs for per_token_quant_int8."""
    print("=" * 70)
    print("TRITON per_token_quant_int8 - Config Exploration")
    print("=" * 70)
    from lmslim.layers.gemm.int8_utils import per_token_quant_int8 as baseline_fn

    kernel_trace = {"name": "per_token_quant_int8", "dsl": "triton", "iterations": []}

    for M in [16, 64, 256]:
        N = 4096
        x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        ref_q, ref_s = baseline_fn(x)
        ref_q = ref_q.reshape(M, N); ref_s = ref_s.reshape(M)
        ms_base = bench(lambda: baseline_fn(x))

        best_ms = float('inf')
        best_config = ""
        # Try different num_warps
        for num_warps in [4, 8, 16]:
            BLOCK_N = triton.next_power_of_2(N)
            out_q = torch.empty(M, N, device='cuda', dtype=torch.int8)
            out_s = torch.empty(M, device='cuda', dtype=torch.float32)
            try:
                def run():
                    triton_per_token_quant_kernel[(M,)](
                        x, out_q, out_s, x.stride(0), out_q.stride(0), N,
                        BLOCK_N=BLOCK_N, num_warps=num_warps)
                run()
                torch.cuda.synchronize()
                ms = bench(run)
                diff = (out_q.int() - ref_q.int()).abs().max().item()
                correct = diff <= 1
                if ms < best_ms and correct:
                    best_ms = ms
                    best_config = "warps=%d,BLOCK=%d" % (num_warps, BLOCK_N)
                kernel_trace["iterations"].append({
                    "M": M, "N": N, "num_warps": num_warps, "BLOCK_N": BLOCK_N,
                    "ms": round(ms, 4), "correct": correct, "max_diff": diff
                })
            except Exception as e:
                kernel_trace["iterations"].append({
                    "M": M, "N": N, "num_warps": num_warps, "error": str(e)[:50]
                })

        speedup = ms_base / best_ms if best_ms < float('inf') else 0
        print("  M=%3d: baseline=%.4fms best_triton=%.4fms (%.1fx) [%s]" %
              (M, ms_base, best_ms, speedup, best_config))

    TRACE["kernels"]["per_token_quant_int8_triton"] = kernel_trace


def explore_triton_fused_silu_mul_quant():
    """Explore Triton fused_silu_mul_quant configs."""
    print("\n" + "=" * 70)
    print("TRITON fused_silu_mul_quant - Config Exploration")
    print("=" * 70)

    kernel_trace = {"name": "fused_silu_mul_quant", "dsl": "triton", "iterations": []}

    for M in [16, 64, 256]:
        N = 2048
        gate = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        up = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)

        def baseline():
            h = torch.nn.functional.silu(gate.float()) * up.float()
            hb = h.to(torch.bfloat16)
            am = hb.abs().amax(dim=-1, keepdim=True).float()
            s = am / 127.0
            q = (hb.float() / s).round().clamp(-128, 127).to(torch.int8)
            return q, s.squeeze(-1)

        ref_q, ref_s = baseline()
        ms_base = bench(baseline)

        best_ms = float('inf')
        for num_warps in [4, 8, 16]:
            BLOCK_N = triton.next_power_of_2(N)
            out_q = torch.empty(M, N, device='cuda', dtype=torch.int8)
            out_s = torch.empty(M, device='cuda', dtype=torch.float32)
            try:
                def run():
                    triton_fused_silu_mul_quant_kernel[(M,)](
                        gate, up, out_q, out_s,
                        gate.stride(0), up.stride(0), out_q.stride(0), N,
                        BLOCK_N=BLOCK_N, num_warps=num_warps)
                run(); torch.cuda.synchronize()
                ms = bench(run)
                diff = (out_q.int() - ref_q.int()).abs().max().item()
                correct = diff <= 1
                if ms < best_ms and correct:
                    best_ms = ms
                kernel_trace["iterations"].append({
                    "M": M, "N": N, "num_warps": num_warps,
                    "ms": round(ms, 4), "correct": correct
                })
            except:
                pass

        speedup = ms_base / best_ms if best_ms < float('inf') else 0
        print("  M=%3d: baseline=%.4fms best_triton=%.4fms (%.1fx)" %
              (M, ms_base, best_ms, speedup))

    TRACE["kernels"]["fused_silu_mul_quant_triton"] = kernel_trace


def explore_triton_fused_add_rmsnorm_quant():
    """Explore Triton fused_add_rmsnorm_quant."""
    print("\n" + "=" * 70)
    print("TRITON fused_add_rmsnorm_quant - Config Exploration")
    print("=" * 70)

    import lightop
    from lmslim.layers.gemm.int8_utils import per_token_quant_int8

    kernel_trace = {"name": "fused_add_rmsnorm_quant", "dsl": "triton", "iterations": []}
    eps = 1e-6

    for M in [16, 64, 256]:
        N = 4096
        residual = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(N, device='cuda', dtype=torch.bfloat16).abs() + 0.1

        def baseline():
            r = residual.clone(); xc = x.clone()
            n, ro = lightop.fused_add_rms_norm(xc, r, w, eps)
            q, s = per_token_quant_int8(n)
            return ro, q.reshape(M, N), s.reshape(M)

        _, ref_q, ref_s = baseline()
        ms_base = bench(baseline)

        best_ms = float('inf')
        for num_warps in [4, 8]:
            BLOCK_N = triton.next_power_of_2(N)
            out_q = torch.empty(M, N, device='cuda', dtype=torch.int8)
            out_s = torch.empty(M, device='cuda', dtype=torch.float32)
            try:
                def run():
                    r = residual.clone()
                    triton_fused_add_rmsnorm_quant_kernel[(M,)](
                        r, x, w, out_q, out_s,
                        r.stride(0), x.stride(0), out_q.stride(0), N, eps,
                        BLOCK_N=BLOCK_N, num_warps=num_warps)
                run(); torch.cuda.synchronize()
                ms = bench(run)
                diff = (out_q.int() - ref_q.int()).abs().max().item()
                correct = diff <= 1
                if ms < best_ms and correct:
                    best_ms = ms
                kernel_trace["iterations"].append({
                    "M": M, "N": N, "num_warps": num_warps,
                    "ms": round(ms, 4), "correct": correct, "max_diff": diff
                })
            except:
                pass

        speedup = ms_base / best_ms if best_ms < float('inf') else 0
        print("  M=%3d: baseline=%.4fms best_triton=%.4fms (%.1fx)" %
              (M, ms_base, best_ms, speedup))

    TRACE["kernels"]["fused_add_rmsnorm_quant_triton"] = kernel_trace


def explore_hip_kernels():
    """Benchmark HIP C++ kernels (already compiled)."""
    print("\n" + "=" * 70)
    print("HIP C++ Kernels - Performance")
    print("=" * 70)

    lib_path = "/workspace/hip_kernels/libfused_ops.so"
    lib2_path = "/workspace/hip_kernels/libfused_ops_v2.so"
    lib = ctypes.CDLL(lib_path)
    lib2 = ctypes.CDLL(lib2_path)
    lib.launch_fused_add_rmsnorm_quant.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*2 + [ctypes.c_float, ctypes.c_void_p]
    lib.launch_fused_silu_mul_quant.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_void_p]
    lib2.launch_per_token_quant_int8.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*2 + [ctypes.c_void_p]

    import lightop
    from lmslim.layers.gemm.int8_utils import per_token_quant_int8

    kernel_trace = {"name": "all_hip", "dsl": "hip_cpp", "results": []}
    eps = 1e-6

    for M in [16, 64, 256]:
        N = 4096
        # per_token_quant
        x = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        ref_q, _ = per_token_quant_int8(x)
        ref_q = ref_q.reshape(M, N)
        ms_base_ptq = bench(lambda: per_token_quant_int8(x))
        out_q = torch.empty(M, N, device='cuda', dtype=torch.int8)
        out_s = torch.empty(M, device='cuda', dtype=torch.float32)
        def hip_ptq(): lib2.launch_per_token_quant_int8(x.data_ptr(), out_q.data_ptr(), out_s.data_ptr(), M, N, None)
        hip_ptq(); torch.cuda.synchronize()
        ms_hip_ptq = bench(hip_ptq)
        diff_ptq = (out_q.int() - ref_q.int()).abs().max().item()

        # fused_add_rmsnorm_quant
        residual = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        xx = torch.randn(M, N, device='cuda', dtype=torch.bfloat16)
        w = torch.randn(N, device='cuda', dtype=torch.bfloat16).abs() + 0.1
        def base_arnq():
            r = residual.clone(); xc = xx.clone()
            n, ro = lightop.fused_add_rms_norm(xc, r, w, eps)
            q, s = per_token_quant_int8(n)
            return q.reshape(M, N)
        ref_arnq = base_arnq()
        ms_base_arnq = bench(base_arnq)
        out_q2 = torch.empty(M, N, device='cuda', dtype=torch.int8)
        out_s2 = torch.empty(M, device='cuda', dtype=torch.float32)
        def hip_arnq():
            r = residual.clone()
            lib.launch_fused_add_rmsnorm_quant(r.data_ptr(), xx.data_ptr(), w.data_ptr(), out_q2.data_ptr(), out_s2.data_ptr(), M, N, eps, None)
        hip_arnq(); torch.cuda.synchronize()
        ms_hip_arnq = bench(hip_arnq)
        diff_arnq = (out_q2.int() - ref_arnq.int()).abs().max().item()

        print("  M=%3d: ptq base=%.4f hip=%.4f (%.1fx d=%d) | arnq base=%.4f hip=%.4f (%.1fx d=%d)" %
              (M, ms_base_ptq, ms_hip_ptq, ms_base_ptq/ms_hip_ptq, diff_ptq,
               ms_base_arnq, ms_hip_arnq, ms_base_arnq/ms_hip_arnq, diff_arnq))

        kernel_trace["results"].append({
            "M": M, "N": N,
            "per_token_quant": {"baseline_ms": round(ms_base_ptq, 4), "hip_ms": round(ms_hip_ptq, 4),
                               "speedup": round(ms_base_ptq/ms_hip_ptq, 2), "max_diff": diff_ptq},
            "fused_add_rmsnorm_quant": {"baseline_ms": round(ms_base_arnq, 4), "hip_ms": round(ms_hip_arnq, 4),
                                       "speedup": round(ms_base_arnq/ms_hip_arnq, 2), "max_diff": diff_arnq},
        })

    TRACE["kernels"]["hip_cpp"] = kernel_trace


if __name__ == "__main__":
    init_trace()
    print("Device: %s" % TRACE["metadata"]["device"])
    print("PyTorch: %s, Triton: %s\n" % (TRACE["metadata"]["pytorch"], TRACE["metadata"]["triton"]))

    explore_triton_per_token_quant()
    explore_triton_fused_silu_mul_quant()
    explore_triton_fused_add_rmsnorm_quant()
    explore_hip_kernels()

    # Save complete trace
    with open("/workspace/dsl_exploration_trace.json", "w") as f:
        json.dump(TRACE, f, indent=2)
    print("\n\nComplete trace: /workspace/dsl_exploration_trace.json")
    print("Total iterations logged: %d" % sum(
        len(v.get("iterations", v.get("results", [])))
        for v in TRACE["kernels"].values()))
