"""
End-to-end MoE layer benchmark:
Compare baseline (separate PyTorch ops) vs optimized (HIP fused kernels).
Simulates the real DeepSeek V4 MoE forward pass.
"""
import torch
import ctypes
import subprocess
import time
import json
import os

def compile_and_load():
    src = "/workspace/hip_kernels/fused_ops.hip"
    lib_path = "/workspace/hip_kernels/libfused_ops.so"
    if not os.path.exists(lib_path):
        subprocess.run(["hipcc", "-O3", "--offload-arch=gfx936", "-shared", "-fPIC", "-o", lib_path, src],
                      check=True, capture_output=True)
    lib = ctypes.CDLL(lib_path)
    lib.launch_fused_add_rmsnorm_quant.argtypes = [
        ctypes.c_void_p]*5 + [ctypes.c_int]*2 + [ctypes.c_float, ctypes.c_void_p]
    lib.launch_fused_silu_mul_quant.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_void_p]
    lib.launch_fused_rmsnorm_quant.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_float, ctypes.c_void_p]
    return lib


def bench(fn, warmup=20, repeat=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / repeat * 1000


def e2e_moe_layer_benchmark(lib):
    """Simulate one complete MoE layer pass for DeepSeek V4."""
    print("=" * 70)
    print("END-TO-END MoE LAYER BENCHMARK")
    print("=" * 70)

    K = 4096       # hidden_size
    N = 2048       # moe_intermediate_size
    eps = 1e-6
    topk = 6
    num_experts = 256

    results = []

    for batch_tokens in [4, 8, 16, 32, 64, 128]:
        M = batch_tokens
        M_expanded = M * topk  # after expert dispatch

        # Setup - simulates DeepSeek V4 W8A8 MoE layer
        residual = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
        attn_output = torch.randn(M, K, device='cuda', dtype=torch.bfloat16)
        rmsnorm_w = torch.randn(K, device='cuda', dtype=torch.bfloat16).abs() + 0.5
        gate_w = torch.randn(num_experts, K, device='cuda', dtype=torch.bfloat16)
        expert_w_gate = torch.randint(-128, 127, (N, K), device='cuda', dtype=torch.int8)
        expert_w_up = torch.randint(-128, 127, (N, K), device='cuda', dtype=torch.int8)
        expert_w_down = torch.randint(-128, 127, (K, N), device='cuda', dtype=torch.int8)

        # ===== BASELINE: Separate ops =====
        def baseline():
            # Step 1: Residual add
            r = residual + attn_output
            # Step 2: RMSNorm
            rf = r.float()
            var = rf.pow(2).mean(-1, keepdim=True)
            normed = (rf * torch.rsqrt(var + eps) * rmsnorm_w.float()).to(torch.bfloat16)
            # Step 3: Gate projection + TopK
            scores = normed.float() @ gate_w.float().T
            topk_vals, topk_ids = torch.topk(scores, k=topk, dim=-1)
            # Step 4: Per-token INT8 quant
            abs_max = normed.abs().amax(dim=-1, keepdim=True).float()
            scale = abs_max / 127.0
            quant = (normed.float() / scale).round().clamp(-128, 127).to(torch.int8)
            # Step 5: Expert gate GEMM
            expanded = quant.repeat(topk, 1)
            gate_out = torch._int_mm(expanded, expert_w_gate.T)
            # Step 6: Expert up GEMM
            up_out = torch._int_mm(expanded, expert_w_up.T)
            # Step 7: SiLU * Mul
            act = torch.nn.functional.silu(gate_out.float()) * up_out.float()
            # Step 8: Activation INT8 quant
            act_bf = act.to(torch.bfloat16)
            abs_max2 = act_bf.abs().amax(dim=-1, keepdim=True).float()
            scale2 = abs_max2 / 127.0
            act_q = (act / scale2).round().clamp(-128, 127).to(torch.int8)
            # Step 9: Expert down GEMM
            down_out = torch._int_mm(act_q, expert_w_down.T)
            return down_out

        # ===== OPTIMIZED: HIP fused kernels =====
        # Pre-allocate output buffers
        out_int8_1 = torch.empty(M, K, device='cuda', dtype=torch.int8)
        out_scale_1 = torch.empty(M, device='cuda', dtype=torch.float32)
        out_int8_2 = torch.empty(M_expanded, N, device='cuda', dtype=torch.int8)
        out_scale_2 = torch.empty(M_expanded, device='cuda', dtype=torch.float32)

        def optimized():
            # Step 1+2+4 FUSED: residual += attn; rmsnorm; quant
            residual_copy = residual.clone()
            lib.launch_fused_add_rmsnorm_quant(
                residual_copy.data_ptr(), attn_output.data_ptr(), rmsnorm_w.data_ptr(),
                out_int8_1.data_ptr(), out_scale_1.data_ptr(),
                M, K, eps, None)
            # Step 3: Gate + TopK (still separate - GEMM can't be fused)
            # Use the normed residual for gate projection
            normed_approx = residual_copy  # residual was updated in-place
            scores = normed_approx.float() @ gate_w.float().T
            topk_vals, topk_ids = torch.topk(scores, k=topk, dim=-1)
            # Step 5: Expert gate GEMM
            expanded = out_int8_1.repeat(topk, 1)
            gate_out = torch._int_mm(expanded, expert_w_gate.T)
            # Step 6: Expert up GEMM
            up_out = torch._int_mm(expanded, expert_w_up.T)
            # Step 7+8 FUSED: SiLU * Mul + quant
            gate_bf = gate_out.to(torch.bfloat16)
            up_bf = up_out.to(torch.bfloat16)
            lib.launch_fused_silu_mul_quant(
                gate_bf.data_ptr(), up_bf.data_ptr(),
                out_int8_2.data_ptr(), out_scale_2.data_ptr(),
                M_expanded, N, None)
            # Step 9: Expert down GEMM
            down_out = torch._int_mm(out_int8_2, expert_w_down.T)
            return down_out

        ms_base = bench(baseline, warmup=10, repeat=100)
        ms_opt = bench(optimized, warmup=10, repeat=100)
        speedup = ms_base / ms_opt

        print(f"\n  Batch={batch_tokens:>4} tokens (expanded={M_expanded})")
        print(f"    Baseline:  {ms_base:.4f} ms")
        print(f"    Optimized: {ms_opt:.4f} ms")
        print(f"    Speedup:   {speedup:.2f}x")

        results.append({
            "batch_tokens": batch_tokens,
            "baseline_ms": round(ms_base, 4),
            "optimized_ms": round(ms_opt, 4),
            "speedup": round(speedup, 2)
        })

    # Summary
    print("\n" + "=" * 70)
    print("E2E SUMMARY")
    print("=" * 70)
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    print(f"Average end-to-end MoE layer speedup: {avg_speedup:.2f}x")
    for r in results:
        print(f"  batch={r['batch_tokens']:<4}: {r['baseline_ms']:.4f} -> {r['optimized_ms']:.4f} ms ({r['speedup']:.2f}x)")

    with open("/workspace/e2e_moe_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nTrace: /workspace/e2e_moe_results.json")


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}\n")
    lib = compile_and_load()
    e2e_moe_layer_benchmark(lib)
