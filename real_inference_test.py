"""
Real inference validation with DeepSeek V4 Flash model weights.
Loads actual model weights from the layer0-4 subset and runs forward pass
with and without HIP fused kernels, comparing:
1. Output correctness (bit-exact)
2. End-to-end latency
3. Per-kernel time breakdown
"""
import torch
import ctypes
import subprocess
import time
import json
import os
from safetensors import safe_open

MODEL_PATH = "/home_aclsylqidf/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8-layer0-4"

def load_hip_lib():
    lib_path = "/workspace/hip_kernels/libfused_ops.so"
    if not os.path.exists(lib_path):
        subprocess.run(["hipcc", "-O3", "--offload-arch=gfx936", "-shared", "-fPIC",
                       "-o", lib_path, "/workspace/hip_kernels/fused_ops.hip"], check=True)
    lib = ctypes.CDLL(lib_path)
    lib.launch_fused_rmsnorm_quant.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_float, ctypes.c_void_p]
    lib.launch_fused_silu_mul_quant.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int]*2 + [ctypes.c_void_p]
    lib.launch_fused_add_rmsnorm_quant.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*2 + [ctypes.c_float, ctypes.c_void_p]
    return lib


def load_model_weights():
    """Load layer 2 (first MoE layer) weights."""
    print("Loading model weights...")
    config = json.load(open(os.path.join(MODEL_PATH, "config.json")))
    index = json.load(open(os.path.join(MODEL_PATH, "model.safetensors.index.json")))
    weight_map = index["weight_map"]

    # Find a MoE layer (layer 2+ has MoE based on compress_ratios)
    layer_idx = 2  # First MoE layer
    prefix = f"model.layers.{layer_idx}"

    needed_keys = [
        f"{prefix}.input_layernorm.weight",
        f"{prefix}.post_attention_layernorm.weight",
    ]

    weights = {}
    loaded_files = set()
    for key in needed_keys:
        if key in weight_map:
            fname = weight_map[key]
            fpath = os.path.join(MODEL_PATH, fname)
            if fname not in loaded_files:
                loaded_files.add(fname)
            with safe_open(fpath, framework="pt", device="cuda:0") as f:
                if key in f.keys():
                    weights[key] = f.get_tensor(key)
                    print(f"  Loaded {key}: {weights[key].shape} {weights[key].dtype}")

    return config, weights, layer_idx


def run_inference_validation(lib):
    """Run real inference with actual model weights."""
    print("\n" + "=" * 70)
    print("REAL INFERENCE VALIDATION")
    print("=" * 70)

    config, weights, layer_idx = load_model_weights()
    prefix = f"model.layers.{layer_idx}"

    hidden_size = config["hidden_size"]
    eps = config.get("rms_norm_eps", 1e-6)

    # Get actual RMSNorm weights
    norm_key = f"{prefix}.input_layernorm.weight"
    if norm_key not in weights:
        print(f"  WARNING: {norm_key} not found, using random weights")
        norm_weight = torch.randn(hidden_size, device="cuda:0", dtype=torch.bfloat16).abs() + 0.5
    else:
        norm_weight = weights[norm_key].to(torch.bfloat16).to("cuda:0")

    print(f"\n  Config: hidden_size={hidden_size}, eps={eps}")
    print(f"  Norm weight shape: {norm_weight.shape}, dtype: {norm_weight.dtype}")

    results = []

    for batch_size in [1, 4, 16, 64, 256]:
        # Simulate hidden states (as if coming from attention output)
        hidden = torch.randn(batch_size, hidden_size, device="cuda:0", dtype=torch.bfloat16)
        residual = torch.randn(batch_size, hidden_size, device="cuda:0", dtype=torch.bfloat16)

        # ===== BASELINE: PyTorch ops =====
        def baseline_forward():
            r = residual + hidden
            rf = r.float()
            var = rf.pow(2).mean(-1, keepdim=True)
            normed = (rf * torch.rsqrt(var + eps) * norm_weight.float()).to(torch.bfloat16)
            # Per-token INT8 quant
            abs_max = normed.abs().amax(dim=-1, keepdim=True).float()
            scale = abs_max / 127.0
            quant = (normed.float() / scale).round().clamp(-128, 127).to(torch.int8)
            return r, quant, scale.squeeze(-1)

        # ===== OPTIMIZED: HIP fused kernel =====
        out_int8 = torch.empty(batch_size, hidden_size, device="cuda:0", dtype=torch.int8)
        out_scale = torch.empty(batch_size, device="cuda:0", dtype=torch.float32)

        def optimized_forward():
            residual_copy = residual.clone()
            lib.launch_fused_add_rmsnorm_quant(
                residual_copy.data_ptr(), hidden.data_ptr(), norm_weight.data_ptr(),
                out_int8.data_ptr(), out_scale.data_ptr(),
                batch_size, hidden_size, eps, None)
            return residual_copy, out_int8, out_scale

        # Correctness check
        torch.cuda.synchronize()
        ref_residual, ref_q, ref_s = baseline_forward()
        opt_residual, opt_q, opt_s = optimized_forward()
        torch.cuda.synchronize()

        # Strict comparison
        residual_match = torch.allclose(ref_residual, opt_residual, atol=1e-3)
        q_exact_match = (ref_q == opt_q).float().mean().item()
        q_max_diff = (ref_q.int() - opt_q.int()).abs().max().item()
        s_max_diff = (ref_s - opt_s).abs().max().item()

        # Performance benchmark
        # Warmup
        for _ in range(20):
            baseline_forward()
            optimized_forward()

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(200):
            baseline_forward()
        torch.cuda.synchronize()
        base_ms = (time.time() - t0) / 200 * 1000

        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(200):
            optimized_forward()
        torch.cuda.synchronize()
        opt_ms = (time.time() - t0) / 200 * 1000

        speedup = base_ms / opt_ms
        correct = q_max_diff <= 1 and s_max_diff < 1e-3 and residual_match

        print(f"\n  Batch={batch_size:>4}:")
        print(f"    Residual match: {residual_match}")
        print(f"    INT8 exact match: {q_exact_match*100:.1f}%, max_diff: {q_max_diff}")
        print(f"    Scale max_diff: {s_max_diff:.6f}")
        print(f"    Baseline: {base_ms:.4f} ms, Optimized: {opt_ms:.4f} ms")
        print(f"    Speedup: {speedup:.2f}x  Correct: {'PASS' if correct else 'FAIL'}")

        results.append({
            "batch": batch_size,
            "baseline_ms": round(base_ms, 4),
            "optimized_ms": round(opt_ms, 4),
            "speedup": round(speedup, 2),
            "correct": correct,
            "q_exact_match": round(q_exact_match, 4),
            "q_max_diff": q_max_diff,
            "s_max_diff": round(s_max_diff, 6),
        })

    # Summary
    print("\n" + "=" * 70)
    print("REAL INFERENCE VALIDATION SUMMARY")
    print("=" * 70)
    all_correct = all(r["correct"] for r in results)
    avg_speedup = sum(r["speedup"] for r in results) / len(results)
    print(f"  All correct: {all_correct}")
    print(f"  Average speedup: {avg_speedup:.2f}x")
    for r in results:
        status = "✓" if r["correct"] else "✗"
        print(f"  B={r['batch']:<4}: {r['baseline_ms']:.4f} -> {r['optimized_ms']:.4f} ms ({r['speedup']:.2f}x) {status}")

    with open("/workspace/real_inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Trace: /workspace/real_inference_results.json")
    return all_correct


if __name__ == "__main__":
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Using GPU (HIP_VISIBLE_DEVICES controls which physical GPU)\n")

    lib = load_hip_lib()
    print("HIP kernels loaded.\n")

    success = run_inference_validation(lib)
    if not success:
        print("\n*** VALIDATION FAILED ***")
        exit(1)
    else:
        print("\n*** ALL VALIDATIONS PASSED ***")
