"""Profile the actual DeepSeek V4 MoE decode step to find real bottlenecks.
Measures each kernel category in the W8A8 MoE path using real model shapes.
"""
import torch, time, json
from lmslim.layers.gemm.int8_utils import per_token_quant_int8
from lmslim import quant_ops
import lightop

torch.cuda.set_device(0)
R = {"device": torch.cuda.get_device_name(0), "sections": []}

def bench(fn, w=20, r=200):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(r): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/r*1000

# DeepSeek V4 real shapes
H = 4096       # hidden_size
I = 2048       # moe_intermediate_size
E = 256        # n_routed_experts
TOPK = 6
BATCH = 1      # decode
M = BATCH * TOPK  # expanded tokens = 6

print("="*72)
print("DEEPSEEK V4 W8A8 MoE DECODE PROFILE (batch=1, M=6 after dispatch)")
print("="*72)

# 1. per_token_quant_int8 (activation quant)
x = torch.randn(BATCH, H, device="cuda", dtype=torch.bfloat16)
ms_q = bench(lambda: per_token_quant_int8(x))
print(f"  per_token_quant_int8:      {ms_q:.4f} ms")
R["sections"].append({"op":"per_token_quant_int8","ms":round(ms_q,4)})

# 2. Dense W8A8 GEMM (gate_up_proj) via triton_scaled_mm (default strategy=1)
xq, xs = per_token_quant_int8(x)
w_gate = torch.randint(-128, 127, (H, 2*I), device="cuda", dtype=torch.int8)
ws_gate = torch.ones(1, 2*I, device="cuda", dtype=torch.float32)
try:
    ms_gemm_dense = bench(lambda: quant_ops.triton_scaled_mm(xq, w_gate, scale_a=xs, scale_b=ws_gate, out_dtype=torch.bfloat16))
    print(f"  triton_scaled_mm (dense):  {ms_gemm_dense:.4f} ms")
    R["sections"].append({"op":"triton_scaled_mm_dense","ms":round(ms_gemm_dense,4)})
except Exception as e:
    print(f"  triton_scaled_mm (dense):  ERROR {str(e)[:60]}")

# 3. MoE W8A8 Marlin GEMM (decode path) via lightop
topk_ids = torch.randint(0, E, (BATCH, TOPK), device="cuda", dtype=torch.int32)
topk_weights = torch.ones(BATCH, TOPK, device="cuda", dtype=torch.float32) / TOPK
w_expert = torch.randint(-128, 127, (E, I, H), device="cuda", dtype=torch.int8)
ws_expert = torch.ones(E, 1, I, device="cuda", dtype=torch.float32)

# Try moe_gemm_marlin_w8a8 (decode path)
try:
    import inspect
    sig = inspect.signature(lightop.moe_gemm_marlin_w8a8)
    print(f"  moe_gemm_marlin_w8a8 sig: {sig}")
except Exception as e:
    print(f"  moe_gemm_marlin_w8a8 sig: {e}")

# Try moe_gemm_w8a8
try:
    sig2 = inspect.signature(lightop.moe_gemm_w8a8)
    print(f"  moe_gemm_w8a8 sig: {sig2}")
except:
    pass

# 4. silu_and_mul
gate_out = torch.randn(M, I, device="cuda", dtype=torch.bfloat16)
up_out = torch.randn(M, I, device="cuda", dtype=torch.bfloat16)
ms_silu = bench(lambda: torch.nn.functional.silu(gate_out.float()) * up_out.float())
print(f"  silu_and_mul (PyTorch):     {ms_silu:.4f} ms")
R["sections"].append({"op":"silu_and_mul","ms":round(ms_silu,4)})

# 5. per_token_quant on activation (after silu)
act = torch.randn(M, I, device="cuda", dtype=torch.bfloat16)
ms_q2 = bench(lambda: per_token_quant_int8(act))
print(f"  per_token_quant (act):     {ms_q2:.4f} ms")
R["sections"].append({"op":"per_token_quant_act","ms":round(ms_q2,4)})

# 6. Down projection GEMM
w_down = torch.randint(-128, 127, (I, H), device="cuda", dtype=torch.int8)
ws_down = torch.ones(1, H, device="cuda", dtype=torch.float32)
aq, as_ = per_token_quant_int8(act)
try:
    ms_down = bench(lambda: quant_ops.triton_scaled_mm(aq, w_down, scale_a=as_, scale_b=ws_down, out_dtype=torch.bfloat16))
    print(f"  triton_scaled_mm (down):   {ms_down:.4f} ms")
    R["sections"].append({"op":"triton_scaled_mm_down","ms":round(ms_down,4)})
except Exception as e:
    print(f"  triton_scaled_mm (down):   ERROR {str(e)[:60]}")

# Summary
total = sum(s["ms"] for s in R["sections"])
print(f"\n  Total measured: {total:.4f} ms")
print(f"\n  Breakdown:")
for s in sorted(R["sections"], key=lambda x: -x["ms"]):
    pct = s["ms"] / total * 100 if total > 0 else 0
    print(f"    {s['op']:<30} {s['ms']:.4f} ms ({pct:.1f}%)")

with open("/workspace/moe_decode_profile.json", "w") as f:
    json.dump(R, f, indent=2)
print(f"\nSaved: /workspace/moe_decode_profile.json")
