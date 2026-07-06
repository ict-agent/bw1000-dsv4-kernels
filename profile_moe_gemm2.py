"""Profile actual W8A8 GEMM kernels with correct API calls."""
import torch, time, json, inspect
import lightop
from lmslim.layers.gemm.int8_utils import per_token_quant_int8
from lmslim import quant_ops

torch.cuda.set_device(0)

def bench(fn, w=20, r=200):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(r): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/r*1000

H=4096; I=2048; E=256; TOPK=6; BATCH=1; M=BATCH*TOPK

print("="*72)
print("W8A8 GEMM KERNEL TIMING (real shapes)")
print("="*72)

# 1. triton_scaled_mm (dense, strategy=1 default)
x = torch.randn(BATCH, H, device="cuda", dtype=torch.bfloat16)
xq, xs = per_token_quant_int8(x)
xq = xq.reshape(BATCH, H).contiguous()
xs = xs.reshape(BATCH, 1).contiguous()

# Weight: [N, K] = [2*I, H] int8, scale [1, N]
w_gu = torch.randint(-128, 127, (2*I, H), device="cuda", dtype=torch.int8)
ws_gu = torch.ones(1, 2*I, device="cuda", dtype=torch.float32)

# Call triton_scaled_mm
out = quant_ops.triton_scaled_mm(xq, w_gu, scale_a=xs, scale_b=ws_gu, out_dtype=torch.bfloat16)
print(f"triton_scaled_mm (M={BATCH},K={H},N={2*I}): out={out.shape} {out.dtype}")
ms_tsm = bench(lambda: quant_ops.triton_scaled_mm(xq, w_gu, scale_a=xs, scale_b=ws_gu, out_dtype=torch.bfloat16))
print(f"  TIME: {ms_tsm:.4f} ms  ({2*BATCH*H*2*I/ms_tsm/1e9:.1f} TOPS)")

# 2. Down projection
aq = torch.randint(-128,127,(M, I), device="cuda", dtype=torch.int8)
as_ = torch.ones(M, 1, device="cuda", dtype=torch.float32)
w_d = torch.randint(-128, 127, (H, I), device="cuda", dtype=torch.int8)
ws_d = torch.ones(1, H, device="cuda", dtype=torch.float32)
out2 = quant_ops.triton_scaled_mm(aq, w_d, scale_a=as_, scale_b=ws_d, out_dtype=torch.bfloat16)
print(f"\ntriton_scaled_mm down (M={M},K={I},N={H}): out={out2.shape}")
ms_down = bench(lambda: quant_ops.triton_scaled_mm(aq, w_d, scale_a=as_, scale_b=ws_d, out_dtype=torch.bfloat16))
print(f"  TIME: {ms_down:.4f} ms  ({2*M*I*H/ms_down/1e9:.1f} TOPS)")

# 3. lightop gemm_w8a8_asm (alternative dense)
print(f"\n--- lightop gemm_w8a8_asm ---")
print("sig:", inspect.signature(lightop.gemm_w8a8_asm))
try:
    out3 = lightop.gemm_w8a8_asm(xq, w_gu, xs, ws_gu, block_size=128, out_dtype=torch.bfloat16)
    print(f"gemm_w8a8_asm: out={out3.shape}")
    ms_asm = bench(lambda: lightop.gemm_w8a8_asm(xq, w_gu, xs, ws_gu, block_size=128, out_dtype=torch.bfloat16))
    print(f"  TIME: {ms_asm:.4f} ms  ({2*BATCH*H*2*I/ms_asm/1e9:.1f} TOPS)")
except Exception as e:
    print(f"  ERROR: {str(e)[:120]}")

# 4. per_token_quant
ms_q1 = bench(lambda: per_token_quant_int8(x))
ms_q2 = bench(lambda: per_token_quant_int8(torch.randn(M, I, device="cuda", dtype=torch.bfloat16)))
print(f"\nper_token_quant (B={BATCH},N={H}): {ms_q1:.4f} ms")
print(f"per_token_quant (M={M},N={I}):   {ms_q2:.4f} ms")

# 5. silu_and_mul
g = torch.randn(M, I, device="cuda", dtype=torch.bfloat16)
u = torch.randn(M, I, device="cuda", dtype=torch.bfloat16)
ms_silu = bench(lambda: torch.nn.functional.silu(g.float())*u.float())
print(f"silu_and_mul (M={M},N={I}):      {ms_silu:.4f} ms")

# Summary
total = ms_q1 + ms_tsm + ms_down + ms_q2 + ms_silu
print(f"\n{'='*72}")
print(f"MoE DECODE STEP TOTAL: {total:.4f} ms")
print(f"{'='*72}")
print(f"  {'per_token_quant (input)':<35} {ms_q1:.4f} ms ({ms_q1/total*100:.1f}%)")
print(f"  {'triton_scaled_mm (gate_up GEMM)':<35} {ms_tsm:.4f} ms ({ms_tsm/total*100:.1f}%)")
print(f"  {'per_token_quant (act)':<35} {ms_q2:.4f} ms ({ms_q2/total*100:.1f}%)")
print(f"  {'silu_and_mul':<35} {ms_silu:.4f} ms ({ms_silu/total*100:.1f}%)")
print(f"  {'triton_scaled_mm (down GEMM)':<35} {ms_down:.4f} ms ({ms_down/total*100:.1f}%)")
print(f"\nNote: real TPOT=202ms. MoE is ~{total:.2f}ms of that. Rest = attention+other layers")

R = {"total_ms":round(total,4), "breakdown":[
    {"op":"per_token_quant_input","ms":round(ms_q1,4),"pct":round(ms_q1/total*100,1)},
    {"op":"triton_scaled_mm_gate_up","ms":round(ms_tsm,4),"pct":round(ms_tsm/total*100,1)},
    {"op":"per_token_quant_act","ms":round(ms_q2,4),"pct":round(ms_q2/total*100,1)},
    {"op":"silu_and_mul","ms":round(ms_silu,4),"pct":round(ms_silu/total*100,1)},
    {"op":"triton_scaled_mm_down","ms":round(ms_down,4),"pct":round(ms_down/total*100,1)},
]}
with open("/workspace/moe_gemm_profile.json","w") as f: json.dump(R,f,indent=2)
print(f"\nSaved: /workspace/moe_gemm_profile.json")
