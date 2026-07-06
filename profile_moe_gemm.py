"""Profile real MoE decode: call the ACTUAL MoE GEMM kernels."""
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
print("MoE DECODE PROFILE — calling actual vendor GEMM kernels")
print("="*72)

# Inspect triton_scaled_mm signature
print("\ntriton_scaled_mm sig:", inspect.signature(quant_ops.triton_scaled_mm))

# Try calling it correctly
x = torch.randn(BATCH, H, device="cuda", dtype=torch.bfloat16)
xq, xs = per_token_quant_int8(x)
xq = xq.reshape(BATCH, H)
xs = xs.reshape(BATCH, 1)

# weight must be int8, [out, in] or [in, out]?
w = torch.randint(-128, 127, (2*I, H), device="cuda", dtype=torch.int8)
ws = torch.ones(1, 2*I, device="cuda", dtype=torch.float32)

try:
    out = quant_ops.triton_scaled_mm(xq, w, scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16)
    print("triton_scaled_mm output:", out.shape, out.dtype)
    ms = bench(lambda: quant_ops.triton_scaled_mm(xq, w, scale_a=xs, scale_b=ws, out_dtype=torch.bfloat16))
    print(f"  triton_scaled_mm (gate_up, M={BATCH},K={H},N={2*I}): {ms:.4f} ms")
except Exception as e:
    print(f"  triton_scaled_mm: {str(e)[:100]}")

# Down projection
aq, as_ = per_token_quant_int8(torch.randn(M, I, device="cuda", dtype=torch.bfloat16))
aq = aq.reshape(M, I)
as_ = as_.reshape(M, 1)
w_down = torch.randint(-128, 127, (H, I), device="cuda", dtype=torch.int8)
ws_down = torch.ones(1, H, device="cuda", dtype=torch.float32)
try:
    out2 = quant_ops.triton_scaled_mm(aq, w_down, scale_a=as_, scale_b=ws_down, out_dtype=torch.bfloat16)
    print("triton_scaled_mm (down) output:", out2.shape)
    ms2 = bench(lambda: quant_ops.triton_scaled_mm(aq, w_down, scale_a=as_, scale_b=ws_down, out_dtype=torch.bfloat16))
    print(f"  triton_scaled_mm (down, M={M},K={I},N={H}): {ms2:.4f} ms")
except Exception as e:
    print(f"  triton_scaled_mm (down): {str(e)[:100]}")

# lightop W8A8 GEMM (non-Marlin, simpler API)
print("\n--- lightop gemm_w8a8_asm ---")
try:
    sig = inspect.signature(lightop.gemm_w8a8_asm)
    print("sig:", sig)
except: pass

# Marlin MoE GEMM
print("\n--- lightop moe_gemm_marlin_w8a8 ---")
topk_ids = torch.tensor([[0,1,2,3,4,5]], device="cuda", dtype=torch.int32)
topk_weights = torch.ones(1, TOPK, device="cuda", dtype=torch.float32) / TOPK

# Need moe_align to get sorted_token_ids etc
try:
    from aiter_decode.utils import moe_align_block_size
    sorted_ids, expert_ids, num_post = moe_align_block_size(topk_ids, 16, E, packed=False)
    print(f"moe_align OK: sorted={sorted_ids.shape} experts={expert_ids.shape} post={num_post}")
except Exception as e:
    print(f"moe_align: {e}")
    # Manual simple align
    sorted_ids = torch.arange(M, device="cuda", dtype=torch.int32)
    expert_ids = torch.tensor([0,1,2,3,4,5], device="cuda", dtype=torch.int32)
    num_post = torch.tensor([M], device="cuda", dtype=torch.int32)

# Marlin weight format is special (packed). Try direct int8 weight.
w_m = torch.randint(-128, 127, (E, I, H), device="cuda", dtype=torch.int8)
ws_m = torch.ones(E, 1, I, device="cuda", dtype=torch.float32)
out_m = torch.empty(M, H, device="cuda", dtype=torch.bfloat16)
try:
    lightop.moe_gemm_marlin_w8a8(xq, w_m.reshape(E, I*H), out_m, xs, ws_m.reshape(E, I),
                                  topk_weights, sorted_ids, expert_ids, num_post, TOPK, {})
    print("Marlin OK")
    ms_m = bench(lambda: lightop.moe_gemm_marlin_w8a8(xq, w_m.reshape(E, I*H), out_m, xs, ws_m.reshape(E, I),
                                                       topk_weights, sorted_ids, expert_ids, num_post, TOPK, {}))
    print(f"  moe_gemm_marlin_w8a8: {ms_m:.4f} ms")
except Exception as e:
    print(f"  Marlin: {str(e)[:120]}")

# Also try m_grouped_w8a8_gemm_nt_masked (deepgemm)
print("\n--- deepgemm m_grouped_w8a8_gemm_nt_masked ---")
try:
    import deepgemm
    sig = inspect.signature(deepgemm.m_grouped_w8a8_gemm_nt_masked)
    print("sig:", sig)
except Exception as e:
    print(f"  deepgemm: {e}")

print("\n--- Summary ---")
print("Key finding: which kernel dominates MoE decode time?")
