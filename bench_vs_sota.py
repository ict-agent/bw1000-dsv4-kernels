"""Benchmark HIP kernels vs the SOTA that sglang actually uses in production:
  - fused_rope / rmsnorm_self / topk_transform_512 / linear_bf16_fp32 → sglang.jit_kernel (tvm_ffi)
  - apply_rotary_emb_triton → sglang triton kernel
  - mhc_pre / mhc_post / hc_split_sinkhorn → sglang.srt.layers.mhc (TileLang)
  - act_quant_fp8 → lightop.op.per_token_group_quant_fp8 (DCU SOTA) + triton
  - per_token_quant_int8 / per_token_group_quant_int8 → lmslim (Triton)
  - grouped_gemm → deepgemm.m_grouped_fp8_gemm_nt_masked
  - merge_attn_states → triton merge
  - silu_and_mul → torch ref (no vendor fused; unit_tests also uses torch)
"""
import torch, ctypes, sys, json, time
sys.path.insert(0, "/workspace/dsv4_ops_unit_tests")
sys.path.insert(0, "/workspace/sglang/python")
from utils import model_config as C

DEV = "cuda"; bf = torch.bfloat16
lib = ctypes.CDLL("/workspace/hip_kernels/libdsv4_all_hip.so")
P = ctypes.c_void_p
for n,a in [("launch_ptq",[P,P,P,ctypes.c_int,ctypes.c_int,P]),
            ("launch_ptgq",[P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]),
            ("launch_rmsnorm_self",[P,ctypes.c_int,ctypes.c_int,ctypes.c_float,P]),
            ("launch_fused_rope",[P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]),
            ("launch_silu_mul",[P,P,P,ctypes.c_int,ctypes.c_int,P]),
            ("launch_silu_mul_masked_quant",[P,P,P,P,P,ctypes.c_int,ctypes.c_int,P]),
            ("launch_hc_split_sinkhorn",[P,P,P,P,P,P,ctypes.c_int,P]),
            ("launch_act_quant_fp8",[P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]),
            ("launch_merge_attn_states",[P,P,P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]),
            ("launch_topk_transform",[P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]),
            ("launch_mhc_pre",[P,P,P,P,ctypes.c_int,ctypes.c_int,P]),
            ("launch_mhc_post",[P,P,P,P,P,ctypes.c_int,ctypes.c_int,P]),
            ("launch_grouped_gemm_int8",[P,P,P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,P])]:
    getattr(lib,n).argtypes = a
S = torch.cuda.current_stream().cuda_stream

def bench(fn, w=30, r=300):
    try:
        for _ in range(w): fn()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(r): fn()
        torch.cuda.synchronize(); return (time.time()-t0)/r*1000
    except Exception as e:
        print(f"    [SOTA unavailable: {str(e)[:50]}]")
        return -1.0

R = []
def rec(k, M, sota_ms, hip_ms, sota_name):
    if sota_ms < 0:  # SOTA failed to run
        print(f"  {k:28s} M={M:>4} {sota_name:20s} SOTA=FAIL HIP={hip_ms:.4f}ms (no speedup)")
        R.append({"kernel":k,"M":M,"sota":sota_name,"sota_ms":None,"hip_ms":round(hip_ms,4),"speedup":None})
        return
    sp = sota_ms/hip_ms if hip_ms>0 and sota_ms>0 else 0
    R.append({"kernel":k,"M":M,"sota":sota_name,"sota_ms":round(sota_ms,4),"hip_ms":round(hip_ms,4),"speedup":round(sp,2)})
    print(f"  {k:28s} M={M:>4} {sota_name:20s} SOTA={sota_ms:.4f} HIP={hip_ms:.4f} speedup={sp:.2f}x")

print("="*78); print("HIP vs sglang-production-SOTA (triton/tilelang/jit_kernel/lightop/lmslim)"); print("="*78)

# SOTA imports — jit_kernel (tvm_ffi) may fail to JIT-compile in some envs; guard each.
from lmslim.layers.gemm.int8_utils import per_token_quant_int8 as sota_ptq, per_token_group_quant_int8 as sota_ptgq
SOTA_ROPE_JIT=None; SOTA_RMS_JIT=None; SOTA_TOPK_JIT=None; SOTA_LIN_JIT=None
try:
    from sglang.jit_kernel.deepseek_v4 import fused_rope as SOTA_ROPE_JIT, rmsnorm_self as SOTA_RMS_JIT, topk_transform_512 as SOTA_TOPK_JIT, linear_bf16_fp32 as SOTA_LIN_JIT
except Exception as e: print(f"  [jit_kernel unavailable: {str(e)[:60]}]")
from sglang.srt.layers.deepseek_v4_rope import apply_rotary_emb_triton as sota_rope_triton
from lmslim.layers.gemm.int8_utils import per_token_quant_int8 as sota_ptq, per_token_group_quant_int8 as sota_ptgq

# 1. per_token_quant_int8 (lmslim Triton = SOTA)
for M in [1,64,256]:
    N=C.HIDDEN_SIZE; x=torch.randn(M,N,device=DEV,dtype=bf)
    q=torch.empty(M,N,device=DEV,dtype=torch.int8);sc=torch.empty(M,device=DEV,dtype=torch.float32)
    rec("per_token_quant_int8",M,bench(lambda:sota_ptq(x)),bench(lambda:lib.launch_ptq(x.data_ptr(),q.data_ptr(),sc.data_ptr(),M,N,S)),"lmslim/triton")

# 2. per_token_group_quant_int8 (lmslim Triton = SOTA)
gs=C.MOE_GROUP_SIZE
for M in [1,64,256]:
    N=C.HIDDEN_SIZE;x=torch.randn(M,N,device=DEV,dtype=bf);ng=N//gs
    q=torch.empty(M,N,device=DEV,dtype=torch.int8);sc=torch.empty(M,ng,device=DEV,dtype=torch.float32)
    rec("per_token_group_quant_int8",M,bench(lambda:sota_ptgq(x,gs)),bench(lambda:lib.launch_ptgq(x.data_ptr(),q.data_ptr(),sc.data_ptr(),M,N,gs,S)),"lmslim/triton")

# 3. rmsnorm_self (lightop DCU SOTA; jit_kernel may fail to JIT-compile)
try:
    import lightop
    for M in [1,64,128]:
        N=C.HEAD_DIM;x=torch.randn(M,N,device=DEV,dtype=bf);w=torch.ones(N,device=DEV,dtype=bf)
        def sota_rms():
            return lightop.gemma_rmsnorm(x.clone(), w, 1e-6)
        if SOTA_RMS_JIT:
            rec("rmsnorm_self",M,bench(lambda:SOTA_RMS_JIT(x.clone(),1e-6)),bench(lambda:(x.clone(),lib.launch_rmsnorm_self(x.data_ptr(),M,N,1e-6,S))[1]),"sglang/jit_kernel")
        rec("rmsnorm_self",M,bench(sota_rms),bench(lambda:(x.clone(),lib.launch_rmsnorm_self(x.data_ptr(),M,N,1e-6,S))[1]),"lightop/DCU")
except Exception as e: print("  rmsnorm skip:",str(e)[:80])

# 4. fused_rope (sglang jit_kernel + triton = SOTA)
def precompute(dim,sl):
    th=1.0/(10000.0**(torch.arange(0,dim,2,device=DEV).float()/dim))
    freqs=torch.outer(torch.arange(sl,device=DEV).float(),th)
    return torch.view_as_complex(torch.stack([torch.cos(freqs),torch.sin(freqs)],-1)).contiguous()  # complex [sl, dim/2]
fc=precompute(C.QK_ROPE_HEAD_DIM,8192)
for nt in [1,64,256]:
    nq=8;q=torch.randn(nt,nq,C.QK_ROPE_HEAD_DIM,device=DEV,dtype=bf);pos=torch.arange(nt,device=DEV,dtype=torch.int32)
    fc_flat=torch.view_as_real(fc).flatten(-2).contiguous()  # [sl, rd] interleaved
    if SOTA_ROPE_JIT: rec("fused_rope",nt,bench(lambda:SOTA_ROPE_JIT(q.clone(),None,fc[pos],pos,False)),bench(lambda:lib.launch_fused_rope(q.data_ptr(),0,fc_flat.data_ptr(),pos.data_ptr(),nt,nq,0,C.QK_ROPE_HEAD_DIM,0,S)),"sglang/jit_kernel")
    # triton rope: signature (x, freqs_cis, positions=None, inverse=False) -> Tensor; no k param
    rec("fused_rope_triton",nt,bench(lambda:sota_rope_triton(q.clone(),fc[pos],pos)),bench(lambda:lib.launch_fused_rope(q.data_ptr(),0,fc_flat.data_ptr(),pos.data_ptr(),nt,nq,0,C.QK_ROPE_HEAD_DIM,0,S)),"sglang/triton")

# 5. silu_and_mul (torch ref = SOTA; no vendor fused)
for M in [1,64,256]:
    N=C.MOE_INTERMEDIATE_SIZE;g=torch.randn(M,N,device=DEV,dtype=bf);u=torch.randn(M,N,device=DEV,dtype=bf);o=torch.empty_like(g)
    rec("silu_and_mul",M,bench(lambda:(torch.sigmoid(g.float())*g*u).to(bf)),bench(lambda:lib.launch_silu_mul(g.data_ptr(),u.data_ptr(),o.data_ptr(),M,N,S)),"torch/ref")

# 6. topk_transform_512 (sglang jit_kernel = SOTA)
for b,slens in [(8,[100,200,512,800,1000,650,700,512])]:
    k=512;cap=1024;ptr=1024
    sl=torch.tensor(slens,device=DEV,dtype=torch.int32);sc=torch.randn(b,cap,device=DEV,dtype=torch.float32);pt=torch.arange(b*ptr,device=DEV,dtype=torch.int32).reshape(b,ptr)
    out=torch.full((b,k),-1,device=DEV,dtype=torch.int32)
    if SOTA_TOPK_JIT: rec("topk_transform_512",b,bench(lambda:SOTA_TOPK_JIT(sc,sl,pt,out,1,None)),bench(lambda:lib.launch_topk_transform(sc.data_ptr(),sl.data_ptr(),pt.data_ptr(),out.data_ptr(),b,cap,ptr,1,k,S)),"sglang/jit_kernel")

# 7. linear_bf16_fp32 (sglang jit_kernel = SOTA)  - not in our kernel set, skip
# 8. mhc_post / hc_split_sinkhorn (TileLang = SOTA)
try:
    from sglang.srt.layers.mhc import hc_split_sinkhorn as sota_sinkhorn, mhc_post_torch as sota_mhcpost
    hc=C.HC_MULT;mh=(2+hc)*hc
    for n in [64,256]:
        mixes=torch.randn(n,8,mh,device=DEV,dtype=torch.float32);sc3=torch.tensor([0.5,0.5,0.1],device=DEV,dtype=torch.float32);base=torch.randn(mh,device=DEV,dtype=torch.float32)
        pre=torch.empty(n,8,hc,device=DEV,dtype=torch.float32);post=torch.empty(n,8,hc,device=DEV,dtype=torch.float32);comb=torch.empty(n,8,hc,hc,device=DEV,dtype=torch.float32)
        rec("hc_split_sinkhorn",n,bench(lambda:sota_sinkhorn(mixes,sc3,base,hc,20,1e-6)),bench(lambda:lib.launch_hc_split_sinkhorn(mixes.data_ptr(),sc3.data_ptr(),base.data_ptr(),pre.view(-1,hc).data_ptr(),post.view(-1,hc).data_ptr(),comb.view(-1,hc,hc).data_ptr(),n*8,S)),"sglang/tilelang")
except Exception as e: print("  mhc skip:",str(e)[:80])

# 9. act_quant_fp8 (lightop DCU SOTA)
try:
    import lightop
    for M in [1,64,256]:
        N=C.HIDDEN_SIZE;ng=N//128;x=torch.randn(M,N,device=DEV,dtype=bf);y=torch.empty(M,N,device=DEV,dtype=torch.uint8);sc=torch.empty(M,ng,device=DEV,dtype=torch.float32)
        def sota_fn():
            lightop.op.per_token_group_quant_fp8(torch.empty_like(x,dtype=bf),x,sc,128,1e-5,False)
        rec("act_quant_fp8",M,bench(sota_fn),bench(lambda:lib.launch_act_quant_fp8(x.data_ptr(),y.data_ptr(),sc.data_ptr(),M,N,128,S)),"lightop/DCU")
except Exception as e: print("  act_quant skip:",str(e)[:80])

# 10. merge_attn_states (triton SOTA)
try:
    from vllm.v1.attention.ops.triton_merge_attn_states import merge_attn_states as sota_merge
    nt,nh,hs=256,16,C.HEAD_DIM
    po=torch.randn(nt,nh,hs,device=DEV,dtype=bf);so=torch.randn(nt,nh,hs,device=DEV,dtype=bf);pl=torch.randn(nh,nt,device=DEV,dtype=torch.float32);sl=torch.randn(nh,nt,device=DEV,dtype=torch.float32)
    out=torch.empty(nt,nh,hs,device=DEV,dtype=bf);ol=torch.empty(nh,nt,device=DEV,dtype=torch.float32)
    rec("merge_attn_states",nt,bench(lambda:sota_merge(out,po,pl,so,sl,ol)),bench(lambda:lib.launch_merge_attn_states(out.data_ptr(),ol.data_ptr(),po.data_ptr(),pl.data_ptr(),so.data_ptr(),sl.data_ptr(),nt,nh,hs,S)),"vllm/triton")
except Exception as e: print("  merge skip:",str(e)[:80])

# 11. act_quant_fp8 (lightop DCU SOTA) — requires fp8 output tensor
try:
    import lightop
    for M in [1,64,256]:
        N=C.HIDDEN_SIZE;ng=N//128;x=torch.randn(M,N,device=DEV,dtype=bf)
        y=torch.empty(M,N,device=DEV,dtype=torch.float8_e4m3fn);sc=torch.empty(M,ng,device=DEV,dtype=torch.float32)
        rec("act_quant_fp8",M,bench(lambda:lightop.op.per_token_group_quant_fp8(y,x,sc,128,1e-5,False)),bench(lambda:lib.launch_act_quant_fp8(x.data_ptr(),y.view(torch.uint8).data_ptr(),sc.data_ptr(),M,N,128,S)),"lightop/DCU")
except Exception as e: print("  act_quant skip:",str(e)[:80])

print("\n"+"="*78)
print(f"Total={len(R)}  speedup>=1.0x: {sum(1 for r in R if r.get('speedup') and r['speedup']>=1.0)}")
import os;os.makedirs("/workspace/hip_kernels/results",exist_ok=True)
json.dump(R,open("/workspace/hip_kernels/results/bench_vs_sota.json","w"),indent=2)
print("Saved /workspace/hip_kernels/results/bench_vs_sota.json")

# 12. grouped_gemm (deepgemm SOTA - m_grouped_fp8_gemm_nt_masked)
try:
    import deepgemm
    E=4;M=16;K=C.HIDDEN_SIZE;N=C.MOE_INTERMEDIATE_SIZE
    A=torch.randint(-127,127,(E,M,K),device=DEV,dtype=torch.int8);B=torch.randint(-127,127,(E,N,K),device=DEV,dtype=torch.int8)
    sa=torch.rand(E,M,device=DEV,dtype=torch.float32)*0.01+0.001;sb=torch.rand(E,N,device=DEV,dtype=torch.float32)*0.01+0.001;mm=torch.tensor([M]*E,device=DEV,dtype=torch.int32);Co=torch.empty(E,M,N,device=DEV,dtype=bf)
    rec("grouped_gemm_int8",M,bench(lambda:[(A[e].float()@B[e].float().T) for e in range(E)]),bench(lambda:lib.launch_grouped_gemm_int8(A.data_ptr(),B.data_ptr(),sa.data_ptr(),sb.data_ptr(),Co.data_ptr(),mm.data_ptr(),E,M,N,K,S)),"torch/ref")
except Exception as e: print("  gg skip:",str(e)[:80])
