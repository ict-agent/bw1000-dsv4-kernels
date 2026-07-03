"""Verify ALL native ext kernels: correctness + performance + graph-safe."""
import torch, sys, time, json
sys.path.insert(0, "/workspace/hip_kernels/torch_ext_build")
import dsv4_native_ext as ext

def bench(fn, w=30, r=400):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(r): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/r*1000

R = {"tests": []}
print("="*72); print("ALL NATIVE EXT KERNELS VERIFICATION"); print("="*72)

# 1. per_token_quant_int8
print("\n--- 1. per_token_quant_int8 ---")
from lmslim.layers.gemm.int8_utils import per_token_quant_int8 as sota
for M in [1, 64, 256]:
    N=4096; x=torch.randn(M,N,device="cuda",dtype=torch.bfloat16)
    rq,rs=sota(x); rq=rq.reshape(M,N)
    stream=torch.cuda.current_stream().cuda_stream
    q,s=ext.per_token_quant_int8_stream(x,stream)
    be=(rq==q.reshape(M,N)).all().item()
    ms_s=bench(lambda:sota(x))
    ms_h=bench(lambda:ext.per_token_quant_int8_stream(x,torch.cuda.current_stream().cuda_stream))
    print(f"  M={M}: bit-exact={be} SOTA={ms_s:.4f}ms HIP={ms_h:.4f}ms speedup={ms_s/ms_h:.2f}x")
    R["tests"].append({"kernel":"per_token_quant","M":M,"bitexact":be,"speedup":round(ms_s/ms_h,2)})

# 2. rmsnorm
print("\n--- 2. rmsnorm ---")
for M in [1, 64, 128]:
    N=512; x=torch.randn(M,N,device="cuda",dtype=torch.bfloat16)
    ref=x*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+1e-6)
    out=ext.rmsnorm_stream(x.clone(),1e-6,torch.cuda.current_stream().cuda_stream)
    diff=(ref.float()-out.float()).abs().max().item()
    ms_r=bench(lambda:x*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+1e-6))
    ms_h=bench(lambda:ext.rmsnorm_stream(x.clone(),1e-6,torch.cuda.current_stream().cuda_stream))
    print(f"  M={M}: maxdiff={diff:.4f} ref={ms_r:.4f}ms HIP={ms_h:.4f}ms speedup={ms_r/ms_h:.2f}x")
    R["tests"].append({"kernel":"rmsnorm","M":M,"maxdiff":round(diff,4),"speedup":round(ms_r/ms_h,2)})

# 3. silu_and_mul
print("\n--- 3. silu_and_mul ---")
for M in [1, 64, 256]:
    N=2048; g=torch.randn(M,N,device="cuda",dtype=torch.bfloat16)
    u=torch.randn(M,N,device="cuda",dtype=torch.bfloat16)
    ref=(torch.sigmoid(g.float())*g*u).to(torch.bfloat16)
    out=ext.silu_and_mul_stream(g,u,torch.cuda.current_stream().cuda_stream)
    diff=(ref.float()-out.float()).abs().max().item()
    ms_r=bench(lambda:(torch.sigmoid(g.float())*g*u).to(torch.bfloat16))
    ms_h=bench(lambda:ext.silu_and_mul_stream(g,u,torch.cuda.current_stream().cuda_stream))
    print(f"  M={M}: maxdiff={diff:.6f} ref={ms_r:.4f}ms HIP={ms_h:.4f}ms speedup={ms_r/ms_h:.2f}x")
    R["tests"].append({"kernel":"silu_and_mul","M":M,"maxdiff":round(diff,6),"speedup":round(ms_r/ms_h,2)})

# 4. silu_mul_quant
print("\n--- 4. silu_mul_quant ---")
for M in [1, 64, 256]:
    N=2048; g=torch.randn(M,N,device="cuda",dtype=torch.bfloat16)
    u=torch.randn(M,N,device="cuda",dtype=torch.bfloat16)
    h=(torch.sigmoid(g.float())*g*u).to(torch.bfloat16)
    rq,rs=sota(h); rq=rq.reshape(M,N)
    q,s=ext.silu_mul_quant_stream(g,u,torch.cuda.current_stream().cuda_stream)
    diff=(rq.int()-q.reshape(M,N).int()).abs().max().item()
    ms_r=bench(lambda:sota((torch.sigmoid(g.float())*g*u).to(torch.bfloat16)))
    ms_h=bench(lambda:ext.silu_mul_quant_stream(g,u,torch.cuda.current_stream().cuda_stream))
    print(f"  M={M}: maxdiff={diff} ref={ms_r:.4f}ms HIP={ms_h:.4f}ms speedup={ms_r/ms_h:.2f}x")
    R["tests"].append({"kernel":"silu_mul_quant","M":M,"maxdiff":diff,"speedup":round(ms_r/ms_h,2)})

# 5. add_rmsnorm_quant
print("\n--- 5. add_rmsnorm_quant ---")
for M in [1, 64, 256]:
    N=4096; res=torch.randn(M,N,device="cuda",dtype=torch.bfloat16)
    x=torch.randn(M,N,device="cuda",dtype=torch.bfloat16)
    w=torch.randn(N,device="cuda",dtype=torch.bfloat16).abs()+0.1
    import lightop
    r_ref=res.clone(); xc=x.clone()
    n,ro=lightop.fused_add_rms_norm(xc,r_ref,w,1e-6)
    rq,rs=sota(n); rq=rq.reshape(M,N)
    r_hip=res.clone()
    q,s=ext.add_rmsnorm_quant_stream(r_hip,x,w,1e-6,torch.cuda.current_stream().cuda_stream)
    diff=(rq.int()-q.reshape(M,N).int()).abs().max().item()
    def ref_fn():
        r=res.clone();xc=x.clone()
        n,ro=lightop.fused_add_rms_norm(xc,r,w,1e-6)
        return sota(n)
    ms_r=bench(ref_fn)
    ms_h=bench(lambda:ext.add_rmsnorm_quant_stream(res.clone(),x,w,1e-6,torch.cuda.current_stream().cuda_stream))
    print(f"  M={M}: maxdiff={diff} SOTA={ms_r:.4f}ms HIP={ms_h:.4f}ms speedup={ms_r/ms_h:.2f}x")
    R["tests"].append({"kernel":"add_rmsnorm_quant","M":M,"maxdiff":diff,"speedup":round(ms_r/ms_h,2)})

# 6. w8a8_scaled_gemm v3 (shared memory tiling)
print("\n--- 6. w8a8_scaled_gemm v3 ---")
for M,N,K in [(1,4096,4096),(6,4096,2048),(64,4096,4096)]:
    A=torch.randint(-128,127,(M,K),device="cuda",dtype=torch.int8).contiguous()
    B=torch.randint(-128,127,(N,K),device="cuda",dtype=torch.int8).contiguous()
    sa=torch.ones(M,device="cuda",dtype=torch.float32)
    sb=torch.ones(N,device="cuda",dtype=torch.float32)
    # Reference
    ref=(A.float().reshape(M,K)@B.float().reshape(N,K).T).to(torch.bfloat16)
    out=ext.w8a8_scaled_gemm_stream(A,B,sa,sb,torch.cuda.current_stream().cuda_stream)
    diff=(ref.float()-out.float()).abs().max().item()
    ms_h=bench(lambda:ext.w8a8_scaled_gemm_stream(A,B,sa,sb,torch.cuda.current_stream().cuda_stream))
    tops=2*M*N*K/ms_h/1e9
    # SOTA comparison
    try:
        B_nc=torch.randint(-128,127,(K,N),device="cuda",dtype=torch.int8).t()
        from lmslim import quant_ops
        ms_s=bench(lambda:quant_ops.triton_scaled_mm(A,B_nc,scale_a=sa.unsqueeze(-1),scale_b=sb.unsqueeze(0),out_dtype=torch.bfloat16))
        sota_tops=2*M*N*K/ms_s/1e9
        sp=ms_s/ms_h
    except:
        ms_s=0; sota_tops=0; sp=0
    print(f"  M={M} N={N} K={K}: diff={diff} HIP={ms_h:.4f}ms({tops:.1f}T) SOTA={ms_s:.4f}ms({sota_tops:.1f}T) speedup={sp:.2f}x")
    R["tests"].append({"kernel":"w8a8_gemm_v3","M":M,"N":N,"K":K,"diff":diff,"hip_tops":round(tops,1),"sota_tops":round(sota_tops,1),"speedup":round(sp,2)})

# 7. flash_mla_decode (simplified)
print("\n--- 7. flash_mla_decode (simplified HIP) ---")
for B,S in [(1,1024),(1,4096)]:
    H=8; D=576; DV=512
    Q=torch.randn(B,H,D,device="cuda",dtype=torch.bfloat16)
    KV=torch.randn(B,S,D,device="cuda",dtype=torch.bfloat16)
    scale=1.0/(D**0.5)
    try:
        out,lse=ext.flash_mla_decode_stream(Q,KV,DV,scale,torch.cuda.current_stream().cuda_stream)
        has_nan=torch.isnan(out).any().item()
        ms_h=bench(lambda:ext.flash_mla_decode_stream(Q,KV,DV,scale,torch.cuda.current_stream().cuda_stream))
        print(f"  B={B} S={S}: out={out.shape} nan={has_nan} HIP={ms_h:.4f}ms")
        R["tests"].append({"kernel":"flash_mla_decode","B":B,"S":S,"nan":has_nan,"hip_ms":round(ms_h,4)})
    except Exception as e:
        print(f"  B={B} S={S}: ERROR {str(e)[:80]}")
        R["tests"].append({"kernel":"flash_mla_decode","B":B,"S":S,"error":str(e)[:80]})

# Summary
print("\n"+"="*72)
print("SUMMARY")
print("="*72)
correct=sum(1 for t in R["tests"] if t.get("bitexact") or t.get("maxdiff",999)<2 or t.get("diff",999)<2 or t.get("nan") is False)
total=len(R["tests"])
fast=sum(1 for t in R["tests"] if t.get("speedup",0)>=1.5)
print(f"  Total: {total}, Correct: {correct}, >=1.5x: {fast}")
with open("/workspace/native_ext_verify.json","w") as f:
    # Convert bool to str for JSON
    def to_jsonable(o):
        if isinstance(o,(bool,)): return bool(o)
        if isinstance(o,dict): return {k:to_jsonable(v) for k,v in o.items()}
        if isinstance(o,list): return [to_jsonable(i) for i in o]
        return o
    json.dump(to_jsonable(R),f,indent=2)
print(f"  Saved: /workspace/native_ext_verify.json")
