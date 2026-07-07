"""Benchmark hip_wrapper vs sglang SOTA — actual inference shapes (decode bs=256 TP=8).
Tests the Python wrapper layer (Layer 2), not raw kernel.
"""
import torch, sys, time, json
sys.path.insert(0, "/workspace/dsv4_ops_unit_tests")
sys.path.insert(0, "/workspace/sglang/python")
sys.path.insert(0, "/workspace/hip_kernels")
from utils import model_config as C
import hip_wrapper as W

DEV = "cuda"; bf = torch.bfloat16
def bench(fn, w=30, r=300):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(r): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/r*1000

R = []
def rec(k, sota_ms, hip_ms, sota_name):
    sp = sota_ms/hip_ms if hip_ms>0 and sota_ms>0 else 0
    R.append({"kernel":k,"sota":sota_name,"sota_ms":round(sota_ms,4),"hip_ms":round(hip_ms,4),"speedup":round(sp,2)})
    print(f"  {k:28s} {sota_name:18s} SOTA={sota_ms:.4f} HIP={hip_ms:.4f} sp={sp:.2f}x")

print("="*74); print("hip_wrapper vs sglang SOTA (actual decode shapes)"); print("="*74)

# 1. per_token_quant_int8  x [256,4096]
from lmslim.layers.gemm.int8_utils import per_token_quant_int8 as sota_ptq
x = torch.randn(256, C.HIDDEN_SIZE, device=DEV, dtype=bf)
rec("per_token_quant_int8", bench(lambda: sota_ptq(x)), bench(lambda: W.per_token_quant_int8(x)), "lmslim/triton")

# 2. per_token_group_quant_int8  x [1536,2048]
from lmslim.layers.gemm.int8_utils import per_token_group_quant_int8 as sota_ptgq
x = torch.randn(1536, C.MOE_INTERMEDIATE_SIZE, device=DEV, dtype=bf)
rec("per_token_group_quant_int8", bench(lambda: sota_ptgq(x,128)), bench(lambda: W.per_token_group_quant_int8(x,128)), "lmslim/triton")

# 3. silu_and_mul  x [1536,4096] -> out [1536,2048]
x = torch.randn(1536, C.MOE_INTERMEDIATE_SIZE*2, device=DEV, dtype=bf)
d = x.shape[-1]//2
def torch_silu():
    return (torch.sigmoid(x[...,:d].float())*x[...,:d]*x[...,d:]).to(bf)
rec("silu_and_mul", bench(torch_silu), bench(lambda: W.silu_and_mul(x)), "torch/ref")

# 4. rmsnorm_self  q [256,8,512]
q = torch.randn(256, 8, C.HEAD_DIM, device=DEV, dtype=bf)
def torch_rms():
    return q * torch.rsqrt(q.float().pow(2).mean(-1,keepdim=True)+1e-6).to(bf)
rec("rmsnorm_self", bench(torch_rms), bench(lambda: W.rmsnorm_self(q.clone(),1e-6)), "torch/ref")

# 5. fused_rope  q [256,8,64]
rd = C.QK_ROPE_HEAD_DIM
th = 1.0/(10000.0**(torch.arange(0,rd,2,device=DEV).float()/rd))
f = torch.outer(torch.arange(8192,device=DEV).float(), th)
fc = torch.view_as_complex(torch.stack([torch.cos(f),torch.sin(f)],-1)).contiguous()
pos = torch.arange(256, device=DEV, dtype=torch.int32)
q = torch.randn(256, 8, rd, device=DEV, dtype=bf)
def torch_rope():
    fr = torch.view_as_real(fc[pos]).unsqueeze(1)
    t = q.float().view(256,8,-1,2)
    return torch.stack([t[...,0]*fr[...,0]-t[...,1]*fr[...,1], t[...,0]*fr[...,1]+t[...,1]*fr[...,0]],-1).reshape(q.shape).to(bf)
rec("fused_rope", bench(torch_rope), bench(lambda: W.fused_rope(q.clone(),None,fc,pos)), "torch/ref")

# 6. hc_split_sinkhorn  mixes [256,1,24]
hc = C.HC_MULT; mh = (2+hc)*hc
mixes = torch.randn(256, 1, mh, device=DEV, dtype=torch.float32)
sc = torch.tensor([0.5,0.5,0.1], device=DEV, dtype=torch.float32)
base = torch.randn(mh, device=DEV, dtype=torch.float32)
def torch_sinkhorn():
    mf = mixes.view(256, mh)
    pre = torch.sigmoid(mf[:,:hc]*sc[0]+base[:hc])+1e-6
    comb = mf[:,2*hc:].reshape(256,hc,hc)*sc[2]+base[2*hc:].reshape(1,hc,hc)
    rmax = comb.amax(-1,keepdim=True); comb = torch.exp(comb-rmax)
    for _ in range(20):
        comb = comb/(comb.sum(-1,keepdim=True)+1e-6); comb = comb/(comb.sum(-2,keepdim=True)+1e-6)
    return comb
rec("hc_split_sinkhorn", bench(torch_sinkhorn), bench(lambda: W.hc_split_sinkhorn(mixes,sc,base,hc,20,1e-6)), "torch/ref")

# 7. act_quant  x [256,64,128]
x = torch.randn(256, C.INDEX_N_HEADS, C.INDEX_HEAD_DIM, device=DEV, dtype=bf)
def torch_act_quant():
    N = x.shape[-1]; fp8m=448.0
    xr = x.float().reshape(256,64,N//128,128)
    amax = xr.abs().amax(-1, keepdim=True).clamp(min=1e-4)  # [256,64,1,1]
    y = torch.clamp(xr/(amax/fp8m), -fp8m, fp8m).to(torch.float8_e4m3fn)
    return y
rec("act_quant_fp8", bench(torch_act_quant), bench(lambda: W.act_quant(x,128)), "torch/ref")

# 8. merge_attn_states  output [256,8,512]
nt,nh,hs = 256,8,C.HEAD_DIM
po = torch.randn(nt,nh,hs,device=DEV,dtype=bf); so = torch.randn(nt,nh,hs,device=DEV,dtype=bf)
pl = torch.randn(nh,nt,device=DEV,dtype=torch.float32); sl = torch.randn(nh,nt,device=DEV,dtype=torch.float32)
out = torch.empty(nt,nh,hs,device=DEV,dtype=bf); ol = torch.empty(nh,nt,device=DEV,dtype=torch.float32)
def torch_merge():
    p=pl.permute(1,0); s2=sl.permute(1,0); mx=torch.max(p,s2)
    pe=torch.exp(p-mx); se=torch.exp(s2-mx); ss=pe+se
    return (po*(pe/ss).unsqueeze(-1)+so*(se/ss).unsqueeze(-1)).to(bf)
rec("merge_attn_states", bench(torch_merge), bench(lambda: W.merge_attn_states(out,po,pl,so,sl,ol)), "torch/ref")

# 9. topk_transform_512  scores [256,1024]
b,k,cap,ptr = 256, C.INDEX_TOPK, 1024, 1024
sl = torch.randint(100,cap,(b,),device=DEV,dtype=torch.int32)
sc2 = torch.randn(b,cap,device=DEV,dtype=torch.float32)
pt = torch.arange(b*ptr,device=DEV,dtype=torch.int32).reshape(b,ptr)
out2 = torch.full((b,k),-1,device=DEV,dtype=torch.int32)
def torch_topk():
    o = torch.full_like(out2, -1)
    for bi in range(b):
        sli = sl[bi].item()
        if sli <= k: idx = torch.arange(sli,device=DEV)
        else: idx = sc2[bi,:sli].topk(k).indices
        o[bi,:len(idx)] = idx
    return o
rec("topk_transform_512", bench(torch_topk), bench(lambda: W.topk_transform_512(sc2,sl,pt,out2,1)), "torch/ref")

# 10. mhc_post  x [256,4096], residual [256,4,4096]
n,hc,hidden = 256, C.HC_MULT, C.HIDDEN_SIZE
x = torch.randn(n,hidden,device=DEV,dtype=bf)
res = torch.randn(n,hc,hidden,device=DEV,dtype=bf)
plm = torch.randn(n,hc,1,device=DEV,dtype=torch.float32)
crm = torch.randn(n,hc,hc,device=DEV,dtype=torch.float32)
def torch_mhcpost():
    return plm.squeeze(-1).unsqueeze(-1)*x.unsqueeze(1).float()+torch.einsum("nij,njh->nih",crm,res.float())
rec("mhc_post", bench(torch_mhcpost), bench(lambda: W.mhc_post_torch(x,res,plm,crm)), "torch/ref")

print("\n"+"="*74)
print(f"Total={len(R)}  sp>=1.0x: {sum(1 for r in R if r['speedup']>=1.0)}")
import os; os.makedirs("/workspace/hip_kernels/results",exist_ok=True)
json.dump(R,open("/workspace/hip_kernels/results/bench_wrapper_perf.json","w"),indent=2)
print("Saved /workspace/hip_kernels/results/bench_wrapper_perf.json")
