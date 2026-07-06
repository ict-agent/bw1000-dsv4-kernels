"""Benchmark all 14 HIP kernels vs SOTA references (lmslim/torch/TileLang-equiv).
Output perf report + JSON.
"""
import torch, ctypes, sys, json, time
sys.path.insert(0,"/workspace/dsv4_ops_unit_tests")
from utils import model_config as C
lib=ctypes.CDLL("/workspace/hip_kernels/libdsv4_all_hip.so")
P=ctypes.c_void_p
lib.launch_ptq.argtypes=[P,P,P,ctypes.c_int,ctypes.c_int,P]
lib.launch_ptgq.argtypes=[P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
lib.launch_rmsnorm_self.argtypes=[P,ctypes.c_int,ctypes.c_int,ctypes.c_float,P]
lib.launch_fused_rope.argtypes=[P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
lib.launch_silu_mul.argtypes=[P,P,P,ctypes.c_int,ctypes.c_int,P]
lib.launch_silu_mul_masked_quant.argtypes=[P,P,P,P,P,ctypes.c_int,ctypes.c_int,P]
lib.launch_hc_split_sinkhorn.argtypes=[P,P,P,P,P,P,ctypes.c_int,P]
lib.launch_act_quant_fp8.argtypes=[P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
lib.launch_merge_attn_states.argtypes=[P,P,P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
lib.launch_topk_transform.argtypes=[P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
lib.launch_mhc_pre.argtypes=[P,P,P,P,ctypes.c_int,ctypes.c_int,P]
lib.launch_mhc_post.argtypes=[P,P,P,P,P,ctypes.c_int,ctypes.c_int,P]
lib.launch_swa_prefill_indices.argtypes=[P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
lib.launch_grouped_gemm_int8.argtypes=[P,P,P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]

bf=torch.bfloat16;DEV="cuda";R=[]
def bench(fn,w=30,r=300):
    for _ in range(w):fn()
    torch.cuda.synchronize();t0=time.time()
    for _ in range(r):fn()
    torch.cuda.synchronize();return (time.time()-t0)/r*1000
def rec(k,M,sota,hip):
    sp=sota/hip if hip>0 else 0
    R.append({"kernel":k,"M":M,"sota_ms":round(sota,4),"hip_ms":round(hip,4),"speedup":round(sp,2)})
    print(f"  {k:28s} M={M:>4} SOTA={sota:.4f}ms HIP={hip:.4f}ms speedup={sp:.2f}x")

s=torch.cuda.current_stream().cuda_stream
print("="*70);print("PERF BENCHMARK — 14 HIP kernels vs SOTA");print("="*70)

# 1 ptq
from lmslim.layers.gemm.int8_utils import per_token_quant_int8 as sota_ptq
for M in [1,64,256,1024]:
    N=C.HIDDEN_SIZE;x=torch.randn(M,N,device=DEV,dtype=bf)
    q=torch.empty(M,N,device=DEV,dtype=torch.int8);sc=torch.empty(M,device=DEV,dtype=torch.float32)
    rec("per_token_quant_int8",M,bench(lambda:sota_ptq(x)),bench(lambda:lib.launch_ptq(x.data_ptr(),q.data_ptr(),sc.data_ptr(),M,N,s)))

# 2 ptgq
from lmslim.layers.gemm.int8_utils import per_token_group_quant_int8 as sota_ptgq
gs=C.MOE_GROUP_SIZE
for M in [1,64,256]:
    N=C.HIDDEN_SIZE;x=torch.randn(M,N,device=DEV,dtype=bf)
    q=torch.empty(M,N,device=DEV,dtype=torch.int8);ng=N//gs;sc=torch.empty(M,ng,device=DEV,dtype=torch.float32)
    rec("per_token_group_quant_int8",M,bench(lambda:sota_ptgq(x,gs)),bench(lambda:lib.launch_ptgq(x.data_ptr(),q.data_ptr(),sc.data_ptr(),M,N,gs,s)))

# 3 rmsnorm
for M in [1,64,128,1024]:
    N=C.HEAD_DIM;x=torch.randn(M,N,device=DEV,dtype=bf)
    def ref():return x*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+1e-6).to(bf)
    rec("rmsnorm_self",M,bench(ref),bench(lambda:(x.clone(),lib.launch_rmsnorm_self(x.data_ptr(),M,N,1e-6,s))[1]))

# 4 fused_rope
def precompute(dim,sl):
    th=1.0/(10000.0**(torch.arange(0,dim,2,device=DEV).float()/dim))
    fr=torch.outer(torch.arange(sl,device=DEV).float(),th)
    return torch.stack([torch.cos(fr),torch.sin(fr)],-1).reshape(sl,dim).contiguous()
rd=C.QK_ROPE_HEAD_DIM;freqs=precompute(rd,8192)
for nt in [1,32,256,1024]:
    nq,nk=8,2;q=torch.randn(nt,nq,rd,device=DEV,dtype=bf);k=torch.randn(nt,nk,rd,device=DEV,dtype=bf);pos=torch.arange(nt,device=DEV,dtype=torch.int32)
    def ref():
        f=freqs[pos].view(nt,-1,2).unsqueeze(1);t=q.float().view(nt,nq,-1,2)
        return torch.stack([t[...,0]*f[...,0]-t[...,1]*f[...,1],t[...,0]*f[...,1]+t[...,1]*f[...,0]],-1).reshape(q.shape).to(bf)
    rec("fused_rope",nt,bench(ref),bench(lambda:lib.launch_fused_rope(q.data_ptr(),k.data_ptr(),freqs.data_ptr(),pos.data_ptr(),nt,nq,nk,rd,1,s)))

# 5 silu_mul
for M in [1,64,256,1024]:
    N=C.MOE_INTERMEDIATE_SIZE;g=torch.randn(M,N,device=DEV,dtype=bf);u=torch.randn(M,N,device=DEV,dtype=bf);o=torch.empty_like(g)
    rec("silu_and_mul",M,bench(lambda:(torch.sigmoid(g.float())*g*u).to(bf)),bench(lambda:lib.launch_silu_mul(g.data_ptr(),u.data_ptr(),o.data_ptr(),M,N,s)))

# 6 silu_mul_masked_quant  (SOTA: silu+mul + per_token_quant separate)
for M in [64,256]:
    N=C.MOE_INTERMEDIATE_SIZE;g=torch.randn(M,N,device=DEV,dtype=bf);u=torch.randn(M,N,device=DEV,dtype=bf);mask=torch.ones(M,device=DEV,dtype=torch.int32)
    q=torch.empty(M,N,device=DEV,dtype=torch.int8);sc=torch.empty(M,device=DEV,dtype=torch.float32)
    def ref():return sota_ptq((torch.sigmoid(g.float())*g*u).to(bf))
    rec("silu_mul_masked_quant",M,bench(ref),bench(lambda:lib.launch_silu_mul_masked_quant(g.data_ptr(),u.data_ptr(),mask.data_ptr(),q.data_ptr(),sc.data_ptr(),M,N,s)))

# 7 hc_split_sinkhorn  (SOTA: torch reference)
hc=C.HC_MULT;mh=(2+hc)*hc
def ref_sk(mixes,sc,base,n):
    pre=torch.sigmoid(mixes[:,:hc]*sc[0]+base[:hc])+1e-6
    post=2*torch.sigmoid(mixes[:,hc:2*hc]*sc[1]+base[hc:2*hc])
    comb=mixes[:,2*hc:].reshape(n,hc,hc)*sc[2]+base[2*hc:].reshape(1,hc,hc)
    rmax=comb.amax(-1,keepdim=True);comb=torch.exp(comb-rmax)
    comb=comb/(comb.sum(-1,keepdim=True)+1e-6);comb=comb/(comb.sum(-2,keepdim=True)+1e-6)
    for _ in range(19):
        comb=comb/(comb.sum(-1,keepdim=True)+1e-6);comb=comb/(comb.sum(-2,keepdim=True)+1e-6)
    return pre,post,comb
for n in [64,256,1024]:
    mixes=torch.randn(n,mh,device=DEV,dtype=torch.float32);sc=torch.tensor([0.5,0.5,0.1],device=DEV,dtype=torch.float32);base=torch.randn(mh,device=DEV,dtype=torch.float32)
    pre=torch.empty(n,hc,device=DEV,dtype=torch.float32);post=torch.empty(n,hc,device=DEV,dtype=torch.float32);comb=torch.empty(n,hc,hc,device=DEV,dtype=torch.float32)
    rec("hc_split_sinkhorn",n,bench(lambda:ref_sk(mixes,sc,base,n)),bench(lambda:lib.launch_hc_split_sinkhorn(mixes.data_ptr(),sc.data_ptr(),base.data_ptr(),pre.data_ptr(),post.data_ptr(),comb.data_ptr(),n,s)))

# 8 act_quant_fp8  (SOTA: torch per-token-group)
fp8m=448.0
for M in [1,64,256]:
    N=C.HIDDEN_SIZE;ng=N//128;x=torch.randn(M,N,device=DEV,dtype=bf);y=torch.empty(M,N,device=DEV,dtype=torch.uint8);sc=torch.empty(M,ng,device=DEV,dtype=torch.float32)
    def ref():
        xr=x.float().reshape(M,ng,128);am=xr.abs().amax(-1).clamp(min=1e-4)/fp8m
        return torch.clamp(xr/(am.unsqueeze(-1)+1e-12),-fp8m,fp8m).to(torch.float8_e4m3fn)
    rec("act_quant_fp8",M,bench(ref),bench(lambda:lib.launch_act_quant_fp8(x.data_ptr(),y.data_ptr(),sc.data_ptr(),M,N,128,s)))

# 9 merge_attn_states  (SOTA: torch LSE merge)
nt,nh,hs=256,16,C.HEAD_DIM
po=torch.randn(nt,nh,hs,device=DEV,dtype=bf);so=torch.randn(nt,nh,hs,device=DEV,dtype=bf)
pl=torch.randn(nh,nt,device=DEV,dtype=torch.float32);sl=torch.randn(nh,nt,device=DEV,dtype=torch.float32)
out=torch.empty(nt,nh,hs,device=DEV,dtype=bf);ol=torch.empty(nh,nt,device=DEV,dtype=torch.float32)
def ref_merge():
    p=pl.permute(1,0);s2=sl.permute(1,0);mx=torch.max(p,s2)
    pe=torch.exp(p-mx);se=torch.exp(s2-mx);ss=pe+se
    return (po*(pe/ss).unsqueeze(-1)+so*(se/ss).unsqueeze(-1)).to(bf)
rec("merge_attn_states",nt,bench(ref_merge),bench(lambda:lib.launch_merge_attn_states(out.data_ptr(),ol.data_ptr(),po.data_ptr(),pl.data_ptr(),so.data_ptr(),sl.data_ptr(),nt,nh,hs,s)))

# 10 topk_transform  (SOTA: torch.topk + page transform)
b=8;k=C.INDEX_TOPK;cap=1024;ptr=1024
slens=torch.tensor([100,200,512,800,300,1024,150,512],device=DEV,dtype=torch.int32)
scores=torch.randn(b,cap,device=DEV,dtype=torch.float32);ptabs=torch.arange(b*ptr,device=DEV,dtype=torch.int32).reshape(b,ptr)
out=torch.full((b,k),-1,device=DEV,dtype=torch.int32)
def ref_topk():
    for bi in range(b):
        sl=slens[bi].item()
        if sl<=k: idx=torch.arange(sl,device=DEV)
        else: idx=scores[bi,:sl].topk(k).indices
        # page-transform
        out[bi,:len(idx)]=(ptabs[bi,idx]+0)  # page_size=1
    return out
rec("topk_transform_512",b,bench(ref_topk),bench(lambda:lib.launch_topk_transform(scores.data_ptr(),slens.data_ptr(),ptabs.data_ptr(),out.data_ptr(),b,cap,ptr,1,k,s)))

# 11 mhc_pre
M=256;im=torch.randn(M,hc,device=DEV,dtype=bf);sc3=torch.tensor([0.5],device=DEV,dtype=torch.float32);base3=torch.zeros(hc,device=DEV,dtype=torch.float32);o=torch.empty(M,hc,device=DEV,dtype=bf)
rec("mhc_pre",M,bench(lambda:(torch.sigmoid(im.float()*sc3[0]+base3)+1e-6).to(bf)),bench(lambda:lib.launch_mhc_pre(im.data_ptr(),sc3.data_ptr(),base3.data_ptr(),o.data_ptr(),M,hc,s)))

# 12 mhc_post
n=64;hidden=C.HIDDEN_SIZE
a=torch.randn(n,hc,hc,device=DEV,dtype=torch.float32);bb=torch.randn(n,hc,hidden,device=DEV,dtype=bf);cc=torch.randn(n,hc,device=DEV,dtype=torch.float32);dd=torch.randn(n,hidden,device=DEV,dtype=bf);xx=torch.empty(n,hc,hidden,device=DEV,dtype=bf)
def ref_post():
    return (cc.unsqueeze(-1)*dd.unsqueeze(1).float()+torch.einsum("nmh,nmc->nch",bb.float(),a)).to(bf)
rec("mhc_post",n,bench(ref_post),bench(lambda:lib.launch_mhc_post(a.data_ptr(),bb.data_ptr(),cc.data_ptr(),dd.data_ptr(),xx.data_ptr(),n,hidden,s)))

# 13 swa_prefill_indices
seq_q=torch.tensor([256,512,1024,512],device=DEV,dtype=torch.int32);seq_k=seq_q.clone()
cu=torch.cat([torch.zeros(1,device=DEV,dtype=torch.int32),seq_q.cumsum(0)]).to(torch.int32);nq=seq_q.sum().item()
idx=torch.full((nq,C.SLIDING_WINDOW),-1,device=DEV,dtype=torch.int32)
# SOTA: python ref (slow); just measure hip
rec("swa_prefill_indices",int(nq),0.0,bench(lambda:lib.launch_swa_prefill_indices(idx.data_ptr(),seq_k.data_ptr(),seq_q.data_ptr(),cu.data_ptr(),nq,4,C.SLIDING_WINDOW,s)))

# 14 grouped_gemm
E=4;M=16;K=C.HIDDEN_SIZE;N2=C.MOE_INTERMEDIATE_SIZE
A=torch.randint(-127,127,(E,M,K),device=DEV,dtype=torch.int8);B=torch.randint(-127,127,(E,N2,K),device=DEV,dtype=torch.int8)
sa=torch.rand(E,M,device=DEV,dtype=torch.float32)*0.01+0.001;sb=torch.rand(E,N2,device=DEV,dtype=torch.float32)*0.01+0.001;mm=torch.tensor([M]*E,device=DEV,dtype=torch.int32);Co=torch.empty(E,M,N2,device=DEV,dtype=bf)
def ref_gg():
    out=torch.empty(E,M,N2,device=DEV,dtype=torch.float32)
    for e in range(E): out[e]=(A[e].float()@B[e].float().T)*sa[e].unsqueeze(1)*sb[e].unsqueeze(0)
    return out
hip_ms=bench(lambda:lib.launch_grouped_gemm_int8(A.data_ptr(),B.data_ptr(),sa.data_ptr(),sb.data_ptr(),Co.data_ptr(),mm.data_ptr(),E,M,N2,K,s))
rec("grouped_gemm_int8",M,bench(ref_gg),hip_ms)
tops=2*E*M*N2*K/hip_ms/1e9 if hip_ms>0 else 0
R[-1]["TFlops"]=round(tops,1)
print(f"  grouped_gemm TFlops={tops:.1f}")

print("\n"+"="*70)
fast=sum(1 for r in R if r["speedup"]>=1.0)
print(f"Total={len(R)}  speedup>=1.0x: {fast}")
import os;os.makedirs("/workspace/hip_kernels/results",exist_ok=True)
json.dump(R,open("/workspace/hip_kernels/results/bench_all.json","w"),indent=2)
print("Saved /workspace/hip_kernels/results/bench_all.json")
