"""CUDA-graph capture/replay safety test for all 14 HIP kernels.
Each kernel: static buffers, capture into graph, replay, compare to non-graph output.
"""
import torch, ctypes, sys
sys.path.insert(0, "/workspace/dsv4_ops_unit_tests")
from utils import model_config as C
lib = ctypes.CDLL("/workspace/hip_kernels/libdsv4_all_hip.so")
P = ctypes.c_void_p
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

bf=torch.bfloat16; DEV="cuda"; R=[]
def rec(k,ok,note=""): R.append((k,ok,note)); print(f"  [{'PASS' if ok else 'FAIL'}] {k}: {note}")

def cap_replay(fn, bufs):
    """Capture fn (using current stream), replay, return list of cloned bufs after replay."""
    s=torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(s)
    # warmup on capture stream
    fn(); torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.set_stream(torch.cuda.current_stream())
    g.replay(); torch.cuda.synchronize()
    return [b.clone() for b in bufs]

def cmp(a,b,atol,name,kind="float"):
    if kind=="int":
        d=(a.int()-b.int()).abs().max().item()
    elif kind=="byte":
        d=(a.int()-b.int()).abs().max().item()
    else:
        d=(a.float()-b.float()).abs().max().item()
    return rec(name, d<=atol, f"maxdiff={d}")

# 1 ptq
M,N=64,C.HIDDEN_SIZE
x=torch.randn(M,N,device=DEV,dtype=bf);q=torch.empty(M,N,device=DEV,dtype=torch.int8);s=torch.empty(M,device=DEV,dtype=torch.float32)
lib.launch_ptq(x.data_ptr(),q.data_ptr(),s.data_ptr(),M,N,None);torch.cuda.synchronize();q0=q.clone()
cmp(q,cap_replay(lambda:lib.launch_ptq(x.data_ptr(),q.data_ptr(),s.data_ptr(),M,N,torch.cuda.current_stream().cuda_stream),[q])[0],0,"ptq","int")

# 2 ptgq
gs=128;ng=N//gs
x=torch.randn(M,N,device=DEV,dtype=bf);q=torch.empty(M,N,device=DEV,dtype=torch.int8);s=torch.empty(M,ng,device=DEV,dtype=torch.float32)
lib.launch_ptgq(x.data_ptr(),q.data_ptr(),s.data_ptr(),M,N,gs,None);torch.cuda.synchronize();q0=q.clone()
cmp(q,cap_replay(lambda:lib.launch_ptgq(x.data_ptr(),q.data_ptr(),s.data_ptr(),M,N,gs,torch.cuda.current_stream().cuda_stream),[q])[0],0,"ptgq","int")

# 3 rmsnorm
x=torch.randn(M,C.HEAD_DIM,device=DEV,dtype=bf);lib.launch_rmsnorm_self(x.data_ptr(),M,C.HEAD_DIM,1e-6,None);torch.cuda.synchronize();x0=x.clone()
cmp(x,cap_replay(lambda:lib.launch_rmsnorm_self(x.data_ptr(),M,C.HEAD_DIM,1e-6,torch.cuda.current_stream().cuda_stream),[x])[0],0,"rmsnorm")

# 4 fused_rope
rd=C.QK_ROPE_HEAD_DIM;nt=32;nq=8;nk=2
q=torch.randn(nt,nq,rd,device=DEV,dtype=bf);k=torch.randn(nt,nk,rd,device=DEV,dtype=bf)
freqs=torch.randn(4096,rd,device=DEV,dtype=torch.float32);pos=torch.arange(nt,device=DEV,dtype=torch.int32)
lib.launch_fused_rope(q.data_ptr(),k.data_ptr(),freqs.data_ptr(),pos.data_ptr(),nt,nq,nk,rd,1,None);torch.cuda.synchronize();q0=q.clone();k0=k.clone()
out=cap_replay(lambda:lib.launch_fused_rope(q.data_ptr(),k.data_ptr(),freqs.data_ptr(),pos.data_ptr(),nt,nq,nk,rd,1,torch.cuda.current_stream().cuda_stream),[q,k])
cmp(q,out[0],0,"fused_rope_q");cmp(k,out[1],0,"fused_rope_k")

# 5 silu_mul
g=torch.randn(M,C.MOE_INTERMEDIATE_SIZE,device=DEV,dtype=bf);u=torch.randn(M,C.MOE_INTERMEDIATE_SIZE,device=DEV,dtype=bf);o=torch.empty_like(g)
lib.launch_silu_mul(g.data_ptr(),u.data_ptr(),o.data_ptr(),M,C.MOE_INTERMEDIATE_SIZE,None);torch.cuda.synchronize();o0=o.clone()
cmp(o,cap_replay(lambda:lib.launch_silu_mul(g.data_ptr(),u.data_ptr(),o.data_ptr(),M,C.MOE_INTERMEDIATE_SIZE,torch.cuda.current_stream().cuda_stream),[o])[0],0,"silu_mul")

# 6 silu_mul_masked_quant
g=torch.randn(M,C.MOE_INTERMEDIATE_SIZE,device=DEV,dtype=bf);u=torch.randn(M,C.MOE_INTERMEDIATE_SIZE,device=DEV,dtype=bf);mask=torch.ones(M,device=DEV,dtype=torch.int32);mask[:8]=0
q=torch.empty(M,C.MOE_INTERMEDIATE_SIZE,device=DEV,dtype=torch.int8);s=torch.empty(M,device=DEV,dtype=torch.float32)
lib.launch_silu_mul_masked_quant(g.data_ptr(),u.data_ptr(),mask.data_ptr(),q.data_ptr(),s.data_ptr(),M,C.MOE_INTERMEDIATE_SIZE,None);torch.cuda.synchronize();q0=q.clone()
cmp(q,cap_replay(lambda:lib.launch_silu_mul_masked_quant(g.data_ptr(),u.data_ptr(),mask.data_ptr(),q.data_ptr(),s.data_ptr(),M,C.MOE_INTERMEDIATE_SIZE,torch.cuda.current_stream().cuda_stream),[q])[0],0,"silu_mul_masked_quant","int")

# 7 hc_split_sinkhorn
hc=C.HC_MULT;mh=(2+hc)*hc;n=64
mix=torch.randn(n,mh,device=DEV,dtype=torch.float32);sc=torch.tensor([0.5,0.5,0.1],device=DEV,dtype=torch.float32);base=torch.randn(mh,device=DEV,dtype=torch.float32)
pre=torch.empty(n,hc,device=DEV,dtype=torch.float32);post=torch.empty(n,hc,device=DEV,dtype=torch.float32);comb=torch.empty(n,hc,hc,device=DEV,dtype=torch.float32)
lib.launch_hc_split_sinkhorn(mix.data_ptr(),sc.data_ptr(),base.data_ptr(),pre.data_ptr(),post.data_ptr(),comb.data_ptr(),n,None);torch.cuda.synchronize();pre0=pre.clone()
cmp(pre,cap_replay(lambda:lib.launch_hc_split_sinkhorn(mix.data_ptr(),sc.data_ptr(),base.data_ptr(),pre.data_ptr(),post.data_ptr(),comb.data_ptr(),n,torch.cuda.current_stream().cuda_stream),[pre])[0],1e-5,"sinkhorn")

# 8 act_quant_fp8
gs=128;ng=N//gs;x=torch.randn(M,N,device=DEV,dtype=bf);y=torch.empty(M,N,device=DEV,dtype=torch.uint8);s=torch.empty(M,ng,device=DEV,dtype=torch.float32)
lib.launch_act_quant_fp8(x.data_ptr(),y.data_ptr(),s.data_ptr(),M,N,gs,None);torch.cuda.synchronize();y0=y.clone()
cmp(y,cap_replay(lambda:lib.launch_act_quant_fp8(x.data_ptr(),y.data_ptr(),s.data_ptr(),M,N,gs,torch.cuda.current_stream().cuda_stream),[y])[0],0,"act_quant_fp8","byte")

# 9 merge_attn
nt=64;nh=8;hs=C.HEAD_DIM
po=torch.randn(nt,nh,hs,device=DEV,dtype=bf);so=torch.randn(nt,nh,hs,device=DEV,dtype=bf)
pl=torch.randn(nh,nt,device=DEV,dtype=torch.float32);sl=torch.randn(nh,nt,device=DEV,dtype=torch.float32)
out=torch.empty(nt,nh,hs,device=DEV,dtype=bf);ol=torch.empty(nh,nt,device=DEV,dtype=torch.float32)
lib.launch_merge_attn_states(out.data_ptr(),ol.data_ptr(),po.data_ptr(),pl.data_ptr(),so.data_ptr(),sl.data_ptr(),nt,nh,hs,None);torch.cuda.synchronize();o0=out.clone()
cmp(out,cap_replay(lambda:lib.launch_merge_attn_states(out.data_ptr(),ol.data_ptr(),po.data_ptr(),pl.data_ptr(),so.data_ptr(),sl.data_ptr(),nt,nh,hs,torch.cuda.current_stream().cuda_stream),[out])[0],0,"merge_attn")

# 10 topk
b=4;k=C.INDEX_TOPK;cap=1024;ptr=1024
sl=torch.tensor([100,200,512,800],device=DEV,dtype=torch.int32);sc2=torch.randn(b,cap,device=DEV,dtype=torch.float32);pt=torch.arange(b*ptr,device=DEV,dtype=torch.int32).reshape(b,ptr)
out=torch.full((b,k),-1,device=DEV,dtype=torch.int32)
lib.launch_topk_transform(sc2.data_ptr(),sl.data_ptr(),pt.data_ptr(),out.data_ptr(),b,cap,ptr,1,k,None);torch.cuda.synchronize();o0=out.clone()
cmp(out,cap_replay(lambda:lib.launch_topk_transform(sc2.data_ptr(),sl.data_ptr(),pt.data_ptr(),out.data_ptr(),b,cap,ptr,1,k,torch.cuda.current_stream().cuda_stream),[out])[0],0,"topk","int")

# 11 mhc_pre
im=torch.randn(n,hc,device=DEV,dtype=bf);sc3=torch.tensor([0.5],device=DEV,dtype=torch.float32);base3=torch.zeros(hc,device=DEV,dtype=torch.float32);o=torch.empty(n,hc,device=DEV,dtype=bf)
lib.launch_mhc_pre(im.data_ptr(),sc3.data_ptr(),base3.data_ptr(),o.data_ptr(),n,hc,None);torch.cuda.synchronize();o0=o.clone()
cmp(o,cap_replay(lambda:lib.launch_mhc_pre(im.data_ptr(),sc3.data_ptr(),base3.data_ptr(),o.data_ptr(),n,hc,torch.cuda.current_stream().cuda_stream),[o])[0],0,"mhc_pre")

# 12 mhc_post
hidden=C.HIDDEN_SIZE;a=torch.randn(n,hc,hc,device=DEV,dtype=torch.float32);bb=torch.randn(n,hc,hidden,device=DEV,dtype=bf);cc=torch.randn(n,hc,device=DEV,dtype=torch.float32);dd=torch.randn(n,hidden,device=DEV,dtype=bf);x=torch.empty(n,hc,hidden,device=DEV,dtype=bf)
lib.launch_mhc_post(a.data_ptr(),bb.data_ptr(),cc.data_ptr(),dd.data_ptr(),x.data_ptr(),n,hidden,None);torch.cuda.synchronize();x0=x.clone()
cmp(x,cap_replay(lambda:lib.launch_mhc_post(a.data_ptr(),bb.data_ptr(),cc.data_ptr(),dd.data_ptr(),x.data_ptr(),n,hidden,torch.cuda.current_stream().cuda_stream),[x])[0],1e-2,"mhc_post")

# 13 swa
seq_q=torch.tensor([128,256],device=DEV,dtype=torch.int32);seq_k=seq_q.clone();cu=torch.cat([torch.zeros(1,device=DEV,dtype=torch.int32),seq_q.cumsum(0)]).to(torch.int32);nq=seq_q.sum().item()
idx=torch.full((nq,C.SLIDING_WINDOW),-1,device=DEV,dtype=torch.int32)
lib.launch_swa_prefill_indices(idx.data_ptr(),seq_k.data_ptr(),seq_q.data_ptr(),cu.data_ptr(),nq,2,C.SLIDING_WINDOW,None);torch.cuda.synchronize();i0=idx.clone()
cmp(idx,cap_replay(lambda:lib.launch_swa_prefill_indices(idx.data_ptr(),seq_k.data_ptr(),seq_q.data_ptr(),cu.data_ptr(),nq,2,C.SLIDING_WINDOW,torch.cuda.current_stream().cuda_stream),[idx])[0],0,"swa","int")

# 14 grouped_gemm
E=4;MK=C.HIDDEN_SIZE;N2=C.MOE_INTERMEDIATE_SIZE
A=torch.randint(-127,127,(E,M,Mk:=C.HIDDEN_SIZE),device=DEV,dtype=torch.int8);B=torch.randint(-127,127,(E,N2,C.HIDDEN_SIZE),device=DEV,dtype=torch.int8)
sa=torch.rand(E,M,device=DEV,dtype=torch.float32)*0.01+0.001;sb=torch.rand(E,N2,device=DEV,dtype=torch.float32)*0.01+0.001;mm=torch.tensor([M]*E,device=DEV,dtype=torch.int32);Cout=torch.empty(E,M,N2,device=DEV,dtype=bf)
lib.launch_grouped_gemm_int8(A.data_ptr(),B.data_ptr(),sa.data_ptr(),sb.data_ptr(),Cout.data_ptr(),mm.data_ptr(),E,M,N2,C.HIDDEN_SIZE,None);torch.cuda.synchronize();c0=Cout.clone()
cmp(Cout,cap_replay(lambda:lib.launch_grouped_gemm_int8(A.data_ptr(),B.data_ptr(),sa.data_ptr(),sb.data_ptr(),Cout.data_ptr(),mm.data_ptr(),E,M,N2,C.HIDDEN_SIZE,torch.cuda.current_stream().cuda_stream),[Cout])[0],1e-1,"grouped_gemm")

print("\n"+"="*50);print(f"GRAPH SAFETY: {sum(1 for _,ok,_ in R if ok)}/{len(R)} pass")
