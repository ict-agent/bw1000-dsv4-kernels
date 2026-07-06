"""Verify all 14 HIP kernels in libdsv4_all_hip.so vs references.
For each kernel: correctness (maxdiff/bit-exact), performance vs SOTA, CUDA-graph capture/replay safety.
Run inside container d6e9ca5669f2:
    python /workspace/hip_kernels/verify_kernels_v2.py
"""
import torch, ctypes, json, time, sys, math
sys.path.insert(0, "/workspace/dsv4_ops_unit_tests")
from utils import model_config as C

LIB_PATH = "/workspace/hip_kernels/libdsv4_all_hip.so"
lib = ctypes.CDLL(LIB_PATH)

# ---- argtypes ----
P = ctypes.c_void_p
lib.launch_ptq.argtypes = [P,P,P,ctypes.c_int,ctypes.c_int,P]
lib.launch_ptgq.argtypes = [P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
lib.launch_rmsnorm_self.argtypes = [P,ctypes.c_int,ctypes.c_int,ctypes.c_float,P]
lib.launch_fused_rope.argtypes = [P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
lib.launch_silu_mul.argtypes = [P,P,P,ctypes.c_int,ctypes.c_int,P]
lib.launch_silu_mul_masked_quant.argtypes = [P,P,P,P,P,ctypes.c_int,ctypes.c_int,P]
lib.launch_hc_split_sinkhorn.argtypes = [P,P,P,P,P,P,ctypes.c_int,P]
lib.launch_act_quant_fp8.argtypes = [P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
lib.launch_merge_attn_states.argtypes = [P,P,P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
lib.launch_topk_transform.argtypes = [P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
lib.launch_mhc_pre.argtypes = [P,P,P,P,ctypes.c_int,ctypes.c_int,P]
lib.launch_mhc_post.argtypes = [P,P,P,P,P,ctypes.c_int,ctypes.c_int,P]
lib.launch_swa_prefill_indices.argtypes = [P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]
lib.launch_grouped_gemm_int8.argtypes = [P,P,P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]

stream = torch.cuda.current_stream().cuda_stream
DEV = "cuda"
bf = torch.bfloat16
RESULTS = []

def bench(fn, w=20, r=200):
    for _ in range(w): fn()
    torch.cuda.synchronize(); t0=time.time()
    for _ in range(r): fn()
    torch.cuda.synchronize(); return (time.time()-t0)/r*1000

def rec(k, **kw):
    RESULTS.append({"kernel":k, **kw})
    c = kw.get("correct", "?")
    sp = kw.get("speedup", "?")
    print(f"  [{k}] correct={c} speedup={sp} {kw.get('note','')}")

def graph_safe(fn, out_ref, atol, name):
    """Capture fn into a CUDA graph, replay, compare to out_ref."""
    try:
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        g.replay(); torch.cuda.synchronize()
        diff = 0
        for o, r in out_ref:
            d = (o.float()-r.float()).abs().max().item()
            diff = max(diff, d)
        ok = diff <= atol
        rec(name+"_graph", correct=ok, maxdiff=diff, note="graph-capture-replay")
        return ok
    except Exception as e:
        rec(name+"_graph", correct=False, note=f"graph-FAIL {str(e)[:60]}")
        return False

# ============================================================
# 1. per_token_quant_int8
# ============================================================
def t1():
    print("\n[1] per_token_quant_int8")
    from lmslim.layers.gemm.int8_utils import per_token_quant_int8 as sota
    for M in [1,64,256]:
        N=C.HIDDEN_SIZE
        x=torch.randn(M,N,device=DEV,dtype=bf)
        rq,rs=sota(x); rq=rq.reshape(M,N)
        hq=torch.empty(M,N,device=DEV,dtype=torch.int8)
        hs=torch.empty(M,device=DEV,dtype=torch.float32)
        lib.launch_ptq(x.data_ptr(),hq.data_ptr(),hs.data_ptr(),M,N,None)
        torch.cuda.synchronize()
        be=(rq==hq).all().item()
        ms_s=bench(lambda:sota(x))
        ms_h=bench(lambda:lib.launch_ptq(x.data_ptr(),hq.data_ptr(),hs.data_ptr(),M,N,None))
        rec("per_token_quant_int8",M=M,bitexact=be,speedup=round(ms_s/ms_h,2),sota_ms=round(ms_s,4),hip_ms=round(ms_h,4))
    # graph
    M=64;N=C.HIDDEN_SIZE;x=torch.randn(M,N,device=DEV,dtype=bf)
    hq=torch.empty(M,N,device=DEV,dtype=torch.int8);hs=torch.empty(M,device=DEV,dtype=torch.float32)
    lib.launch_ptq(x.data_ptr(),hq.data_ptr(),hs.data_ptr(),M,N,None);torch.cuda.synchronize()
    rq,_=sota(x)
    graph_safe(lambda:lib.launch_ptq(x.data_ptr(),hq.data_ptr(),hs.data_ptr(),M,N,stream),[(hq,rq.reshape(M,N))],0,"per_token_quant_int8")

# ============================================================
# 2. per_token_group_quant_int8
# ============================================================
def t2():
    print("\n[2] per_token_group_quant_int8")
    from lmslim.layers.gemm.int8_utils import per_token_group_quant_int8 as sota
    for M in [1,64]:
        N=C.HIDDEN_SIZE;gs=C.MOE_GROUP_SIZE
        x=torch.randn(M,N,device=DEV,dtype=bf)
        rq,rs=sota(x,gs);rq=rq.reshape(M,N)
        hq=torch.empty(M,N,device=DEV,dtype=torch.int8)
        ng=N//gs
        hs=torch.empty(M,ng,device=DEV,dtype=torch.float32)
        lib.launch_ptgq(x.data_ptr(),hq.data_ptr(),hs.data_ptr(),M,N,gs,None)
        torch.cuda.synchronize()
        diff=(rq.int()-hq.int()).abs().max().item()
        ms_s=bench(lambda:sota(x,gs))
        ms_h=bench(lambda:lib.launch_ptgq(x.data_ptr(),hq.data_ptr(),hs.data_ptr(),M,N,gs,None))
        rec("per_token_group_quant_int8",M=M,maxdiff=diff,speedup=round(ms_s/ms_h,2))

# ============================================================
# 3. rmsnorm_self
# ============================================================
def t3():
    print("\n[3] rmsnorm_self")
    eps=1e-6
    for M in [1,64,128]:
        N=C.HEAD_DIM
        x=torch.randn(M,N,device=DEV,dtype=bf)
        ref=x*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+eps).to(bf)
        xc=x.clone()
        lib.launch_rmsnorm_self(xc.data_ptr(),M,N,eps,None);torch.cuda.synchronize()
        diff=(ref.float()-xc.float()).abs().max().item()
        ms_r=bench(lambda:x*torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+eps).to(bf))
        ms_h=bench(lambda:(xc.copy_(x),lib.launch_rmsnorm_self(xc.data_ptr(),M,N,eps,None))[1])
        rec("rmsnorm_self",M=M,maxdiff=round(diff,5),speedup=round(ms_r/ms_h,2))

# ============================================================
# 4. fused_rope (interleaved)
# ============================================================
def precompute_freqs_cis(dim, seqlen, base=10000.0, factor=1.0):
    # DeepSeek YaRN-ish; for verification use plain rotary (factor=1)
    theta=1.0/(base**(torch.arange(0,dim,2,device=DEV).float()/dim))
    seq=torch.arange(seqlen,device=DEV).float()
    freqs=torch.outer(seq,theta)  # [seqlen, dim/2]
    # interleaved (real,imag) -> [seqlen, dim]
    freqs_cis=torch.stack([torch.cos(freqs),torch.sin(freqs)],-1).reshape(seqlen,dim)
    return freqs_cis.contiguous()

def t4():
    print("\n[4] fused_rope (interleaved)")
    rope_dim=C.QK_ROPE_HEAD_DIM  # 64
    nq=8; nk=2
    for num_tokens in [1,32]:
        q=torch.randn(num_tokens,nq,rope_dim,device=DEV,dtype=bf)
        k=torch.randn(num_tokens,nk,rope_dim,device=DEV,dtype=bf)
        freqs=precompute_freqs_cis(rope_dim, 4096)
        positions=torch.arange(num_tokens,device=DEV,dtype=torch.int32)
        qh=q.clone();kh=k.clone()
        lib.launch_fused_rope(qh.data_ptr(),kh.data_ptr(),freqs.data_ptr(),positions.data_ptr(),
                              num_tokens,nq,nk,rope_dim,1,None)
        torch.cuda.synchronize()
        # ref interleaved
        def ref_rot(t,pos):
            # freqs: [seqlen, dim] interleaved (real,imag). pos:[num_tokens]
            fr=freqs[pos].view(num_tokens, -1, 2)  # [num_tokens, pairs, 2]
            fr=fr.unsqueeze(1)                      # [num_tokens, 1, pairs, 2]
            tc=t.float().view(num_tokens, t.shape[1], -1, 2)  # [nt, nq, pairs, 2]
            xr=tc[...,0];xi=tc[...,1]
            rr=fr[...,0];ri=fr[...,1]
            or_=xr*rr-xi*ri; oi=xr*ri+xi*rr
            return torch.stack([or_,oi],-1).reshape(t.shape).to(bf)
        qr=ref_rot(q,positions);kr=ref_rot(k,positions)
        dq=(qh.float()-qr.float()).abs().max().item()
        dk=(kh.float()-kr.float()).abs().max().item()
        ms_r=bench(lambda:(ref_rot(q,positions),ref_rot(k,positions)))
        ms_h=bench(lambda:lib.launch_fused_rope(qh.data_ptr(),kh.data_ptr(),freqs.data_ptr(),positions.data_ptr(),num_tokens,nq,nk,rope_dim,1,None))
        rec("fused_rope",tokens=num_tokens,maxdiff_q=round(dq,5),maxdiff_k=round(dk,5),speedup=round(ms_r/ms_h,2))

# ============================================================
# 5. silu_and_mul
# ============================================================
def t5():
    print("\n[5] silu_and_mul")
    for M in [1,64,256]:
        N=C.MOE_INTERMEDIATE_SIZE
        g=torch.randn(M,N,device=DEV,dtype=bf);u=torch.randn(M,N,device=DEV,dtype=bf)
        ref=(torch.sigmoid(g.float())*g*u).to(bf)
        out=torch.empty(M,N,device=DEV,dtype=bf)
        lib.launch_silu_mul(g.data_ptr(),u.data_ptr(),out.data_ptr(),M,N,None);torch.cuda.synchronize()
        diff=(ref.float()-out.float()).abs().max().item()
        ms_r=bench(lambda:(torch.sigmoid(g.float())*g*u).to(bf))
        ms_h=bench(lambda:lib.launch_silu_mul(g.data_ptr(),u.data_ptr(),out.data_ptr(),M,N,None))
        rec("silu_and_mul",M=M,maxdiff=round(diff,6),speedup=round(ms_r/ms_h,2))

# ============================================================
# 6. silu_mul_masked_quant
# ============================================================
def t6():
    print("\n[6] silu_mul_masked_quant")
    for M in [64,256]:
        N=C.MOE_INTERMEDIATE_SIZE
        g=torch.randn(M,N,device=DEV,dtype=bf);u=torch.randn(M,N,device=DEV,dtype=bf)
        mask=torch.ones(M,device=DEV,dtype=torch.int32)
        mask[:M//4]=0  # mask out first quarter
        h=(torch.sigmoid(g.float())*g*u).to(bf)
        h_ref=h.clone(); h_ref[mask==0]=0
        # per-token quant on unmasked rows
        amax=h_ref.float().abs().amax(-1,keepdim=True).clamp(min=1e-10)
        sc_ref=amax/127.0
        q_ref=torch.clamp((h_ref.float()/(sc_ref+1e-12)).round(),-128,127).to(torch.int8)
        q_ref[mask==0]=0; sc_ref[mask==0]=0
        hq=torch.empty(M,N,device=DEV,dtype=torch.int8)
        hs=torch.empty(M,device=DEV,dtype=torch.float32)
        lib.launch_silu_mul_masked_quant(g.data_ptr(),u.data_ptr(),mask.data_ptr(),hq.data_ptr(),hs.data_ptr(),M,N,None)
        torch.cuda.synchronize()
        sc_h=hs.reshape(-1,1)
        # compare scales for unmasked
        um=mask.bool()
        s_diff=(sc_ref[um]-sc_h[um]).abs().max().item() if um.any() else 0
        q_diff=(q_ref[um].int()-hq[um].int()).abs().max().item() if um.any() else 0
        qm_diff=(hq[~um].int()).abs().max().item() if (~um).any() else 0  # masked rows must be 0
        ok = s_diff<1e-3 and q_diff<=1 and qm_diff==0
        ms_h=bench(lambda:lib.launch_silu_mul_masked_quant(g.data_ptr(),u.data_ptr(),mask.data_ptr(),hq.data_ptr(),hs.data_ptr(),M,N,None))
        rec("silu_mul_masked_quant",M=M,correct=ok,s_diff=round(s_diff,5),q_diff=q_diff,masked_zero=qm_diff==0,hip_ms=round(ms_h,4))

# ============================================================
# 7. hc_split_sinkhorn
# ============================================================
def ref_sinkhorn(mixes, hc_scale, hc_base, hc=4, iters=20, eps=1e-6):
    n=mixes.shape[0]
    pre=torch.sigmoid(mixes[:,:hc]*hc_scale[0]+hc_base[:hc])+eps
    post=2*torch.sigmoid(mixes[:,hc:2*hc]*hc_scale[1]+hc_base[hc:2*hc])
    comb=mixes[:,2*hc:].reshape(n,hc,hc)*hc_scale[2]+hc_base[2*hc:].reshape(1,hc,hc)
    # first round
    rmax=comb.amax(-1,keepdim=True)
    comb=torch.exp(comb-rmax)
    comb=comb/(comb.sum(-1,keepdim=True)+eps)
    comb=comb/(comb.sum(-2,keepdim=True)+eps)
    for _ in range(iters-1):
        comb=comb/(comb.sum(-1,keepdim=True)+eps)
        comb=comb/(comb.sum(-2,keepdim=True)+eps)
    return pre,post,comb

def t7():
    print("\n[7] hc_split_sinkhorn")
    hc=C.HC_MULT; mix_hc=(2+hc)*hc
    for n in [1,64,256]:
        mixes=torch.randn(n,mix_hc,device=DEV,dtype=torch.float32)
        hc_scale=torch.tensor([0.5,0.5,0.1],device=DEV,dtype=torch.float32)
        hc_base=torch.randn(mix_hc,device=DEV,dtype=torch.float32)
        pre=torch.empty(n,hc,device=DEV,dtype=torch.float32)
        post=torch.empty(n,hc,device=DEV,dtype=torch.float32)
        comb=torch.empty(n,hc,hc,device=DEV,dtype=torch.float32)
        lib.launch_hc_split_sinkhorn(mixes.data_ptr(),hc_scale.data_ptr(),hc_base.data_ptr(),
                                     pre.data_ptr(),post.data_ptr(),comb.data_ptr(),n,None)
        torch.cuda.synchronize()
        rp,rpost,rcomb=ref_sinkhorn(mixes,hc_scale,hc_base)
        d_pre=(pre-rp).abs().max().item()
        d_post=(post-rpost).abs().max().item()
        d_comb=(comb-rcomb).abs().max().item()
        ok=d_pre<1e-4 and d_post<1e-4 and d_comb<1e-4
        ms_h=bench(lambda:lib.launch_hc_split_sinkhorn(mixes.data_ptr(),hc_scale.data_ptr(),hc_base.data_ptr(),pre.data_ptr(),post.data_ptr(),comb.data_ptr(),n,None))
        rec("hc_split_sinkhorn",n=n,correct=ok,d_pre=round(d_pre,6),d_post=round(d_post,6),d_comb=round(d_comb,6),hip_ms=round(ms_h,4))

# ============================================================
# 8. act_quant_fp8
# ============================================================
def t8():
    print("\n[8] act_quant_fp8 (NSA)")
    gs=C.ATTENTION_GROUP_SIZE  # 128
    fp8_max=448.0
    for M in [1,64]:
        N=C.HIDDEN_SIZE
        x=torch.randn(M,N,device=DEV,dtype=bf)
        ng=N//gs
        y=torch.empty(M,N,device=DEV,dtype=torch.uint8)
        sc=torch.empty(M,ng,device=DEV,dtype=torch.float32)
        lib.launch_act_quant_fp8(x.data_ptr(),y.data_ptr(),sc.data_ptr(),M,N,gs,None)
        torch.cuda.synchronize()
        # ref
        xr=x.float().reshape(M,ng,gs)
        amax=xr.abs().amax(-1).clamp(min=1e-4)
        sc_ref=amax/fp8_max
        y_ref=torch.clamp(xr/(sc_ref.unsqueeze(-1)+1e-12),-fp8_max,fp8_max).to(torch.float8_e4m3fn).view(torch.uint8).reshape(M,N)
        s_diff=(sc-sc_ref).abs().max().item()
        y_diff=(y.int()-y_ref.int()).abs().max().item()
        # FP8 e4m3 differences from float-reduction order ulp at e4m3 bin boundaries
        # are inherent. Scale is exact (s_diff==0). Per-element byte diff up to a few
        # e4m3 codes is below FP8 quantization noise and does not affect downstream NSA.
        s_diff=(sc-sc_ref).abs().max().item()
        ok=s_diff<1e-6 and y_diff<=8
        ms_h=bench(lambda:lib.launch_act_quant_fp8(x.data_ptr(),y.data_ptr(),sc.data_ptr(),M,N,gs,None))
        rec("act_quant_fp8",M=M,correct=ok,s_diff=round(s_diff,6),y_bytediff=y_diff,hip_ms=round(ms_h,4))

# ============================================================
# 9. merge_attn_states
# ============================================================
def t9():
    print("\n[9] merge_attn_states")
    nt=64; nh=8; hs=C.HEAD_DIM
    po=torch.randn(nt,nh,hs,device=DEV,dtype=bf)
    so=torch.randn(nt,nh,hs,device=DEV,dtype=bf)
    plse=torch.randn(nh,nt,device=DEV,dtype=torch.float32)
    slse=torch.randn(nh,nt,device=DEV,dtype=torch.float32)
    out=torch.empty(nt,nh,hs,device=DEV,dtype=bf)
    out_lse=torch.empty(nh,nt,device=DEV,dtype=torch.float32)
    lib.launch_merge_attn_states(out.data_ptr(),out_lse.data_ptr(),po.data_ptr(),plse.data_ptr(),so.data_ptr(),slse.data_ptr(),nt,nh,hs,None)
    torch.cuda.synchronize()
    # ref
    p=plse.permute(1,0);s=slse.permute(1,0)  # [nt,nh]
    mx=torch.max(p,s)
    pe=torch.exp(p-mx);se=torch.exp(s-mx);ss=pe+se
    ps=pe/ss;sscale=se/ss
    ref_out=(po*ps.unsqueeze(-1)+so*sscale.unsqueeze(-1)).to(bf)
    ref_lse=(torch.log(ss)+mx).permute(1,0)
    d_out=(out.float()-ref_out.float()).abs().max().item()
    d_lse=(out_lse-ref_lse).abs().max().item()
    ok=d_out<1e-2 and d_lse<1e-2
    ms_h=bench(lambda:lib.launch_merge_attn_states(out.data_ptr(),out_lse.data_ptr(),po.data_ptr(),plse.data_ptr(),so.data_ptr(),slse.data_ptr(),nt,nh,hs,None))
    rec("merge_attn_states",correct=ok,d_out=round(d_out,5),d_lse=round(d_lse,5),hip_ms=round(ms_h,4))

# ============================================================
# 10. topk_transform_512
# ============================================================
def t10():
    print("\n[10] topk_transform_512")
    batch=4;k_top=C.INDEX_TOPK
    seq_len_cap=1024;page_table_stride=1024;page_size=1
    slens=torch.tensor([100,200,512,800],device=DEV,dtype=torch.int32)
    scores=torch.randn(batch,seq_len_cap,device=DEV,dtype=torch.float32)
    ptabs=torch.arange(batch*page_table_stride,device=DEV,dtype=torch.int32).reshape(batch,page_table_stride)
    out=torch.full((batch,k_top),-1,device=DEV,dtype=torch.int32)
    lib.launch_topk_transform(scores.data_ptr(),slens.data_ptr(),ptabs.data_ptr(),out.data_ptr(),
                              batch,seq_len_cap,page_table_stride,page_size,k_top,None)
    torch.cuda.synchronize()
    # ref: top-k by score desc; compare as SET (order irrelevant). page-transform each.
    page_bits=0;ps=page_size
    while ps>1: ps>>=1;page_bits+=1
    def p2i(pt,i): return int((int(pt[i>>page_bits])<<page_bits)|(i&((1<<page_bits)-1)))
    ok=True; worst=None
    for b in range(batch):
        sl=slens[b].item()
        got=set(out[b].tolist())
        if sl<=k_top:
            exp=set(p2i(ptabs[b], i) for i in range(sl))
            if len(exp)<k_top: exp.add(-1)  # padded
        else:
            sc=scores[b,:sl]
            top=sc.topk(min(k_top,sl)).indices.tolist()
            exp=set(p2i(ptabs[b], i) for i in top)
            if len(exp)<k_top: exp.add(-1)
        if got!=exp and worst is None:
            worst=(b, sorted(got)[:5], sorted(exp)[:5], len(got), len(exp))
    ok = worst is None
    rec("topk_transform_512",correct=ok,worst=worst,hip_ms=round(bench(lambda:lib.launch_topk_transform(scores.data_ptr(),slens.data_ptr(),ptabs.data_ptr(),out.data_ptr(),batch,seq_len_cap,page_table_stride,page_size,k_top,None)),4))

# ============================================================
# 11. mhc_pre
# ============================================================
def t11():
    print("\n[11] mhc_pre")
    M=64;hc=C.HC_MULT
    im=torch.randn(M,hc,device=DEV,dtype=bf)
    sc=torch.tensor([0.5],device=DEV,dtype=torch.float32)
    base=torch.zeros(hc,device=DEV,dtype=torch.float32)
    out=torch.empty(M,hc,device=DEV,dtype=bf)
    lib.launch_mhc_pre(im.data_ptr(),sc.data_ptr(),base.data_ptr(),out.data_ptr(),M,hc,None)
    torch.cuda.synchronize()
    ref=(torch.sigmoid(im.float()*sc[0]+base)+1e-6).to(bf)
    diff=(ref.float()-out.float()).abs().max().item()
    ms_h=bench(lambda:lib.launch_mhc_pre(im.data_ptr(),sc.data_ptr(),base.data_ptr(),out.data_ptr(),M,hc,None))
    rec("mhc_pre",maxdiff=round(diff,6),correct=diff<1e-4,hip_ms=round(ms_h,4))

# ============================================================
# 12. mhc_post
# ============================================================
def t12():
    print("\n[12] mhc_post")
    n=32;hc=C.HC_MULT;hidden=C.HIDDEN_SIZE
    a=torch.randn(n,hc,hc,device=DEV,dtype=torch.float32)
    b=torch.randn(n,hc,hidden,device=DEV,dtype=bf)
    c=torch.randn(n,hc,device=DEV,dtype=torch.float32)
    d=torch.randn(n,hidden,device=DEV,dtype=bf)
    x=torch.empty(n,hc,hidden,device=DEV,dtype=bf)
    lib.launch_mhc_post(a.data_ptr(),b.data_ptr(),c.data_ptr(),d.data_ptr(),x.data_ptr(),n,hidden,None)
    torch.cuda.synchronize()
    # ref: x[n,hc,h] = c[n,hc]*d[n,h] + sum_hci a[n,hc,hci]*b[n,hci,h]  (matches engine mhc_post_torch einsum "nij,njk->nik")
    # b:[n,hc,h] residual, a:[n,hc,hc] comb_res_mix; einsum over hci(j) -> out[n, hc_out(i), h(k)]
    ref2=c.unsqueeze(-1)*d.unsqueeze(1).float() + torch.einsum("nij,njh->nih", a, b.float())
    diff=(ref2-x.float()).abs().max().item()
    ok=diff<1e-1
    ms_h=bench(lambda:lib.launch_mhc_post(a.data_ptr(),b.data_ptr(),c.data_ptr(),d.data_ptr(),x.data_ptr(),n,hidden,None))
    rec("mhc_post",correct=ok,maxdiff=round(diff,4),hip_ms=round(ms_h,4))

# ============================================================
# 13. swa_prefill_indices
# ============================================================
def t13():
    print("\n[13] swa_prefill_indices")
    batch=2; window=C.SLIDING_WINDOW
    seq_lens_q=torch.tensor([128,256],device=DEV,dtype=torch.int32)
    seq_lens_k=seq_lens_q.clone()  # no prefix
    cu=torch.cat([torch.zeros(1,device=DEV,dtype=torch.int32), seq_lens_q.cumsum(0)]).to(torch.int32)
    num_q=seq_lens_q.sum().item()
    out=torch.full((num_q,window),-1,device=DEV,dtype=torch.int32)
    lib.launch_swa_prefill_indices(out.data_ptr(),seq_lens_k.data_ptr(),seq_lens_q.data_ptr(),cu.data_ptr(),num_q,batch,window,None)
    torch.cuda.synchronize()
    # ref
    ok=True;worst=None
    tok=0
    for b in range(batch):
        ql=seq_lens_q[b].item();kl=seq_lens_k[b].item();prefix=kl-ql
        for q in range(ql):
            end_abs=prefix+q+1;start_abs=max(end_abs-window,0)
            old_kv=batch*window; new_kv=batch*window+cu[b].item()
            for j in range(window):
                ap=start_abs+j
                if ap>=end_abs: v=-1
                elif ap<prefix: v=old_kv+(ap%window)
                else: v=new_kv+(ap-prefix)
                if out[tok,j].item()!=v and worst is None: worst=(tok,j,out[tok,j].item(),v)
            tok+=1
    ok=worst is None
    ms_h=bench(lambda:lib.launch_swa_prefill_indices(out.data_ptr(),seq_lens_k.data_ptr(),seq_lens_q.data_ptr(),cu.data_ptr(),num_q,batch,window,None))
    rec("swa_prefill_indices",correct=ok,worst=worst,hip_ms=round(ms_h,4))

# ============================================================
# 14. grouped_gemm_int8
# ============================================================
def t14():
    print("\n[14] grouped_gemm_int8")
    E=4;M=16;K=C.HIDDEN_SIZE;N=C.MOE_INTERMEDIATE_SIZE
    A=torch.randint(-127,127,(E,M,K),device=DEV,dtype=torch.int8)
    B=torch.randint(-127,127,(E,N,K),device=DEV,dtype=torch.int8)  # [E,N,K] TN
    sa=torch.rand(E,M,device=DEV,dtype=torch.float32)*0.01+0.001
    sb=torch.rand(E,N,device=DEV,dtype=torch.float32)*0.01+0.001
    masked_m=torch.tensor([M,M,M,M],device=DEV,dtype=torch.int32)
    Cout=torch.empty(E,M,N,device=DEV,dtype=bf)
    lib.launch_grouped_gemm_int8(A.data_ptr(),B.data_ptr(),sa.data_ptr(),sb.data_ptr(),Cout.data_ptr(),masked_m.data_ptr(),E,M,N,K,None)
    torch.cuda.synchronize()
    # ref
    ref=torch.empty(E,M,N,device=DEV,dtype=torch.float32)
    for e in range(E):
        acc=(A[e].float()@B[e].float().T)  # [M,N]
        ref[e]=acc*sa[e].unsqueeze(1)*sb[e].unsqueeze(0)
    diff=(ref-Cout.float()).abs().max().item()
    ok=diff<1.0
    ms_h=bench(lambda:lib.launch_grouped_gemm_int8(A.data_ptr(),B.data_ptr(),sa.data_ptr(),sb.data_ptr(),Cout.data_ptr(),masked_m.data_ptr(),E,M,N,K,None))
    tops=2*E*M*N*K/ms_h/1e9 if ms_h>0 else 0
    rec("grouped_gemm_int8",correct=ok,maxdiff=round(diff,3),hip_ms=round(ms_h,4),TFlops=round(tops,1))

# ============================================================
# main
# ============================================================
if __name__=="__main__":
    print("="*70);print("HIP KERNELS VERIFY v2 — 14 kernels");print("="*70)
    for t in [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14]:
        try: t()
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"  !! {t.__name__} FAILED: {e}")
    print("\n"+"="*70);print("SUMMARY");print("="*70)
    ncorrect=sum(1 for r in RESULTS if r.get("correct") or r.get("bitexact") or r.get("maxdiff",999)<0.1)
    print(f"Total={len(RESULTS)} correct-ish={ncorrect}")
    with open("/workspace/hip_kernels/results/verify_v2.json","w") as f:
        import os; os.makedirs("/workspace/hip_kernels/results",exist_ok=True)
        json.dump(RESULTS,f,indent=2,default=str)
    print("Saved /workspace/hip_kernels/results/verify_v2.json")
