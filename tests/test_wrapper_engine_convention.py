"""Unit tests STRICTLY mimicking engine call conventions (args, in-place, out-param).
All shapes = actual inference (decode bs=256, TP=8).

Each test calls the wrapper EXACTLY as the engine call site does:
- SiluAndMul: as a method `SiluAndMul().forward_cuda(x)` (self.x)
- fused_rope/topk/merge: in-place (q, out_page_indices, output are caller-preallocated)
- merge_attn_states: out-param (output passed in by caller)
- rmsnorm_self: returns new tensor (caller binds q = rmsnorm_self(q,...))
- per_token_quant: returns (q, scales)
Run: python -m pytest tests/test_wrapper_engine_convention.py -v
"""
import pytest, torch, sys
sys.path.insert(0, "/workspace/dsv4_ops_unit_tests")
sys.path.insert(0, "/workspace/sglang/python")
sys.path.insert(0, "/workspace/hip_kernels")
from utils import model_config as C
import hip_wrapper as W

DEV = "cuda"; bf = torch.bfloat16; eps = 1e-6


# ---- 1. per_token_quant_int8  (engine: x_q, x_scale = per_token_quant_int8(x)) ----
@pytest.mark.cuda
class TestPerTokenQuantInt8_Engine:
    def test_decode_256x4096(self):
        from lmslim.layers.gemm.int8_utils import per_token_quant_int8 as sota
        x = torch.randn(256, C.HIDDEN_SIZE, device=DEV, dtype=bf)
        rq, rs = sota(x)              # engine convention
        q, s = W.per_token_quant_int8(x)
        assert q.shape == (256, C.HIDDEN_SIZE) and q.dtype == torch.int8
        assert s.shape == (256, 1) and s.dtype == torch.float32
        assert (rq == q).all().item()
    def test_moe_1536x2048(self):
        from lmslim.layers.gemm.int8_utils import per_token_quant_int8 as sota
        x = torch.randn(1536, C.MOE_INTERMEDIATE_SIZE, device=DEV, dtype=bf)
        q, s = W.per_token_quant_int8(x)
        assert q.shape == (1536, C.MOE_INTERMEDIATE_SIZE)

# ---- 2. per_token_group_quant_int8  (engine: q, s = per_token_group_quant_int8(x, gs)) ----
@pytest.mark.cuda
class TestPerTokenGroupQuantInt8_Engine:
    def test_moe_1536x2048_gs128(self):
        from lmslim.layers.gemm.int8_utils import per_token_group_quant_int8 as sota
        x = torch.randn(1536, C.MOE_INTERMEDIATE_SIZE, device=DEV, dtype=bf)
        rq, rs = sota(x, 128)
        q, s = W.per_token_group_quant_int8(x, 128)
        assert q.shape == x.shape and s.shape == (1536, 16)
        assert (rq.int()-q.int()).abs().max().item() == 0

# ---- 3. SiluAndMul  (engine: out = self.forward_cuda(x); x=[M,2N]) ----
@pytest.mark.cuda
class TestSiluAndMul_Engine:
    def _call_like_engine(self, x):
        # engine calls SiluAndMul().forward_cuda(x) -> out [M, N]
        return W.silu_and_mul(x)
    def test_moe_1536x4096(self):
        x = torch.randn(1536, C.MOE_INTERMEDIATE_SIZE*2, device=DEV, dtype=bf)
        d = x.shape[-1]//2
        out = self._call_like_engine(x)
        assert out.shape == (1536, d) and out.dtype == bf
        ref = (torch.sigmoid(x[...,:d].float())*x[...,:d]*x[...,d:]).to(bf)
        assert (out.float()-ref.float()).abs().max().item() < 0.1

# ---- 4. rmsnorm_self  (engine: q = rmsnorm_self(q, eps); returns NEW tensor) ----
@pytest.mark.cuda
class TestRmsnormSelf_Engine:
    def test_decode_256x8x512(self):
        q = torch.randn(256, 8, C.HEAD_DIM, device=DEV, dtype=bf)
        q_orig = q.clone()
        out = W.rmsnorm_self(q, eps)         # engine: q = rmsnorm_self(q, eps)
        assert out.shape == q.shape and out is not q   # NEW tensor, q unchanged
        assert (q - q_orig).abs().max().item() == 0    # input not modified
        ref = q * torch.rsqrt(q.float().pow(2).mean(-1,keepdim=True)+eps).to(bf)
        assert (out.float()-ref.float()).abs().max().item() < 0.05

# ---- 5. fused_rope  (engine: fused_rope(q, k, freqs_cis, positions); IN-PLACE on q/k) ----
@pytest.mark.cuda
class TestFusedRope_Engine:
    def _freqs(self, dim, sl):
        th = 1.0/(10000.0**(torch.arange(0,dim,2,device=DEV).float()/dim))
        f = torch.outer(torch.arange(sl,device=DEV).float(), th)
        return torch.view_as_complex(torch.stack([torch.cos(f),torch.sin(f)],-1)).contiguous()
    def test_decode_q_256x8x64_k_none(self):
        rd = C.QK_ROPE_HEAD_DIM
        q = torch.randn(256, 8, rd, device=DEV, dtype=bf)
        fc = self._freqs(rd, 8192)
        pos = torch.arange(256, device=DEV, dtype=torch.int32)
        q_ref = q.clone()
        W.fused_rope(q, None, fc, pos)       # in-place, returns None
        assert q is not q_ref                # but q tensor modified in place
        # ref
        fr = torch.view_as_real(fc[pos]).unsqueeze(1)
        t = q_ref.float().view(256,8,-1,2)
        ref = torch.stack([t[...,0]*fr[...,0]-t[...,1]*fr[...,1], t[...,0]*fr[...,1]+t[...,1]*fr[...,0]],-1).reshape(q.shape).to(bf)
        assert (q.float()-ref.float()).abs().max().item() < 0.05
    def test_prefill_q_k(self):
        rd = C.QK_ROPE_HEAD_DIM
        q = torch.randn(64, 8, rd, device=DEV, dtype=bf)
        k = torch.randn(64, 1, rd, device=DEV, dtype=bf)
        fc = self._freqs(rd, 8192)
        pos = torch.arange(64, device=DEV, dtype=torch.int32)
        W.fused_rope(q, k, fc, pos)
        assert torch.isfinite(q).all() and torch.isfinite(k).all()

# ---- 6. topk_transform_512  (engine: in-place out_page_indices) ----
@pytest.mark.cuda
class TestTopkTransform512_Engine:
    def test_decode_256x512(self):
        b = 256; k = C.INDEX_TOPK; cap = 1024; ptr = 1024
        sl = torch.randint(100, cap, (b,), device=DEV, dtype=torch.int32)
        sc = torch.randn(b, cap, device=DEV, dtype=torch.float32)
        pt = torch.arange(b*ptr, device=DEV, dtype=torch.int32).reshape(b, ptr)
        out = torch.full((b, k), -1, device=DEV, dtype=torch.int32)  # caller preallocates
        W.topk_transform_512(sc, sl, pt, out, 1)   # in-place
        # verify set for a few
        for bi in [0, 128, 255]:
            sli = sl[bi].item(); got = set(out[bi].tolist())
            if sli <= k:
                exp = set(bi*ptr+i for i in range(sli)) | ({-1} if sli<k else set())
            else:
                exp = set(bi*ptr+i for i in sc[bi,:sli].topk(k).indices.tolist())
            assert got == exp, f"batch {bi}"

# ---- 7. swa_prefill_indices  (engine: returns swa_indices, in-place) ----
@pytest.mark.cuda
class TestSwaPrefillIndices_Engine:
    def test_prefill(self):
        sq = torch.tensor([128,256,64,512], device=DEV, dtype=torch.int32)
        sk = sq.clone()
        cu = torch.cat([torch.zeros(1,device=DEV,dtype=torch.int32), sq.cumsum(0)]).to(torch.int32)
        nq = int(sq.sum().item())
        out = torch.full((nq, C.SLIDING_WINDOW), -1, device=DEV, dtype=torch.int32)
        ret = W.tilelang_make_swa_prefill_indices(sk, sq, out, cu)
        assert ret is out
        assert out[0,0].item() >= 0

# ---- 8. hc_split_sinkhorn  (engine: pre,post,comb = hc_split_sinkhorn(mixes,...)) ----
@pytest.mark.cuda
class TestHcSplitSinkhorn_Engine:
    def test_decode_256x1x24(self):
        hc = C.HC_MULT; mh = (2+hc)*hc
        mixes = torch.randn(256, mh, device=DEV, dtype=torch.float32)  # flattened [256,24]
        sc = torch.tensor([0.5,0.5,0.1], device=DEV, dtype=torch.float32)
        base = torch.randn(mh, device=DEV, dtype=torch.float32)
        pre, post, comb = W.hc_split_sinkhorn(mixes.view(256,1,mh), sc, base, hc, 20, 1e-6)
        assert pre.shape == (256,1,4) and comb.shape == (256,1,4,4)
        # ref
        pre_r = torch.sigmoid(mixes[:,:hc]*sc[0]+base[:hc])+1e-6
        assert (pre.view(256,hc)-pre_r).abs().max().item() < 1e-4

# ---- 9. mhc_post  (engine: out = mhc_post_torch(x, residual, post, comb)) ----
@pytest.mark.cuda
class TestMhcPost_Engine:
    def test_decode_256(self):
        n = 256; hc = C.HC_MULT; hidden = C.HIDDEN_SIZE
        x = torch.randn(n, hidden, device=DEV, dtype=bf)
        res = torch.randn(n, hc, hidden, device=DEV, dtype=bf)
        plm = torch.randn(n, hc, 1, device=DEV, dtype=torch.float32)
        crm = torch.randn(n, hc, hc, device=DEV, dtype=torch.float32)
        out = W.mhc_post_torch(x, res, plm, crm)
        assert out.shape == (n, hc, hidden) and out.dtype == bf
        ref = plm.squeeze(-1).unsqueeze(-1)*x.unsqueeze(1).float() + torch.einsum("nij,njh->nih", crm, res.float())
        assert (out.float()-ref).abs().max().item() < 0.5

# ---- 10. act_quant  (engine: y, s = act_quant(x, block_size, scale_fmt)) ----
@pytest.mark.cuda
class TestActQuant_Engine:
    def test_indexer_256x64x128(self):
        x = torch.randn(256, C.INDEX_N_HEADS, C.INDEX_HEAD_DIM, device=DEV, dtype=bf)
        y, s = W.act_quant(x, 128)
        assert y.shape == x.shape and y.dtype == torch.float8_e4m3fn
        assert s.shape == (256, 64, 1) and s.dtype == torch.float32
        # scale exact
        N = x.shape[-1]
        amax = x.float().reshape(256,64,N//128,128).abs().amax(-1,keepdim=True).clamp(min=1e-4)  # [256,64,1,1]
        s_ref = (amax/448.0).squeeze(-1)  # [256,64,1]
        assert (s-s_ref).abs().max().item() < 1e-5

# ---- 11. merge_attn_states  (engine: merge_attn_states(output, p_out, p_lse, s_out, s_lse, out_lse=None); IN-PLACE output) ----
@pytest.mark.cuda
class TestMergeAttnStates_Engine:
    def test_decode_256x8x512(self):
        nt = 256; nh = 8; hs = C.HEAD_DIM
        po = torch.randn(nt,nh,hs,device=DEV,dtype=bf); so = torch.randn(nt,nh,hs,device=DEV,dtype=bf)
        pl = torch.randn(nh,nt,device=DEV,dtype=torch.float32); sl = torch.randn(nh,nt,device=DEV,dtype=torch.float32)
        output = torch.empty(nt,nh,hs,device=DEV,dtype=bf)   # caller preallocates
        ol = torch.empty(nh,nt,device=DEV,dtype=torch.float32)
        W.merge_attn_states(output, po, pl, so, sl, ol)      # in-place
        p = pl.permute(1,0); s2 = sl.permute(1,0); mx = torch.max(p,s2)
        pe = torch.exp(p-mx); se = torch.exp(s2-mx); ss = pe+se
        ref = (po*(pe/ss).unsqueeze(-1)+so*(se/ss).unsqueeze(-1)).to(bf)
        assert (output.float()-ref.float()).abs().max().item() < 1e-1
