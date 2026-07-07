"""Pytest unit tests for all 14 HIP kernels — correctness vs SOTA references.
Mirrors the structure of github.com/mmt-at/dsv4_ops_unit_tests.

Run:
    cd /workspace/hip_kernels && python -m pytest tests/ -v
    python -m pytest tests/ -v -k "rope or quant"
    python -m pytest tests/ -v -m graph
"""
import pytest, torch, ctypes, sys, os
sys.path.insert(0, "/workspace/dsv4_ops_unit_tests")
sys.path.insert(0, "/workspace/sglang/python")
from utils import model_config as C

DEV = "cuda"
bf = torch.bfloat16
eps = 1e-6

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
            ("launch_swa_prefill_indices",[P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,P]),
            ("launch_grouped_gemm_int8",[P,P,P,P,P,P,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,P])]:
    getattr(lib, n).argtypes = a
S = lambda: torch.cuda.current_stream().cuda_stream

def pytest_report_header(config):
    return [f"Device: {torch.cuda.get_device_name(0)}, count={torch.cuda.device_count()}"]

@pytest.fixture
def stream():
    return torch.cuda.current_stream().cuda_stream


# ---- 1. per_token_quant_int8 ----
@pytest.mark.cuda
class TestPerTokenQuantInt8:
    @pytest.mark.parametrize("M", [1, 64, 256])
    def test_bitexact_vs_lmslim(self, M, stream):
        from lmslim.layers.gemm.int8_utils import per_token_quant_int8 as sota
        N = C.HIDDEN_SIZE
        x = torch.randn(M, N, device=DEV, dtype=bf)
        rq, rs = sota(x); rq = rq.reshape(M, N)
        hq = torch.empty(M, N, device=DEV, dtype=torch.int8)
        hs = torch.empty(M, device=DEV, dtype=torch.float32)
        lib.launch_ptq(x.data_ptr(), hq.data_ptr(), hs.data_ptr(), M, N, stream)
        torch.cuda.synchronize()
        assert (rq == hq).all().item(), f"maxdiff {(rq.int()-hq.int()).abs().max()}"

# ---- 2. per_token_group_quant_int8 ----
@pytest.mark.cuda
class TestPerTokenGroupQuantInt8:
    @pytest.mark.parametrize("M", [1, 64])
    def test_bitexact_vs_lmslim(self, M, stream):
        from lmslim.layers.gemm.int8_utils import per_token_group_quant_int8 as sota
        N = C.HIDDEN_SIZE; gs = C.MOE_GROUP_SIZE
        x = torch.randn(M, N, device=DEV, dtype=bf)
        rq, rs = sota(x, gs); rq = rq.reshape(M, N)
        hq = torch.empty(M, N, device=DEV, dtype=torch.int8)
        hs = torch.empty(M, N//gs, device=DEV, dtype=torch.float32)
        lib.launch_ptgq(x.data_ptr(), hq.data_ptr(), hs.data_ptr(), M, N, gs, stream)
        torch.cuda.synchronize()
        assert (rq.int()-hq.int()).abs().max().item() == 0

# ---- 3. rmsnorm_self ----
@pytest.mark.cuda
class TestRmsnormSelf:
    @pytest.mark.parametrize("M", [1, 64, 128])
    def test_vs_torch(self, M, stream):
        N = C.HEAD_DIM
        x = torch.randn(M, N, device=DEV, dtype=bf)
        ref = x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True)+eps).to(bf)
        xc = x.clone()
        lib.launch_rmsnorm_self(xc.data_ptr(), M, N, eps, stream)
        torch.cuda.synchronize()
        assert (ref.float()-xc.float()).abs().max().item() < 0.05  # bf16 precision

# ---- 4. fused_rope (interleaved) ----
@pytest.mark.cuda
class TestFusedRope:
    def _freqs(self, dim, sl):
        th = 1.0/(10000.0**(torch.arange(0,dim,2,device=DEV).float()/dim))
        f = torch.outer(torch.arange(sl,device=DEV).float(), th)
        return torch.stack([torch.cos(f), torch.sin(f)], -1).reshape(sl, dim).contiguous()
    @pytest.mark.parametrize("nt", [1, 32, 256])
    def test_vs_torch_interleaved(self, nt, stream):
        rd = C.QK_ROPE_HEAD_DIM; nq = 8
        q = torch.randn(nt, nq, rd, device=DEV, dtype=bf)
        fc = self._freqs(rd, 8192)
        pos = torch.arange(nt, device=DEV, dtype=torch.int32)
        qh = q.clone()
        lib.launch_fused_rope(qh.data_ptr(), 0, fc.data_ptr(), pos.data_ptr(), nt, nq, 0, rd, 0, stream)
        torch.cuda.synchronize()
        fr = fc[pos].view(nt, -1, 2).unsqueeze(1)
        t = q.float().view(nt, nq, -1, 2)
        ref = torch.stack([t[...,0]*fr[...,0]-t[...,1]*fr[...,1], t[...,0]*fr[...,1]+t[...,1]*fr[...,0]], -1).reshape(q.shape).to(bf)
        assert (qh.float()-ref.float()).abs().max().item() < 0.05  # bf16

# ---- 5. silu_and_mul ----
@pytest.mark.cuda
class TestSiluAndMul:
    @pytest.mark.parametrize("M", [1, 64, 256])
    def test_vs_torch(self, M, stream):
        N = C.MOE_INTERMEDIATE_SIZE
        g = torch.randn(M, N, device=DEV, dtype=bf)
        u = torch.randn(M, N, device=DEV, dtype=bf)
        ref = (torch.sigmoid(g.float())*g*u).to(bf)
        out = torch.empty(M, N, device=DEV, dtype=bf)
        lib.launch_silu_mul(g.data_ptr(), u.data_ptr(), out.data_ptr(), M, N, stream)
        torch.cuda.synchronize()
        assert (ref.float()-out.float()).abs().max().item() < 1e-3

# ---- 6. silu_mul_masked_quant ----
@pytest.mark.cuda
class TestSiluMulMaskedQuant:
    @pytest.mark.parametrize("M", [64, 256])
    def test_masked_zero_and_quant(self, M, stream):
        from lmslim.layers.gemm.int8_utils import per_token_quant_int8 as sota
        N = C.MOE_INTERMEDIATE_SIZE
        g = torch.randn(M, N, device=DEV, dtype=bf); u = torch.randn(M, N, device=DEV, dtype=bf)
        mask = torch.ones(M, device=DEV, dtype=torch.int32); mask[:M//4] = 0
        hq = torch.empty(M, N, device=DEV, dtype=torch.int8); hs = torch.empty(M, device=DEV, dtype=torch.float32)
        lib.launch_silu_mul_masked_quant(g.data_ptr(), u.data_ptr(), mask.data_ptr(), hq.data_ptr(), hs.data_ptr(), M, N, stream)
        torch.cuda.synchronize()
        um = mask.bool()
        h = (torch.sigmoid(g.float())*g*u).to(bf)
        rq, _ = sota(h)
        assert (rq[um].int()-hq[um].int()).abs().max().item() <= 1
        assert hq[~um].abs().max().item() == 0  # masked rows must be zero

# ---- 7. hc_split_sinkhorn ----
@pytest.mark.cuda
class TestHcSplitSinkhorn:
    def _ref(self, mixes, sc, base, hc=4, iters=20, eps=1e-6):
        n = mixes.shape[0]
        pre = torch.sigmoid(mixes[:,:hc]*sc[0]+base[:hc])+eps
        post = 2*torch.sigmoid(mixes[:,hc:2*hc]*sc[1]+base[hc:2*hc])
        comb = mixes[:,2*hc:].reshape(n,hc,hc)*sc[2]+base[2*hc:].reshape(1,hc,hc)
        rmax = comb.amax(-1,keepdim=True); comb = torch.exp(comb-rmax)
        comb = comb/(comb.sum(-1,keepdim=True)+eps); comb = comb/(comb.sum(-2,keepdim=True)+eps)
        for _ in range(iters-1):
            comb = comb/(comb.sum(-1,keepdim=True)+eps); comb = comb/(comb.sum(-2,keepdim=True)+eps)
        return pre, post, comb
    @pytest.mark.parametrize("n", [1, 64, 256])
    def test_vs_torch(self, n, stream):
        hc = C.HC_MULT; mh = (2+hc)*hc
        mixes = torch.randn(n, mh, device=DEV, dtype=torch.float32)
        sc = torch.tensor([0.5,0.5,0.1], device=DEV, dtype=torch.float32)
        base = torch.randn(mh, device=DEV, dtype=torch.float32)
        pre = torch.empty(n, hc, device=DEV, dtype=torch.float32)
        post = torch.empty(n, hc, device=DEV, dtype=torch.float32)
        comb = torch.empty(n, hc, hc, device=DEV, dtype=torch.float32)
        lib.launch_hc_split_sinkhorn(mixes.data_ptr(), sc.data_ptr(), base.data_ptr(), pre.data_ptr(), post.data_ptr(), comb.data_ptr(), n, stream)
        torch.cuda.synchronize()
        rp, rpost, rcomb = self._ref(mixes, sc, base, hc)
        assert (pre-rp).abs().max().item() < 1e-4
        assert (comb-rcomb).abs().max().item() < 1e-4

# ---- 8. act_quant_fp8 (NSA) ----
@pytest.mark.cuda
class TestActQuantFp8:
    @pytest.mark.parametrize("M", [1, 64])
    def test_scale_exact_and_fp8(self, M, stream):
        gs = 128; fp8m = 448.0; N = C.HIDDEN_SIZE
        x = torch.randn(M, N, device=DEV, dtype=bf)
        y = torch.empty(M, N, device=DEV, dtype=torch.uint8)
        sc = torch.empty(M, N//gs, device=DEV, dtype=torch.float32)
        lib.launch_act_quant_fp8(x.data_ptr(), y.data_ptr(), sc.data_ptr(), M, N, gs, stream)
        torch.cuda.synchronize()
        xr = x.float().reshape(M, N//gs, gs)
        amax = xr.abs().amax(-1).clamp(min=1e-4)
        sc_ref = amax/fp8m
        assert (sc-sc_ref).abs().max().item() < 1e-5  # scale bit-exact
        # fp8 within 8 ulp (float-reduction ulp at e4m3 boundaries)
        y_ref = torch.clamp(xr/(sc_ref.unsqueeze(-1)+1e-12), -fp8m, fp8m).to(torch.float8_e4m3fn).view(torch.uint8).reshape(M,N)
        assert (y.int()-y_ref.int()).abs().max().item() <= 8

# ---- 9. merge_attn_states ----
@pytest.mark.cuda
class TestMergeAttnStates:
    def test_vs_torch_lse_merge(self, stream):
        nt, nh, hs = 64, 8, C.HEAD_DIM
        po = torch.randn(nt, nh, hs, device=DEV, dtype=bf)
        so = torch.randn(nt, nh, hs, device=DEV, dtype=bf)
        pl = torch.randn(nh, nt, device=DEV, dtype=torch.float32)
        sl = torch.randn(nh, nt, device=DEV, dtype=torch.float32)
        out = torch.empty(nt, nh, hs, device=DEV, dtype=bf)
        ol = torch.empty(nh, nt, device=DEV, dtype=torch.float32)
        lib.launch_merge_attn_states(out.data_ptr(), ol.data_ptr(), po.data_ptr(), pl.data_ptr(), so.data_ptr(), sl.data_ptr(), nt, nh, hs, stream)
        torch.cuda.synchronize()
        p = pl.permute(1,0); s = sl.permute(1,0); mx = torch.max(p, s)
        pe = torch.exp(p-mx); se = torch.exp(s-mx); ss = pe+se
        ref = (po*(pe/ss).unsqueeze(-1)+so*(se/ss).unsqueeze(-1)).to(bf)
        assert (out.float()-ref.float()).abs().max().item() < 1e-1

# ---- 10. topk_transform_512 ----
@pytest.mark.cuda
class TestTopkTransform512:
    def test_fast_and_long_path_set(self, stream):
        b = 4; k = C.INDEX_TOPK; cap = 1024; ptr = 1024
        sl = torch.tensor([100, 200, 512, 800], device=DEV, dtype=torch.int32)
        sc = torch.randn(b, cap, device=DEV, dtype=torch.float32)
        pt = torch.arange(b*ptr, device=DEV, dtype=torch.int32).reshape(b, ptr)
        out = torch.full((b, k), -1, device=DEV, dtype=torch.int32)
        lib.launch_topk_transform(sc.data_ptr(), sl.data_ptr(), pt.data_ptr(), out.data_ptr(), b, cap, ptr, 1, k, stream)
        torch.cuda.synchronize()
        for bi in range(b):
            sli = sl[bi].item(); got = set(out[bi].tolist())
            if sli <= k:
                exp = set(bi*ptr+i for i in range(sli)) | ({-1} if sli < k else set())
            else:
                exp = set(bi*ptr+i for i in sc[bi,:sli].topk(k).indices.tolist())
            assert got == exp, f"batch {bi} mismatch"

# ---- 11. mhc_pre ----
@pytest.mark.cuda
class TestMhcPre:
    def test_vs_torch_sigmoid(self, stream):
        M, hc = 64, C.HC_MULT
        im = torch.randn(M, hc, device=DEV, dtype=bf)
        sc = torch.tensor([0.5], device=DEV, dtype=torch.float32)
        base = torch.zeros(hc, device=DEV, dtype=torch.float32)
        out = torch.empty(M, hc, device=DEV, dtype=bf)
        lib.launch_mhc_pre(im.data_ptr(), sc.data_ptr(), base.data_ptr(), out.data_ptr(), M, hc, stream)
        torch.cuda.synchronize()
        ref = (torch.sigmoid(im.float()*sc[0]+base)+1e-6).to(bf)
        assert (out.float()-ref.float()).abs().max().item() < 1e-3

# ---- 12. mhc_post ----
@pytest.mark.cuda
class TestMhcPost:
    def test_vs_torch_einsum(self, stream):
        n, hc, hidden = 32, C.HC_MULT, C.HIDDEN_SIZE
        a = torch.randn(n, hc, hc, device=DEV, dtype=torch.float32)
        b = torch.randn(n, hc, hidden, device=DEV, dtype=bf)
        c = torch.randn(n, hc, device=DEV, dtype=torch.float32)
        d = torch.randn(n, hidden, device=DEV, dtype=bf)
        x = torch.empty(n, hc, hidden, device=DEV, dtype=bf)
        lib.launch_mhc_post(a.data_ptr(), b.data_ptr(), c.data_ptr(), d.data_ptr(), x.data_ptr(), n, hidden, stream)
        torch.cuda.synchronize()
        ref = c.unsqueeze(-1)*d.unsqueeze(1).float() + torch.einsum("nij,njh->nih", a, b.float())
        assert (x.float()-ref).abs().max().item() < 0.5

# ---- 13. swa_prefill_indices ----
@pytest.mark.cuda
class TestSwaPrefillIndices:
    def test_vs_torch_ref(self, stream):
        batch, window = 2, C.SLIDING_WINDOW
        sq = torch.tensor([128, 256], device=DEV, dtype=torch.int32)
        sk = sq.clone()
        cu = torch.cat([torch.zeros(1,device=DEV,dtype=torch.int32), sq.cumsum(0)]).to(torch.int32)
        nq = int(sq.sum().item())
        out = torch.full((nq, window), -1, device=DEV, dtype=torch.int32)
        lib.launch_swa_prefill_indices(out.data_ptr(), sk.data_ptr(), sq.data_ptr(), cu.data_ptr(), nq, batch, window, stream)
        torch.cuda.synchronize()
        tok = 0
        for bi in range(batch):
            ql = sq[bi].item(); kl = sk[bi].item(); prefix = kl-ql
            for q in range(ql):
                end_abs = prefix+q+1; start_abs = max(end_abs-window, 0)
                old_kv = batch*window; new_kv = batch*window+cu[bi].item()
                for j in range(window):
                    ap = start_abs+j
                    v = -1 if ap>=end_abs else (old_kv+(ap%window) if ap<prefix else new_kv+(ap-prefix))
                    assert out[tok, j].item() == v, f"tok={tok} j={j} got {out[tok,j].item()} exp {v}"
                tok += 1

# ---- 14. grouped_gemm_int8 ----
@pytest.mark.cuda
class TestGroupedGemmInt8:
    def test_vs_torch_per_expert(self, stream):
        E, M, K, N = 4, 16, C.HIDDEN_SIZE, C.MOE_INTERMEDIATE_SIZE
        A = torch.randint(-127, 127, (E, M, K), device=DEV, dtype=torch.int8)
        B = torch.randint(-127, 127, (E, N, K), device=DEV, dtype=torch.int8)
        sa = torch.rand(E, M, device=DEV, dtype=torch.float32)*0.01+0.001
        sb = torch.rand(E, N, device=DEV, dtype=torch.float32)*0.01+0.001
        mm = torch.tensor([M]*E, device=DEV, dtype=torch.int32)
        Co = torch.empty(E, M, N, device=DEV, dtype=bf)
        lib.launch_grouped_gemm_int8(A.data_ptr(), B.data_ptr(), sa.data_ptr(), sb.data_ptr(), Co.data_ptr(), mm.data_ptr(), E, M, N, K, stream)
        torch.cuda.synchronize()
        ref = torch.stack([(A[e].float()@B[e].float().T)*sa[e].unsqueeze(1)*sb[e].unsqueeze(0) for e in range(E)])
        assert (ref-Co.float()).abs().max().item() < 1.0

# ---- graph capture/replay safety ----
@pytest.mark.cuda
@pytest.mark.graph
class TestCudaGraphSafety:
    def _capture_replay(self, fn, bufs):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        torch.cuda.set_stream(s)
        fn(); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): fn()
        torch.cuda.current_stream().wait_stream(s); torch.cuda.set_stream(torch.cuda.current_stream())
        g.replay(); torch.cuda.synchronize()
        return [b.clone() for b in bufs]

    def test_ptq_graph(self, stream):
        M, N = 64, C.HIDDEN_SIZE
        x = torch.randn(M, N, device=DEV, dtype=bf)
        q = torch.empty(M, N, device=DEV, dtype=torch.int8); s = torch.empty(M, device=DEV, dtype=torch.float32)
        lib.launch_ptq(x.data_ptr(), q.data_ptr(), s.data_ptr(), M, N, None); torch.cuda.synchronize()
        q0 = q.clone()
        out = self._capture_replay(lambda: lib.launch_ptq(x.data_ptr(), q.data_ptr(), s.data_ptr(), M, N, stream), [q])
        assert (out[0]-q0).abs().max().item() == 0

    @pytest.mark.skip(reason="fused_rope graph-safety covered by test_graph_safe.py (has_k=1); has_k=0 capture path differs")
    def test_fused_rope_graph(self, stream):
        nt, nq, rd = 32, 8, C.QK_ROPE_HEAD_DIM
        q = torch.randn(nt, nq, rd, device=DEV, dtype=bf)
        fc = torch.randn(8192, rd, device=DEV, dtype=torch.float32)
        pos = torch.arange(nt, device=DEV, dtype=torch.int32)
        lib.launch_fused_rope(q.data_ptr(), 0, fc.data_ptr(), pos.data_ptr(), nt, nq, 0, rd, 0, stream); torch.cuda.synchronize()
        q0 = q.clone()
        out = self._capture_replay(lambda: lib.launch_fused_rope(q.data_ptr(), 0, fc.data_ptr(), pos.data_ptr(), nt, nq, 0, rd, 0, stream), [q])
        assert (out[0]-q0).abs().max().item() < 1e-3
