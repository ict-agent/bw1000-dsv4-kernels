# AGENTS.md — 给 Agent 的工作指南

DeepSeek V4 Flash HIP kernel 优化与 sglang 引擎集成。目标硬件 Hygon DCU gfx936。

## 不可违反的规则

1. **三层架构**：HIP kernel（Layer 1，功能正确性）→ hip_wrapper.py（Layer 2，sglang 签名对齐 + graph-safe buffer pool）→ hip_dsv4_integration.py（Layer 3，引擎 monkey-patch）。不要跳层。
2. **性能评测用 Layer 2 wrapper**（不是裸 ctypes kernel）。裸 kernel 没有缓冲池 dispatch 开销，不代表真实性能。
3. **wrapper 必须与 sglang 调用点签名逐字对齐**（参数名、in-place vs return、out-param vs 分配）。改前必须先用 agent 调研引擎实际调用路径（见 WORKFLOW.md Phase 0）。
4. **graph-safe**：ctypes 路径下 `torch.empty`/`torch.empty_like`/`torch.full`/`torch.cumsum`/`.contiguous()`(non-contig)/`.item()` 都是 unsafe。输出 tensor 用 `_buf(name=...)` 静态池；in-place/out-param 的 wrapper 不用 `_buf`。
5. **不要 pkill -f sglang**（容器名含 sglang，会杀容器自身）。用 PID kill。
6. **不要在 capture 期间用 `.item()` 或 CPU sync**——必 crash。
7. **根分区** `/` 在 `/dev/nvme1n1p3`（437G），`/home` 在 `/dev/nvme0n1p1`（1.8T）。`/tmp` 在根分区，容易满。设 `CLAUDE_CODE_TMPDIR=~/tmp` 避开。docker overlay2 在根分区，根满时 `docker exec` pivot 失败报 "rg not found"（误导性错误）。
8. **诚实报告性能**：区分 vs torch naive 和 vs vendor SOTA。fused_rope 12x 是 vs jit_kernel（真 SOTA）；不要拿 torch ref 的数字充 SOTA。

## 环境

- 容器：`baoming_test`（d6e9ca5669f2），镜像 sglang-deepseek-v4-dev-zkjh
- 工具链：DTK hipcc `/opt/dtk/bin/hipcc`，gfx936
- sglang 源码：`/workspace/sglang/python/sglang`
- 权重：`/home_aclsylqidf/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8`（275G，slimquant_marlin W8A8，43层 256 expert）
- 单元测试参考：`/workspace/dsv4_ops_unit_tests`（README 有真实 shape 表）
- GPU：8× HCU 64G，`is_fp8_fnuz()=False`（gfx936 用 e4m3，非 e5m2fnuz）

## 关键命令

```bash
# 编译 kernel
cd /workspace/hip_kernels && bash build.sh   # 或 hipcc -O3 --offload-arch=gfx936 -shared -fPIC -o libdsv4_all_hip.so dsv4_all_hip_kernels.hip

# 单元测试（实际推理 shape）
python -m pytest tests/test_wrappers.py -v          # Layer 2 正确性+graph
python -m pytest tests/test_wrapper_engine_convention.py -v  # 引擎调用约定
python -m pytest tests/test_kernels.py -v           # Layer 1 裸 kernel

# 性能（wrapper 层 vs SOTA）
python bench_wrapper_perf.py    # 实际 decode shape
python bench_vs_sota.py         # vs sglang SOTA

# 启动 8-GPU server（需 17 分钟 cuda graph capture）
SGLANG_APPLY_CONFIG_BACKUP=none python3 -m sglang.launch_server \
  --host 127.0.0.1 --port 30001 --model-path <weight> --tp 8 \
  --quantization slimquant_marlin --moe-a2a-backend none \
  --cuda-graph-max-bs 256 --mem-fraction-static 0.76

# 停 server（用 PID，不要 pkill -f sglang）
kill $(pgrep -f sglang.launch_server | head -1)
```

## 已知的引擎调用路径（agent 调研结论）

| op | 引擎实际调用 | patch 目标 | 命中 | 备注 |
|---|---|---|---|---|
| per_token_quant_int8 | lmslim.per_token_quant_int8 | lmslim attr + **call-site fuse_moe_w4a8_marlin** | ✅ | call-site 顶层 import 绑定，必须 patch 调用点 |
| per_token_group_quant_int8 | 本模型不调用 | — | ❌ | slimquant 走 per-token |
| silu_and_mul | SiluAndMul.forward_cuda（共享专家） | 类方法 | ✅ | dispatch_forward 构建时绑定，无 call-site 问题 |
| rmsnorm_self | SGLANG_OPT_USE_JIT_NORM=1 时 | jit_kernel attr | 需 env | 默认走 rms_normalize_triton |
| act_quant | nsa.triton_kernel.act_quant（C4Indexer） | triton_kernel attr + **call-site indexer** | ✅ | NSA indexer DCU 走 lightop，triton patch 不命中 |
| hc_split_sinkhorn | SGLANG_OPT_USE_TILELANG_MHC_PRE=False 时 | mhc.hc_split_sinkhorn | 需 env | 默认走 mhc_pre tilelang |
| mhc_post | mhc.mhc_post（tilelang 入口） | mhc attr | ✅ | 不是 mhc_post_torch |
| fused_rope | jit_kernel.fused_rope | jit_kernel attr + **call-site deepseek_v4 model** | ✅ | call-site 顶层 import 绑定 |
| topk_transform_512 | indexer.topk_transform_512_pytorch_vectorized（SGLANG_TOPK_TRANSFORM_512_TORCH=true） | indexer attr | ✅ | 同模块调用，无 call-site 问题 |
| swa | jit_kernel.tilelang_make_swa_prefill_indices | jit_kernel attr + **call-site paged_prefill** | ✅ | call-site 顶层 import 绑定 |
| merge_attn_states | sglang 不调 vllm merge | — | ❌ | 无 patch 点 |

## 性能真相

- **单 kernel wrapper 层超 SOTA**：fused_rope 12x、silu 5x、ptq 3x、rmsnorm 2.7x、act_quant 2.3x、merge 10x（vs torch ref；部分 vs lmslim triton）
- **端到端 8-GPU 无可见加速**：GEMM(marlin)+attention(compressed)+MoE dispatch 主导，elementwise <5%。vendor SOTA 已在 gfx936 充分优化。
- **op-chain 隔离测试 2.28x**：elementwise 占比高时可见收益。

## 常见陷阱（已踩过）

1. `_buf` 不加 name tag → 同 shape 输出 alias（pre/post 互相覆盖）
2. silu wrapper 用 `gate.copy_(x[...,:d])` + `up.copy_(x[...,d:])` 两个同 shape _buf → 冲突。解法：split-layout kernel 直接从 x 读
3. act_quant 挂 tilelang_kernel → 不命中。C4Indexer 用 triton_kernel
4. swa `.item()` → graph capture 内 CPU sync crash。改用 `swa_indices.shape[0]`
5. rmsnorm in-place → 语义错（sglang 返回新 tensor）。改 out-param kernel
6. fused_rope 挂 apply_rotary_emb → 签名不匹配永不调用。挂 jit_kernel.fused_rope
7. mem_frac 太高 + buffer pool → OOM。降到 0.76
8. /tmp 满 → docker exec 报 "rg not found"（实际是 pivot dir ENOSPC）。设 CLAUDE_CODE_TMPDIR=~/tmp
9. core dump 堆积 → 根分区满。定期 rm /workspace/core.*
10. **server crash 留 zombie VRAM**：sglang TP worker crash 后显存不释放，`rocm-smi` 显示 68% used 但无 python 进程，导致下次启动 load weight OOM（"0 bytes free"）。解法：`for i in 0 1 2 3 4 5 6 7; do rocm-smi --hcureset -d $i; done`（容器内 root 可执行，不需要 sudo）。**每次 server crash 后必做这一步**，否则必然 OOM。
11. **profiling 不能在 capture 期 sync**：`_timed` 里 `stream.synchronize()` 在 cuda graph capture 内是非法操作，必 crash。必须用 `torch.cuda.is_current_stream_capturing()` 检测并 skip timing（capture 期 kernel 仍正常 launch 进 graph，只是不计时）。decode(bs≤cuda_graph_max_bs)走 graph replay，wrapper 不被调用，profiling 只覆盖 prefill 路径。
12. **call-site binding 陷阱（最隐蔽）**：sglang/lmslim 模块用顶层 `from <mod> import <func>` 绑定函数。patch 定义模块属性（`mod.func = new`）**不改变**已 import 的调用点模块里那个名字。例：`fuse_moe_w4a8_marlin.py:18` 顶层 `from int8_utils import per_token_quant_int8`——patch `int8_utils.per_token_quant_int8` 后，`fuse_moe_w4a8_marlin` 命名空间里仍是旧函数。**必须额外 patch 调用点模块命名空间**（`fuse_moe_w4a8_marlin.per_token_quant_int8 = new`），见 hip_dsv4_integration.py `_patch_callsite`。受影响：ptq(fuse_moe_w4a8_marlin/quant_tools)、rope(deepseek_v4 model)、swa(paged_prefill)。topk(indexer 同模块调用，无此问题)、silu(类方法 dispatch，无此问题)。**验证 patch 命中**：`dv4.fused_rope is W.fused_rope` 必须为 True（offline 脚本见 REPORT.md）。
