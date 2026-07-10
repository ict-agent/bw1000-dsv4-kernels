# SKILLS.md — 所需技能 + 检查清单

## Skill 1: HIP kernel 编写（gfx936）

- hipcc `/opt/dtk/bin/hipcc`，`--offload-arch=gfx936`
- wavefront64（不是 32）：`__shfl_xor(val, o, 64)`，reduce 用 `wsum64`/`wmax64`
- bf16：`__hip_bfloat16`，`b2f`/`f2b` helper
- fp8 e4m3：手写 `f32_to_e4m3`（hip_fp8.h 在本 DTK 版本不可用，缺 `__ockl_fprintf_stderr_begin`/`__clz`）
- block max threads = 256（gfx936），不是 512
- 所有 launch 带 `hipStream_t st`

**检查清单**：
- [ ] kernel 用 stream 参数
- [ ] 无 host 分配/sync
- [ ] shared mem 不超 64KB
- [ ] block ≤ 256 threads
- [ ] 正确性 bit-exact 或量化噪声内

## Skill 2: Python wrapper graph-safety

ctypes 直接 `cudaLaunchKernel` 绕过 PyTorch dispatch，输出 tensor 指针不被 graph 跟踪。

**unsafe 操作**（capture 内必 crash/hang）：
- `torch.empty`/`torch.empty_like`/`torch.full`/`torch.zeros`/`torch.arange`/`torch.cumsum` → 改 `_buf(name=...)`
- `.contiguous()`（non-contig 时 alloc）→ 改 kernel 读 stride 或 split kernel
- `.reshape()`（non-contig 返回 copy）→ 确保 contiguous 或用 view
- `.item()` → CPU sync，改用 `.shape[]` 或 `.numel()`
- `int(tensor.sum().item())` → 同上

**检查清单**：
- [ ] 返回新 tensor 的 wrapper 用 `_buf(name=...)`
- [ ] 同 shape 多输出用不同 name（防 alias）
- [ ] in-place/out-param 的 wrapper 不 alloc
- [ ] 无 `.item()` / CPU sync
- [ ] 签名与 sglang 调用点逐字对齐（agent 调研确认）

## Skill 3: 引擎调用路径调研

派 Explore agent，给定：
- op 函数名 + 源码路径
- 要求：精确签名、实际 decode/prefill shape、Python 层创建的 tensor、env 分支、可移入 kernel 的操作

**检查清单**：
- [ ] 确认引擎实际调用哪个函数（不是猜的）
- [ ] 确认 env 开关（SGLANG_OPT_USE_*）决定走哪个分支
- [ ] 确认 dtype（gfx936 e4m3 vs gfx94x e5m2fnuz）
- [ ] 确认 in-place vs return vs out-param
- [ ] 对照 `/workspace/dsv4_ops_unit_tests/README.md` 真实 shape 表

## Skill 4: Profiling 验证单 kernel 加速比

**目的**：在真实 sglang 推理中 profile，确认单 kernel 加速比真实（不只离线 bench）。

**方法 A: torch profiler（Python 层 op 时间）**
```python
import torch.profiler as prof
with prof.profile(activities=[prof.ProfilerLevel.CPU, prof.ProfilerLevel.GPU]) as p:
    # 发请求 / 跑 forward
print(p.key_averages().table(sort_by="cuda_time", row_limit=20))
```

**方法 B: hipprof / rocprof（kernel 级）**
```bash
# 在容器内
rocprof --stats -o profile.csv -- python3 -c "..."  # 或对 server 进程 attach
# 或用 hipprof_trace
HIP_PROFILE=1 python3 ...
```

**方法 C: 在 wrapper 内加 timing**
```python
import torch
def wrapped(fn):
    def _():
        torch.cuda.synchronize(); t0=time.time(); fn(); torch.cuda.synchronize()
        # log (time.time()-t0)
    return _
```

**检查清单**：
- [ ] profile 时用实际推理 shape（不是 bench 造的）
- [ ] 对比 patch on/off 的同一 op 时间
- [ ] 确认加速来自 kernel（不是 dispatch 减少）
- [ ] 端到端无加速时，profile 找瓶颈在哪个 op

## Skill 5: 8-GPU server 启动 + e2e benchmark

**启动**（17 分钟 cuda graph capture）：
```bash
SGLANG_APPLY_CONFIG_BACKUP=none PYTHONPATH=/workspace/hip_kernels:/workspace/sglang/python \
SGLANG_USE_HIP_DSV4=1 SGLANG_HIP_PTQ=1 SGLANG_HIP_SILU=1 SGLANG_HIP_ROPE=1 \
SGLANG_HIP_TOPK=1 SGLANG_HIP_SWA=1 \
python3 -m sglang.launch_server --host 127.0.0.1 --port 30001 \
  --model-path /home_aclsylqidf/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8 --tp 8 \
  --quantization slimquant_marlin --moe-a2a-backend none \
  --cuda-graph-max-bs 256 --mem-fraction-static 0.76
```

**A/B benchmark**：baseline（不设 SGLANG_USE_HIP_DSV4）vs HIP-on，同 config。

**检查清单**：
- [ ] baseline 和 HIP-on 用相同 mem_frac/max_bs（否则不公平）
- [ ] server "fired up and ready to roll" 后才测
- [ ] 多次请求取中位数
- [ ] 停 server 用 PID kill（不 pkill -f sglang）

## Skill 6: 容器/磁盘管理

- `CLAUDE_CODE_TMPDIR=~/tmp`（/tmp 在根分区易满）
- 根分区满 → docker exec 报 "rg not found"（实际 ENOSPC pivot）→ 清 `/workspace/core.*` + /tmp
- core dump 堆积（server crash 留 core）→ 定期 `rm /workspace/core.*`
- 查根分区大头：`du -sh /workspace/* | sort -rh`
- 杀 server：`kill $(pgrep -f sglang.launch_server | head -1)`，不要 `pkill -f sglang`

## Skill 7: 诚实性能报告

- 区分 vs torch naive 和 vs vendor SOTA
- 端到端无加速时明确说"被 GEMM/attention 主导"
- 不拿 torch ref 的数字冒充 SOTA speedup
- 单 kernel 加速 ≠ 端到端加速（elementwise 占比小）
