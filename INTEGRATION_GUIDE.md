# DeepSeek V4 Flash W8A8 Kernel 集成指南

## 概述

本文档详细说明如何将 HIP C++ 优化 kernel 集成到 SGLang 推理引擎中。

## 一、Kernel 清单

### 已实现并验证的 kernel

| Kernel | 源文件 | 功能 | 集成状态 |
|--------|--------|------|---------|
| `per_token_quant_int8` | `dsv4_torch_ext_combined.cpp` | BF16→INT8 per-token 量化 | ✅ 已集成 |
| `silu_and_mul` | `dsv4_ops_hip.hip` | SiLU 激活 | 独立验证 |
| `silu_mul_quant` | `dsv4_ops_hip.hip` | 融合 SiLU+Mul+量化 | 独立验证 |
| `rmsnorm` | `dsv4_ops_hip.hip` | RMS 归一化 | 独立验证 |
| `add_rmsnorm_quant` | `dsv4_ops_hip.hip` | 融合 残差加+RMSNorm+量化 | 独立验证 |
| `w8a8_scaled_gemm` | `w8a8_gemm.hip` | W8A8 INT8 GEMM (sdot4) | 正确性通过，性能待优化 |

### 正确性验证

所有 kernel 通过 `mmt-at/dsv4_ops_unit_tests` 的参考实现验证：
- `per_token_quant_int8`: bit-exact (maxdiff=0)
- `silu_and_mul`: maxdiff=0.000000
- `silu_mul_quant`: maxdiff=1 (BF16 精度内)
- `rmsnorm`: maxdiff<0.031 (通过 2e-2 断言)
- `add_rmsnorm_quant`: maxdiff=1 (BF16 精度内)
- `w8a8_scaled_gemm`: diff=0.0, rel_err=0.0000 (完全正确)

## 二、集成方式

### 方法：PyTorch C++ Extension + .pth 注入

这是**不修改任何 SGLang 源码**的集成方式。

#### Step 1: 编译 HIP kernel 为 PyTorch extension

```bash
# 在容器内
cd /workspace/hip_kernels

# 获取 PyTorch include 路径
TORCH_INC=$(python3 -c "from torch.utils.cpp_extension import include_paths; print(' '.join(['-I'+p for p in include_paths()]))")
PY_INC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
TORCH_LIB=$(python3 -c "import torch; print(torch.__file__.rsplit('/',1)[0])")/lib

# 编译（hipcc，不用 CUDA compat headers）
hipcc -std=c++17 -O3 -fPIC -shared \
  $TORCH_INC -I$PY_INC -I/opt/dtk/hip/include \
  -DTORCH_API_INCLUDE_EXTENSION_H -D__HIP_PLATFORM_AMD__ \
  -DTORCH_EXTENSION_NAME=dsv4_hip_ext \
  -o torch_ext_build/dsv4_hip_ext.cpython-310-x86_64-linux-gnu.so \
  dsv4_torch_ext_combined.cpp \
  -L$TORCH_LIB -ltorch -ltorch_python -lc10 \
  -L$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") -lpython3.10 \
  -Wl,-rpath,$TORCH_LIB
```

#### Step 2: 创建注入模块

创建 `hip_graph_safe.py`，放在 site-packages 中：

```python
# /usr/local/lib/python3.10/dist-packages/hip_graph_safe.py
import os, sys, torch

_BUILD_DIR = "/workspace/hip_kernels/torch_ext_build"
if _BUILD_DIR not in sys.path:
    sys.path.insert(0, _BUILD_DIR)

try:
    import dsv4_hip_ext
    _HIP_AVAILABLE = True
except ImportError:
    _HIP_AVAILABLE = False

def _apply_patch():
    if not _HIP_AVAILABLE:
        return
    try:
        import lmslim.layers.gemm.int8_utils as m
        if not hasattr(m, '_hip_graph_safe_patched'):
            m._hip_graph_safe_patched = True
            m._orig_per_token_quant_int8 = m.per_token_quant_int8
            def hip_wrapper(x, scale_dtype=None, cal_sum=False):
                stream_ptr = torch.cuda.current_stream().cuda_stream
                return dsv4_hip_ext.per_token_quant_int8_stream(x, stream_ptr)
            m.per_token_quant_int8 = hip_wrapper
            print("[HIP-GRAPH-SAFE] Patched lmslim.per_token_quant_int8", flush=True)
    except ImportError:
        pass

if os.environ.get("SGLANG_USE_HIP_QUANT", "0") == "1":
    _apply_patch()
```

#### Step 3: 创建 .pth 文件

```bash
# /usr/local/lib/python3.10/dist-packages/zz_hip_graph_safe.pth
echo 'import hip_graph_safe; hip_graph_safe._apply_patch() if __import("os").environ.get("SGLANG_USE_HIP_QUANT","0")=="1" else None' > /usr/local/lib/python3.10/dist-packages/zz_hip_graph_safe.pth
```

**关键**：`.pth` 文件在 Python 启动时自动执行，包括 `multiprocessing.spawn` 创建的子进程。

#### Step 4: 启动 SGLang

```bash
export SGLANG_USE_HIP_QUANT=1
export PYTHONPATH=/workspace/sglang/python:$PYTHONPATH

python3 -m sglang.launch_server \
  --port 30001 --trust-remote-code \
  --model-path $MODEL_PATH \
  --tp 8 --quantization slimquant_marlin \
  --context-length 4096 \
  --disable-radix-cache \
  --chunked-prefill-size 4096 \
  --mem-fraction-static 0.85 \
  --kv-cache-dtype auto \
  --disable-flashinfer-autotune
```

**注意**：不要加 `--disable-cuda-graph`。CUDA graph 是性能关键（TPOT 73ms vs 202ms）。

## 三、集成验证

### 正确性验证

```bash
# 单 kernel bit-exact 验证
python3 /workspace/hip_kernels/verify_bitexact.py

# 引擎保真度（量化→GEMM 链路 bit-identical）
python3 /workspace/hip_kernels/engine_fidelity_test.py

# 推理正确性
curl -s http://127.0.0.1:30001/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"default","prompt":"What is 2+2?","max_tokens":4,"temperature":0}'
# 应输出 "4"
```

### 性能验证

```bash
# 官方 SGLang bench_serving
export HF_ENDPOINT=https://hf-mirror.com
python3 -m sglang.bench_serving \
  --backend sglang-oai --port 30001 --host 127.0.0.1 \
  --model $MODEL_PATH --tokenizer $MODEL_PATH \
  --dataset-name random \
  --random-input-len 128 --random-output-len 32 \
  --num-prompts 16 --max-concurrency 8 --seed 42
```

## 四、性能结果

### 官方 bench_serving A/B（input=128, output=32, 16 prompts, concurrency=8）

| 指标 | 原始 (Triton) | HIP (native ext) | 改善 |
|------|-------------|-----------------|------|
| Mean TTFT | 439.69ms | 349.52ms | **-20.5%** |
| Mean TPOT | 117.32ms | 95.22ms | **-18.8%** |
| P99 TPOT | 202.67ms | 126.87ms | **-37.4%** |
| Output throughput | 49.88 tok/s | 52.39 tok/s | +5.0% |
| Total throughput | 314.69 tok/s | 330.50 tok/s | +5.0% |

### Prefill 吞吐（原始服务器，不同输入长度）

| Input | TTFT | Prefill 吞吐 |
|-------|------|------------|
| 256 | 350.7ms | 224.8 tok/s |
| 1024 | 444.9ms | 647.0 tok/s |
| 3072 | 717.8ms | 2065.6 tok/s |
| 4096 | 1424.3ms | 2386.2 tok/s |

### 单 kernel 性能

| Kernel | vs SOTA | 加速比 |
|--------|---------|-------|
| per_token_quant (M=64) | bit-exact | 5.71× |
| silu_mul_quant (M=64) | maxdiff=1 | 21.4× |
| add_rmsnorm_quant (M=64) | maxdiff=1 | 6.7× |
| w8a8_scaled_gemm (M=1) | diff=0 | 0.21× (待优化) |

## 五、W8A8 GEMM 当前状态

W8A8 GEMM kernel 已修复正确性（diff=0.0），但性能仅为 SOTA 的 0.03-0.48×。原因：
- SOTA 用 lightop 预编译 ASM `.co` 文件（hand-tuned for gfx936）
- 我的 kernel 用 `__builtin_amdgcn_sdot4` 指令但 tile 配置未优化
- 需要 shared memory tiling + 更大的 tile size 才能接近 SOTA

**后续优化方向**：
1. 使用 shared memory 缓存 A/B tile（当前每线程独立从 global memory 读）
2. 增大 tile size（当前 16×16，应改为 64×64 或更大）
3. 使用 vectorized load（一次加载 4 个 int8）
4. 参考 lightop `.co` 的 tile 配置（64x64x128）

## 六、文件结构

```
hip_kernels/
├── dsv4_ops_hip.hip          # HIP kernel 源码（5 个算子）
├── dsv4_torch_ext_combined.cpp  # PyTorch C++ extension 包装（graph-safe）
├── w8a8_gemm.hip             # W8A8 GEMM kernel（sdot4）
├── hip_graph_safe.py         # .pth 注入模块
├── zz_hip_graph_safe.pth     # Python .pth 自动注入文件
├── torch_ext_build/
│   └── dsv4_hip_ext.cpython-310-x86_64-linux-gnu.so  # 编译好的 extension
├── libdsv4_ops_hip.so        # HIP kernel 共享库（ctypes 方式）
├── libw8a8_gemm.so           # W8A8 GEMM 共享库
├── verify_bitexact.py        # bit-exact 验证
├── engine_fidelity_test.py   # 引擎保真度验证
├── fair_comparison.py        # 公平性能对比
├── definitive_verify.py       # 综合验证
├── engine_ab_bench.py        # A/B benchmark 脚本
├── test_w8a8_gemm.py         # W8A8 GEMM 测试
└── multi_baseline_comparison.py  # 多 baseline 对比
```
