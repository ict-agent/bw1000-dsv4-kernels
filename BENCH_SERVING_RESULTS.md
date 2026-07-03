# SGLang 官方 bench_serving A/B 对比结果

## 测试参数
- dataset: random, input=128 tokens, output=32 tokens
- 16 prompts, max-concurrency=8, seed=42
- CUDA graph enabled, TP=8, slimquant_marlin W8A8

## 结果

| 指标 | 原始 (Triton) | HIP (native ext) | 改善 |
|------|-------------|-----------------|------|
| Mean TTFT | 439.69ms | 349.52ms | **-20.5%** |
| Mean TPOT | 117.32ms | 95.22ms | **-18.8%** |
| P99 TPOT | 202.67ms | 126.87ms | **-37.4%** |
| Output throughput | 49.88 tok/s | 52.39 tok/s | +5.0% |
| Total throughput | 314.69 tok/s | 330.50 tok/s | +5.0% |

## 集成方式
- HIP kernel 编译为 PyTorch C++ extension（`dsv4_torch_ext_combined.cpp`）
- 通过 `.pth` 文件注入 site-packages，在所有 spawn 子进程生效
- 使用 `per_token_quant_int8_stream(x, stream_ptr)` 传入正确的 CUDA stream
- Graph-safe：PyTorch extension 被 CUDA graph 正确捕获
