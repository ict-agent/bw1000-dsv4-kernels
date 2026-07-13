#!/bin/bash
# Simplified DeepSeek V4 Flash 8-GPU server launch (no DeepEP/hcoll dependency).
# TP=8, cuda graph ON, moe-a2a-backend=none. For HIP patch on/off A/B benchmark.
# Usage: SGLANG_USE_HIP_DSV4=1 (set to enable HIP patches) bash serve_8gpu.sh
set -e
# rm -rf ~/.cache/   # disabled: forces lightop/tilelang full recompile each run (~20min, can hang)
ulimit -c 0          # disable core dumps — sglang crashes produce 5G+ cores that fill disk
ulimit -l unlimited
export HIP_KERNEL_BATCH_CEILING=100
export GPU_MAX_HW_QUEUES=3
export HIP_H2D_DISABLE_COPY_BUFFER=0
export HIP_D2H_DISABLE_COPY_BUFFER=0
export HIP_H2D_DIRECT_COPY_THRESHOLD=32768
export HIP_H2D_HSAAPI_COPY_THRESHOLD=32768
export HIP_D2H_DIRECT_COPY_THRESHOLD=512
export HIP_D2H_HSAAPI_COPY_THRESHOLD=512
export USE_DCU_CUSTOM_ALLREDUCE=1
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export SGLANG_USE_FP8_W8A8_MOE=0
export SGLANG_GROUPGEMM=true
export SGLANG_USE_LIGHTOP=1
export SGLANG_ROCM_USE_AITER_MOE=0
export SGLANG_OPT_USE_FUSED_HASH_TOPK=true
export SGLANG_TOPK_TRANSFORM_512_TORCH=true
export SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=true
export SGLANG_NSA_FUSE_TOPK=false
export SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=128
export SGLANG_DSV4_MODE=2604
export SGLANG_APPLY_CONFIG_BACKUP=none   # critical: default 'auto' rewrites config to fp8, mismatching slimquant

sglang_dir=/workspace/sglang
MODEL_PATH=${MODEL_PATH:-/home_aclsylqidf/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8}
PORT=${PORT:-30001}
export PYTHONPATH=$sglang_dir/python:/workspace/hip_kernels:$PYTHONPATH

python3 -m sglang.launch_server \
  --host 127.0.0.1 \
  --port "$PORT" \
  --trust-remote-code \
  --model-path $MODEL_PATH \
  --tp 8 \
  --moe-a2a-backend none \
  --quantization slimquant_marlin \
  --disable-radix-cache \
  --chunked-prefill-size 8192 \
  --chat-template $sglang_dir/examples/chat_template/tool_chat_template_deepseekv3.jinja \
  --kv-cache-dtype auto \
  --cuda-graph-max-bs 256 \
  --mem-fraction-static ${MEM_FRACTION:-0.76} \
  --disable-overlap-schedule \
  --disable-flashinfer-autotune \
  2>&1 | tee -a /workspace/server_${MODE:-run}.log
