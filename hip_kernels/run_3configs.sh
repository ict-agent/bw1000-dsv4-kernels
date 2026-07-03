#!/bin/bash
# Orchestrator: for each config, patch -> launch -> wait -> bench -> kill
set -e
export MODEL_PATH=/home_aclsylqidf/shared/hygon_DeepSeek-V4-Flash-Channel-INT8-w8a8
export SGLANG_DSV4_MODE=2604 SGLANG_USE_LIGHTOP=1 SGLANG_ROCM_USE_AITER_MOE=0 SGLANG_USE_FP8_W8A8_MOE=0 SGLANG_GROUPGEMM=true USE_DCU_CUSTOM_ALLREDUCE=1 SGLANG_APPLY_CONFIG_BACKUP=none SGLANG_OPT_USE_FUSED_HASH_TOPK=true SGLANG_TOPK_TRANSFORM_512_TORCH=true SGLANG_JIT_DEEPGEMM_PRECOMPILE=0
export PYTHONPATH=/workspace/sglang/python:$PYTHONPATH

for MODE in orig github hip; do
  echo "########## CONFIG: $MODE ##########"
  pkill -9 -f sglang 2>/dev/null; pkill -9 -f launch_server 2>/dev/null; sleep 4
  python3 /workspace/hip_kernels/patch_mode.py $mode 2>/dev/null || python3 /workspace/hip_kernels/patch_mode.py $MODE
  cd /workspace/sglang
  nohup python3 -m sglang.launch_server --port 30001 --trust-remote-code --model-path $MODEL_PATH --tp 8 --quantization slimquant_marlin --context-length 4096 --disable-radix-cache --chunked-prefill-size 4096 --disable-cuda-graph --mem-fraction-static 0.85 --kv-cache-dtype auto --disable-flashinfer-autotune > /workspace/sglang_$MODE.log 2>&1 &
  echo "launched $MODE, waiting for ready..."
  for i in $(seq 1 120); do
    if curl -s --max-time 3 http://127.0.0.1:30001/v1/models >/dev/null 2>&1; then
      echo "READY after ${i}0s"; break
    fi
    sleep 10
  done
  sleep 10  # warmup settle
  echo "--- bench $MODE ---"
  python3 /workspace/hip_kernels/engine_ab_bench.py config_${MODE} 2>&1
  pkill -9 -f sglang 2>/dev/null; pkill -9 -f launch_server 2>/dev/null; sleep 5
  echo "########## DONE: $MODE ##########"
done
echo "ALL CONFIGS DONE"
