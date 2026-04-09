#!/usr/bin/env bash
set -euo pipefail
set -x

mkdir -p /code/verl_learning/eval_results/aime-2024

export LD_LIBRARY_PATH=/opt/conda/envs/verl/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python /code/verl_learning/eval/eval_aime24.py \
  --test_file /code/verl_learning/data/aime-2024.parquet \
  --model /code/verl_learning/base_models/Qwen3-8B \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --min_p 0 \
  --max_tokens 8192 \
  --enable_thinking false \
  --tensor_parallel_size 8 \
  --dtype bfloat16 \
  --gpu_memory_utilization 0.90 \
  --max_model_len 32768 \
  --output_dir /code/verl_learning/eval_results/aime-2024
