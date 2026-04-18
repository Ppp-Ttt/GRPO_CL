#!/usr/bin/env bash
set -euo pipefail
set -x

export LD_LIBRARY_PATH=/opt/conda/envs/verl/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3
TP_SIZE=4

ROOT=/code/verl_learning
# EVAL_DATA=${ROOT}/data/aime-2024.parquet
EVAL_DATA=${ROOT}/data/test/math/aime24_math500_olympiad.parquet

# MODEL_PATH=${ROOT}/base_models/Qwen3-8B
# OUTPUT_DIR=${ROOT}/eval_results/Qwen3-8B/math
MODEL_PATH=${ROOT}/checkpoints/DrGRPO/DrGRPO_math_chem_qwen3-8B/global_step_160/hf_model
OUTPUT_DIR=${ROOT}/eval_results/DrGRPO/DrGRPO_math_chem_qwen3-8B_step160/math

mkdir -p $OUTPUT_DIR

python /code/verl_learning/eval/eval_aime24.py \
  --test_file $EVAL_DATA \
  --model $MODEL_PATH \
  --temperature 0.6 \
  --top_p 0.95 \
  --top_k 20 \
  --min_p 0 \
  --max_tokens 8192 \
  --enable_thinking False \
  --tensor_parallel_size $TP_SIZE \
  --dtype bfloat16 \
  --gpu_memory_utilization 0.80 \
  --max_model_len 16384 \
  --output_dir $OUTPUT_DIR
