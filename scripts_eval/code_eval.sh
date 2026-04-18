#!/usr/bin/env bash
set -euo pipefail
set -x

export LD_LIBRARY_PATH=/opt/conda/envs/verl/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3
TP_SIZE=4

ROOT=/code/verl_learning

# 请替换为你的代码评测数据 parquet，要求至少包含 prompt 与 reward_model.ground_truth
EVAL_DATA=${ROOT}/data/test/code_test.parquet
MODEL_PATH=${ROOT}/base_models/Qwen3-8B
OUTPUT_DIR=${ROOT}/eval_results/Qwen3-8B/code

# 示例：评测 checkpoint
# MODEL_PATH=${ROOT}/checkpoints/DrGRPO_xxx/global_step_xxx/hf_model
# OUTPUT_DIR=${ROOT}/eval_results/DrGRPO_xxx/code

mkdir -p $OUTPUT_DIR

python /code/verl_learning/eval/eval_code.py \
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
  --eval_workers 8 \
  --output_dir $OUTPUT_DIR
