cd /code/verl_learning/verl
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /code/verl_learning/checkpoints/DrGRPO/DrGRPO_math_chem_qwen3-8B/global_step_130/actor \
    --target_dir /code/verl_learning/checkpoints/DrGRPO/DrGRPO_math_chem_qwen3-8B/global_step_130/hf_model
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /code/verl_learning/checkpoints/DrGRPO/DrGRPO_math_chem_qwen3-8B/global_step_120/actor \
    --target_dir /code/verl_learning/checkpoints/DrGRPO/DrGRPO_math_chem_qwen3-8B/global_step_120/hf_model



