cd /code/verl_learning/verl
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /code/verl_learning/checkpoints/DrGRPO_new/DrGRPO_new_math_chem_bio_code_qwen3-8B/global_step_560/actor \
    --target_dir /code/verl_learning/checkpoints/DrGRPO_new/DrGRPO_new_math_chem_bio_code_qwen3-8B/global_step_560/hf_model

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /code/verl_learning/checkpoints/DrGRPO_new/DrGRPO_new_math_chem_bio_code_qwen3-8B/global_step_640/actor \
    --target_dir /code/verl_learning/checkpoints/DrGRPO_new/DrGRPO_new_math_chem_bio_code_qwen3-8B/global_step_640/hf_model

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /code/verl_learning/checkpoints/DrGRPO_new/DrGRPO_new_math_chem_bio_code_qwen3-8B/global_step_720/actor \
    --target_dir /code/verl_learning/checkpoints/DrGRPO_new/DrGRPO_new_math_chem_bio_code_qwen3-8B/global_step_720/hf_model


