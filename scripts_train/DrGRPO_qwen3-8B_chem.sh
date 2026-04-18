# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x

ROOT=/code/verl_learning

EXP_NAME=DrGRPO_math_chem_qwen3-8B
WANDB_PROJECT="DrGRPO"

# MODEL_PATH=${ROOT}/base_models/Qwen3-8B
MODEL_PATH=${ROOT}/checkpoints/DrGRPO_OrderI/DrGRPO_OrderI_math_qwen3-8B/global_step_160/hf_model
CKPTS_DIR=${ROOT}/checkpoints/${WANDB_PROJECT}/${EXP_NAME}
TRAIN_DATA_FILES=${ROOT}/data/train/chem/chem-L3.parquet
TEST_DATA_FILES=${ROOT}/data/test/chem/chem-L3.parquet
TRAIN_ROLLOUT_LOG=${ROOT}/rollout_log/${WANDB_PROJECT}/train_${EXP_NAME}
TEST_ROLLOUT_LOG=${ROOT}/rollout_log/${WANDB_PROJECT}/test_${EXP_NAME}

# Avoid cross-NUMA GPU peer-access issues by defaulting to one NVLink island.
# You can override these at runtime, e.g.:
# CUDA_VISIBLE_DEVICES=0,1,2,3 N_GPUS_PER_NODE=4 bash run_qwen3-8b.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3
N_GPUS_PER_NODE=4


cd ${ROOT}/verl
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${TRAIN_DATA_FILES} \
    data.val_files=${TEST_DATA_FILES} \
    data.train_batch_size=64 \
    data.val_batch_size=512 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.seed=2026 \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.00 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-sum-norm" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.60 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.rollout_data_dir=${TRAIN_ROLLOUT_LOG} \
    trainer.validation_data_dir=${TEST_ROLLOUT_LOG} \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.max_actor_ckpt_to_keep=7 \
    trainer.max_critic_ckpt_to_keep=7 \
    trainer.total_epochs=7 \
    trainer.default_local_dir="${CKPTS_DIR}" $@
