## 环境配置
```python
git clone https://github.com/Ppp-Ttt/GRPO_CL.git

conda create -n verl python=3.12 -y
conda activate verl

# vllm运行环境配置, 该脚本内可能无法一次性顺利完成, 建议手动逐条运行install_vllm_sglang_mcore.sh中内容
cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh

# 可能问题1: flash-attention组件可能下载缓慢, 建议手动下载上传
wget https://github.com/Dao-AILab/flash-at tention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.8cxx11abiFALSE-cp312-cp3 12-linux_x86_64.whl
# 可能问题2:出现 numpy=2.4.4 与 numba 不兼容问题, 将numpy回退至2.0~2.3版本间, >2.0的要求来自opencv-python
pip install -U numpy==2.2.6
# 可能问题3: 出现outlines=1.2.12与outlines-core=0.2.11不兼容问题(后者是vllm=0.11.0要求)，或直接安装outlines=1.2.9
pip install "outlines<1.2.12" outlines-core==0.2.11

# verl依赖安装完成后
pip install --no-deps -e .
```

## 数据集下载
```

```

## 训练脚本
位于```scripts_train```文件夹下
需要修改的参数：
```
ROOT=/code/verl_learning # 修改为你的路径
EXP_NAME=DrGRPO_new_math_chem_bio_code_qwen3-8B # 本次实验名
WANDB_PROJECT="DrGRPO_new" # 本次项目名

MODEL_PATH=${ROOT}/checkpoints/DrGRPO_new/DrGRPO_new_math_chem_bio_qwen3-8B/global_step_380/hf_model # 导入权重路径
CKPTS_DIR=${ROOT}/checkpoints/${WANDB_PROJECT}/${EXP_NAME} # 权重保存路径

TRAIN_ROLLOUT_LOG=${ROOT}/rollout_log/${WANDB_PROJECT}/train_${EXP_NAME} # 训练rollout保存文件，可设置为None以取消保存
TEST_ROLLOUT_LOG=${ROOT}/rollout_log/${WANDB_PROJECT}/test_${EXP_NAME}


    trainer.save_freq=40 \ # 保存频率
    trainer.test_freq=20 \ # eval频率
    trainer.max_actor_ckpt_to_keep=5 \ #ckpt最大保存数量，为5会保存最近五个ckpt
    trainer.max_critic_ckpt_to_keep=5 \
    trainer.total_training_steps=800  \ # 总训练steps

```

目前默认使用DrGRPO训练，通过如下设置切换：
```
# 标准GRPO
actor_rollout_ref.actor.loss_agg_mode="token-mean"
actor_rollout_ref.actor.use_kl_loss=True
algorithm.norm_adv_by_std_in_grpo=True
actor_rollout_ref.actor.kl_loss_coef=0.001 # 控制kl系数
actor_rollout_ref.actor.kl_loss_type=k3 # kl计算方式，GRPO为k3估计

#DrGRPO
actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-sum-norm"
actor_rollout_ref.actor.use_kl_loss=False
algorithm.norm_adv_by_std_in_grpo=False
```

将KL约束修改为JSD约束：
在```/verl/trainer/ppo/core_algos.py:line 2152```kl_penalty_forward中新增JSD约束计算方式, 并修改```actor_rollout_ref.actor.kl_loss_type=jsd```
