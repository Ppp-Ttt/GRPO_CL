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
