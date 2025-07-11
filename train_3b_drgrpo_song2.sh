#!/bin/bash
# alias python='/home/weiji/anaconda3/envs/zero/bin/python'
# alias python3='/home/weiji/anaconda3/envs/zero/bin/python3'
# alias pip='/home/weiji/anaconda3/envs/zero/bin/pip'

### SONG2

export N_GPUS=1
export CUDA_VISIBLE_DEVICES=0
ray stop --force && ray start --head --include-dashboard=True --dashboard-port=8264
export BASE_MODEL="/NAS/chenfeng/models/Qwen/Qwen2.5-3B"
export DATA_DIR="/NAS/chenfeng/dataset/countdown"
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-3b-dr.grpo
export VLLM_ATTENTION_BACKEND=XFORMERS

export SWANLAB_API_KEY=bcqlEvGQToqnDhv9def0X

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
set TORCH_USE_CUDA_DSA=1

bash ./scripts/train_tiny_zero_a100_drgrpo_song2.sh