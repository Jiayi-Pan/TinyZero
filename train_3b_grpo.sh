#!/bin/bash
# alias python='/home/weiji/anaconda3/envs/zero/bin/python'
# alias python3='/home/weiji/anaconda3/envs/zero/bin/python3'
# alias pip='/home/weiji/anaconda3/envs/zero/bin/pip'

export N_GPUS=1
export CUDA_VISIBLE_DEVICES=2
# ray stop --force && ray start --head --include-dashboard=True
export BASE_MODEL="model/Qwen2.5-3B"
export DATA_DIR="data/countdown"
export ROLLOUT_TP_SIZE=1
export EXPERIMENT_NAME=countdown-qwen2.5-3b-grpo
export VLLM_ATTENTION_BACKEND=XFORMERS

bash ./scripts/train_tiny_zero_a100_grpo.sh