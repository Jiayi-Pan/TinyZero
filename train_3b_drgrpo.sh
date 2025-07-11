
export N_GPUS=2
export CUDA_VISIBLE_DEVICES=0,1
# ray stop --force && ray start --head --include-dashboard=True --dashboard-port=8263
export BASE_MODEL="/NAS/chenfeng/models/Qwen/Qwen2.5-3B"
export DATA_DIR="/NAS/chenfeng/dataset/countdown"
export ROLLOUT_TP_SIZE=2
export EXPERIMENT_NAME=countdown-qwen2.5-3b-dr-grpo
export VLLM_ATTENTION_BACKEND=XFORMERS

export SWANLAB_API_KEY=YOUR_API_KEY_HERE

# bash ./scripts/train_tiny_zero_a100_drgrpo.sh
nohup bash ./scripts/train_tiny_zero_a100_drgrpo.sh &>./nohupoutput