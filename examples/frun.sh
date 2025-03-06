set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    data.train_files=/workspace/TinyZeroGRPO/data/countdown/train.parquet \
    data.val_files=/workspace/TinyZeroGRPO/data/countdown/test.parquet \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    # # data.return_raw_input_ids=False \
    # # data.return_raw_chat=False \
    # # actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    # # actor_rollout_ref.model.external_lib=null \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    # # actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    # actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    # actor_rollout_ref.actor.grad_clip=1.0 \
    # actor_rollout_ref.actor.clip_ratio=0.2 \
    # actor_rollout_ref.actor.entropy_coeff=0.001 \
    # actor_rollout_ref.actor.ppo_epochs=1 \
    # actor_rollout_ref.actor.shuffle=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    # actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0 \
    # actor_rollout_ref.actor.optim.warmup_style=constant \
    # actor_rollout_ref.actor.optim.total_training_steps=-1 \
    # actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    # actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params=0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=32 \
    # actor_rollout_ref.rollout.n=1 \
    # actor_rollout_ref.rollout.name=vllm \
    # actor_rollout_ref.rollout.temperature=1.0 \
    # actor_rollout_ref.rollout.top_k=-1 \
    # actor_rollout_ref.rollout.top_p=1 \
    # actor_rollout_ref.rollout.response_length='${data.max_response_length}' \
    # actor_rollout_ref.rollout.dtype=float16 \
    # actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    # actor_rollout_ref.rollout.ignore_eos=False \
    # actor_rollout_ref.rollout.enforce_eager=True \
    # actor_rollout_ref.rollout.free_cache_engine=True \
    # actor_rollout_ref.rollout.load_format=dummy_dtensor \
    # actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    # actor_rollout_ref.rollout.max_num_seqs=512 \
    # actor_rollout_ref.rollout.log_prob_micro_batch_size=32 \
    # actor_rollout_ref.rollout.do_sample=True \
    # # data.tokenizer=null \
    # # data.prompt_key=prompt 
    # algorithm.gamma=1.0 \
    # algorithm.lam=1.0 \
    algorithm.adv_estimator=grpo \
    # algorithm.kl_penalty=kl \
    # algorithm.kl_ctrl.type=fixed \
    algorithm.kl_ctrl.kl_coef=0.001 \
    # trainer.total_epochs=1 \
    # trainer.project_name=rented_test \
    # trainer.experiment_name=test \
    trainer.logger=['console'] \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.critic_warmup=0 $@