#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export DATA_DIR='/home/mhong/zhan9359/work/Search-R1/data/nq_search'

# export WANDB_API_KEY="810f91e58aa0fd1d03b11c60b0d1cffbb1d941f4"
# export WANDB_ENTITY="rl_agent"

WAND_PROJECT='Search-R1'

# export BASE_MODEL='/home/mhong/zhan9359/.cache/models--quanwei0--nq-hotpotqa-search-r1-ppo-qwen2.5-7b-em-gae-maxturn4/snapshots/ee87254eac7617a33dc61a5a4e94beead9a3aab8/actor/global_step_1000'
# export CRITIC_BASE_MODEL='/home/mhong/zhan9359/.cache/models--quanwei0--nq-hotpotqa-search-r1-ppo-qwen2.5-7b-em-gae-maxturn4/snapshots/ee87254eac7617a33dc61a5a4e94beead9a3aab8/critic/global_step_1000'
export BASE_MODEL='/home/mhong/zhan9359/.cache/models--quanwei0--nq-search-r1-ppo-qwen2.5-7b-em-gae-mixed-reward-new7/snapshots/448a8eff359fda6faed1fe7999a96208e3024694/actor/global_step_500'

export CRITIC_BASE_MODEL="/home/mhong/zhan9359/.cache/models--quanwei0--nq-search-r1-ppo-qwen2.5-7b-em-gae-mixed-reward-new7/snapshots/448a8eff359fda6faed1fe7999a96208e3024694/critic/global_step_500"


export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# Best-of-N values to test
beam_VALUES=(3)

# Loop through each Best-of-N value
for N in "${beam_VALUES[@]}"; do
    echo "========================================="
    echo "Running Beam Search with beam=$N"
    echo "========================================="
    n_candidates=$(($N * $N))
    export EXPERIMENT_NAME="nq-search-r1-quan-7b-ckpt1-sampled-512-beamsearch_tree2_beam${N}_budget${n_candidates}"

    VAL_BATCH_SIZE=8
    GPU_MEMORY_UTIL=0.6
    
    # echo "Using train_batch_size=$TRAIN_BATCH_SIZE, val_batch_size=$VAL_BATCH_SIZE"
    
    PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_inference \
        +use_inference_scaling=true \
        +scaling_config.algorithm=beam_search_tree2 \
        +scaling_config.n_candidates=$n_candidates \
        +scaling_config.beam_width=$N \
        +scaling_config.max_turns=3 \
        +scaling_config.selection_metric=critic \
        +scaling_config.temperature=1 \
        data.train_files=$DATA_DIR/train.parquet \
        data.val_files=$DATA_DIR/test.parquet \
        data.train_data_num=null \
        data.val_data_num=null \
        data.train_batch_size=512 \
        data.val_batch_size=$VAL_BATCH_SIZE \
        data.val_data_num=512 \
        data.max_prompt_length=4096 \
        data.max_response_length=500 \
        data.max_start_length=2048 \
        data.max_obs_length=500 \
        data.shuffle_train_dataloader=True \
        algorithm.adv_estimator=gae \
        algorithm.gamma=1 \
        algorithm.lam=1 \
        +algorithm.use_mixed_outcome_reward=False \
        +algorithm.use_mixed_reward=False \
        actor_rollout_ref.model.path=$BASE_MODEL \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size=64 \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.grad_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_micro_batch_size=128 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTIL \
        actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.rollout.n_agent=1 \
        actor_rollout_ref.rollout.temperature=1 \
        actor_rollout_ref.actor.state_masking=True \
        critic.optim.lr=1e-5 \
        critic.model.use_remove_padding=True \
        critic.optim.lr_warmup_steps_ratio=0.015 \
        critic.model.path=$CRITIC_BASE_MODEL \
        critic.model.enable_gradient_checkpointing=True \
        critic.ppo_micro_batch_size=8 \
        critic.model.fsdp_config.param_offload=True \
        critic.model.fsdp_config.grad_offload=True \
        critic.model.fsdp_config.optimizer_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        algorithm.no_think_rl=False \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        +trainer.val_only=True \
        +trainer.val_before_train=True \
        +trainer.is_save_val_traj=True \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=2 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.project_name=$WAND_PROJECT \
        trainer.experiment_name=$EXPERIMENT_NAME \
        trainer.total_epochs=4 \
        trainer.total_training_steps=2000 \
        trainer.default_hdfs_dir=null \
        trainer.default_local_dir=verl_checkpoints/$EXPERIMENT_NAME \
        max_turns=3 \
        retriever.url="http://127.0.0.1:8000/retrieve" \
        retriever.topk=3 \
        2>&1 | tee "${EXPERIMENT_NAME}.log"
    
    echo "Completed Best-of-N with N=$N"
    echo "Results saved to ${EXPERIMENT_NAME}.log"
    echo ""
    
    # Optional: Add a small delay between runs
    sleep 5
done

echo "========================================="
echo "All Best-of-N experiments completed!"
echo "========================================="

# Optional: Summarize results
echo "Summary of experiments:"
for N in "${BON_VALUES[@]}"; do
    LOG_FILE="nq-search-r1-quan-7b-ckpt1-sampled-512-BoN${N}.log"
    if [ -f "$LOG_FILE" ]; then
        echo "BoN-$N: Check $LOG_FILE for results"
    fi
done