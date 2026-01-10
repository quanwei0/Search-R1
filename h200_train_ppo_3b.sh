#!/bin/bash

# Parse command line arguments
CUDA_DEVICES=${1:-"0,1,2,3"}
RETRIEVAL_PORT=${2:-8001}

echo "Using CUDA devices: $CUDA_DEVICES"
echo "Using retrieval port: $RETRIEVAL_PORT"

source /mnt/data1/wei00355/miniconda/bin/activate
conda init

# Set shared configuration parameters
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES
export RETRIEVAL_PORT=$RETRIEVAL_PORT

conda activate retriever
# Pass GPU devices and port to retrieval script
bash retrieval_launch.sh "$CUDA_VISIBLE_DEVICES" "$RETRIEVAL_PORT" &
sleep 60

conda activate searchr1

export DATA_DIR='./data/nq_hotpotqa_train'

export WANDB_API_KEY="810f91e58aa0fd1d03b11c60b0d1cffbb1d941f4"
export WANDB_ENTITY="rl_agent"

WAND_PROJECT='Search-R1-mixed-data'

REWARD_TYPE='turn_reward'

# export BASE_MODEL='Qwen/Qwen2.5-1.5B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-1.5b-em-gae
# export BASE_MODEL='Qwen/Qwen2.5-1.5B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-1.5b-it-em
# export BASE_MODEL='Qwen/Qwen2.5-3B'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-3b-em-gae
# export BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-3b-it-em
export BASE_MODEL='Qwen/Qwen2.5-3B'
EXPERIMENT_NAME=mixed-data-qwen2.5-3b-ppo-${REWARD_TYPE//_/-}-new7-maxturn4
export EXPERIMENT_NAME=qw-$EXPERIMENT_NAME-$(date +%Y%m%d-%H%M%S)
# export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
# export EXPERIMENT_NAME=nq-search-r1-ppo-qwen2.5-7b-it-em

# set -x
export VLLM_ATTENTION_BACKEND=XFORMERS # vllm + qwen2-7b with flash_attn has some issues

# max_prompt_length = (config['training']['max_start_length'] + config['training']['max_response_length'] * (config['training']['max_turns'] - 1) + config['training']['max_obs_length'] * config['training']['max_turns'])

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/test.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=512 \
    data.val_batch_size=256 \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=500 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=gae \
    algorithm.gamma=1 \
    algorithm.lam=1 \
    +algorithm.reward_type=$REWARD_TYPE \
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
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=128 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path=$BASE_MODEL \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_micro_batch_size=8 \
    critic.model.fsdp_config.param_offload=True \
    critic.model.fsdp_config.grad_offload=True \
    critic.model.fsdp_config.optimizer_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    +trainer.val_only=False \
    +trainer.val_before_train=False \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=40 \
    trainer.total_training_steps=1000 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=/home/wei00355/mnt_data1/Search-R1/verl_checkpoints/$EXPERIMENT_NAME \
    max_turns=4 \
    retriever.url="http://127.0.0.1:$RETRIEVAL_PORT/retrieve" \
    retriever.topk=3 \
    2>&1 | tee ./outputs/log/$EXPERIMENT_NAME.log