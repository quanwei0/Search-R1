# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

from verl import DataProto
import torch
from verl.utils.reward_score import qa_em, qa_em_format, qa_em_new
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
import re
import numpy as np

def _select_rm_score_fn(data_source, reward_type='answer_correctness'):
    if data_source in ['nq', 'triviaqa', 'popqa', 'hotpotqa', '2wikimultihopqa', 'musique', 'bamboogle']:
        if reward_type == 'answer_correctness':
            return qa_em_new.compute_score_em
        elif reward_type == 'format_correctness':
            return qa_em_new.compute_score_format
        elif reward_type == 'retrieval_correctness':
            return qa_em_new.compute_score_retrieval
        elif reward_type == 'mixed_outcome_reward':
            return qa_em_new.compute_score_em_format_retrievel
        elif reward_type == 'final_em_format':
            return qa_em_new.compute_score_final_em_format
        elif reward_type == 'step_retrieval_format':
            return qa_em_new.compute_score_step_retrieval_format
        else:
            raise NotImplementedError(f"Unsupported reward type: {reward_type} for data source: {data_source}")
        
    else:
        raise NotImplementedError


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, format_score=0.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.format_score = format_score

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        answer_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        format_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        retrieval_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        mixed_outcome_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        final_em_format_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        step_retrieval_format_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        avg_step_retrieval_format_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        mixed_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        # all_scores = []

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']
            # decoded_full_texts = data_item.meta_info['decoded_full_texts'][i]
            decoded_turn_texts = data_item.meta_info['decoded_turn_texts'][i]

            
            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_answer_score = _select_rm_score_fn(data_source, reward_type='answer_correctness')
            compute_format_score = _select_rm_score_fn(data_source, reward_type='format_correctness')
            compute_retrieval_score = _select_rm_score_fn(data_source, reward_type='retrieval_correctness')
            comupte_mixed_outcome_score = _select_rm_score_fn(data_source, reward_type='mixed_outcome_reward')
            compute_final_em_format_score = _select_rm_score_fn(data_source, reward_type='final_em_format')
            compute_step_retrieval_format_score = _select_rm_score_fn(data_source, reward_type='step_retrieval_format')

            answer_score = compute_answer_score(solution_str=sequences_str, ground_truth=ground_truth)
            format_score = compute_format_score(solution_str=sequences_str)
            retrieval_score = compute_retrieval_score(solution_str=sequences_str, ground_truth=ground_truth)
            mixed_outcome_score = comupte_mixed_outcome_score(solution_str=sequences_str, ground_truth=ground_truth)
            final_em_format_score = compute_final_em_format_score(final_turn_str=decoded_turn_texts[-1], ground_truth=ground_truth)
            step_retrieval_format_score = compute_step_retrieval_format_score(mid_turn_str=decoded_turn_texts[:-1], ground_truth=ground_truth)

            answer_reward_tensor[i, valid_response_length - 1] = answer_score
            format_reward_tensor[i, valid_response_length - 1] = format_score
            retrieval_reward_tensor[i, valid_response_length - 1] = retrieval_score
            mixed_outcome_reward_tensor[i, valid_response_length - 1] = mixed_outcome_score
            final_em_format_reward_tensor[i, valid_response_length - 1] = final_em_format_score

            for j in range(data.meta_info['num_turns'][i] - 1):
                step_retrieval_format_reward_tensor[i, data.meta_info['turn_indices'][i][j][1]] = step_retrieval_format_score[j]
            
            if data.meta_info['num_turns'][i] - 1 == 0:
                avg_step_retrieval_format_reward_tensor[i, valid_response_length - 1] = 0
            else:
                avg_step_retrieval_format_reward_tensor[i, valid_response_length - 1] = step_retrieval_format_reward_tensor[i, :].sum(dim=-1) / (data.meta_info['num_turns'][i] - 1)
            
            mixed_reward_tensor = final_em_format_reward_tensor + step_retrieval_format_reward_tensor
            

            # all_scores.append(score)

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str)
        
        # print(f"[DEBUG] all_scores: {all_scores}")
        # print(f"[DEBUG] all_scores shape: {np.array(all_scores).shape}")
        # print(f"[DEBUG] all_scores mean: {np.mean(all_scores)}")
        # print(f"[DEBUG] all_scores max: {np.max(all_scores)}")
        # print(f"[DEBUG] all_scores min: {np.min(all_scores)}")
        # print(f"[DEBUG] all_scores std: {np.std(all_scores)}")

        return {
            'answer_correctness': answer_reward_tensor,
            'format_correctness': format_reward_tensor,
            'retrieval_correctness': retrieval_reward_tensor,
            'mixed_outcome_reward': mixed_outcome_reward_tensor,
            'final_em_format': final_em_format_reward_tensor,
            'step_retrieval_format': step_retrieval_format_reward_tensor,
            'avg_step_retrieval_format': avg_step_retrieval_format_reward_tensor,
            'mixed_reward': mixed_reward_tensor,
        }

import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config))


@ray.remote
def main_task(config):
    from verl.utils.fs import copy_local_path_from_hdfs
    from transformers import AutoTokenizer

    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # env_class = ENV_CLASS_MAPPING[config.env.name]

    # download the checkpoint from hdfs
    local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer
    tokenizer = hf_tokenizer(local_path)

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_fn = RewardManager(tokenizer=tokenizer, num_examine=0)

    # Note that we always use function-based RM for validation
    val_reward_fn = RewardManager(tokenizer=tokenizer, num_examine=1)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn,
                            )
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
