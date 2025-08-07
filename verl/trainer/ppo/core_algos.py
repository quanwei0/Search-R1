# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config): # seems never used?
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl


def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns

def compute_masked_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float,
    lam: float,
):
    with torch.no_grad():
                
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        
        advantages = torch.zeros_like(token_level_rewards)
        batch_size, gen_len = token_level_rewards.shape

        for b in range(batch_size):
            lastgaelam = 0
            valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]

            for i in range(len(valid_positions) - 1, -1, -1):
                curr_pos = valid_positions[i]
                
                if i != len(valid_positions) - 1:
                    next_pos = valid_positions[i + 1]
                    nextvalues = values[b, next_pos]
                else:
                    nextvalues = 0.0
                
                delta = token_level_rewards[b, curr_pos] + gamma * nextvalues - values[b, curr_pos]
                lastgaelam = delta + gamma * lam * lastgaelam
                advantages[b, curr_pos] = lastgaelam

        advantages = verl_F.masked_whiten(advantages, loss_mask)
    return advantages, returns

def compute_turn_level_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    eos_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float,
    lam: float,
    turn_level_gamma: float,
    turn_level_lam: float,
):
    with torch.no_grad():
        
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        
        turn_level_adv = torch.zeros_like(token_level_rewards)
        batch_size, gen_len = token_level_rewards.shape

        for b in range(batch_size):
            
            turn_end_pos = ((loss_mask[b][1:] == 1) & (loss_mask[b][:-1] == 0)).nonzero(as_tuple=True)[0]
            turn_start_pos = turn_end_pos + 1            
            if loss_mask[b][0] == 1:
                turn_start_pos = torch.cat([torch.tensor([0], device=loss_mask.device), turn_start_pos])
            
            valid_response_length = values[b].nonzero(as_tuple=True)[0].shape[0] - 1
            turn_end_pos = torch.cat([turn_end_pos, torch.tensor([valid_response_length - 1], device=loss_mask.device)])
            
            lastgaelam = 0
            
            # valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            valid_positions = torch.cat([turn_start_pos, torch.tensor([valid_response_length - 1], device=loss_mask.device)])
            
            for i in range(len(valid_positions) - 1, -1, -1):
                if i == 0:
                    break
                last_pos = valid_positions[i-1]
                curr_pos = valid_positions[i]
 
                curr_turn_reward = token_level_rewards[b, curr_pos-1]
                curr_turn_value = values[b, curr_pos]
                 
                # curr_turn_reward = token_level_rewards[b, last_pos:curr_pos].sum()
                # curr_turn_value = values[b, last_pos+1:curr_pos+1].sum()
                
                # curr_turn_reward = token_level_rewards[b, last_pos:curr_pos].mean()
                # curr_turn_value = values[b, last_pos+1:curr_pos+1].mean()
                
                
                if i < len(valid_positions) - 1:
                    next_pos = valid_positions[i+1]
                    nextvalues = values[b, next_pos]
                    # nextvalues = values[b, curr_pos+1:next_pos+1].sum()
                    # nextvalues = values[b, curr_pos+1:next_pos+1].mean()
                else:
                    nextvalues = 0.0
                               
                delta = curr_turn_reward + turn_level_gamma * nextvalues - curr_turn_value
                lastgaelam = delta + turn_level_gamma * turn_level_lam * lastgaelam
                turn_level_adv[b, curr_pos] = lastgaelam

            # each token in the sequence has the same advantage
            for start, end in zip(turn_start_pos, turn_end_pos):
                if end < valid_response_length - 1:
                    adv_value = turn_level_adv[b, end+1]
                else:
                    adv_value = turn_level_adv[b, end]
                for pos in range(start, end + 1):
                    turn_level_adv[b, pos] = adv_value

        advantages = verl_F.masked_whiten(turn_level_adv, loss_mask)
    return advantages, returns

def compute_turn_level_gae_advantage_return_v2(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    eos_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float,
    lam: float,
):
    with torch.no_grad():
        
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
              
        turn_indices = []

        for b in range(loss_mask.size(0)):
            mask = loss_mask[b]
            # valid_response_length = values[b].nonzero(as_tuple=True)[0].shape[0] - 1
            valid_response_length = mask.nonzero(as_tuple=True)[0][-1] + 1


            # Detect where a turn starts: when mask switches from 0 to 1
            turn_end_pos = ((mask[1:] == 1) & (mask[:-1] == 0)).nonzero(as_tuple=True)[0]
            turn_start_pos = turn_end_pos + 1

            # Check if the very first token is part of a turn
            if mask[0] == 1:
                turn_start_pos = torch.cat([torch.tensor([0], device=mask.device), turn_start_pos])

            # Append last token as final turn end if not already included

            turn_end_pos = torch.cat([turn_end_pos, torch.tensor([valid_response_length - 1], device=mask.device)])

            # Build list of (start, end) pairs
            indices = list(zip(turn_start_pos.tolist(), turn_end_pos.tolist()))
            turn_indices.append(indices)
        
        turn_level_adv = torch.zeros_like(advantages)
        
        for b in range(loss_mask.size(0)):
            for start, end in turn_indices[b]:
                turn_level_adv[b, start:end+1] = advantages[b, start:end+1].mean()
        
        # turn_level_adv = advantages.mean(dim=-1, keepdim=True).expand_as(advantages)
        advantages = verl_F.masked_whiten(turn_level_adv, loss_mask)
    return advantages, returns

def compute_weighted_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    eos_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    gamma: float,
    lam: float,
    turn_level_gamma: float,
    turn_level_lam: float,
    turn_level_weight: float = 0.1,
):
    with torch.no_grad():
        
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        
        turn_level_adv = torch.zeros_like(token_level_rewards)
        batch_size, gen_len = token_level_rewards.shape

        for b in range(batch_size):
            
            turn_end_pos = ((loss_mask[b][1:] == 1) & (loss_mask[b][:-1] == 0)).nonzero(as_tuple=True)[0]
            turn_start_pos = turn_end_pos + 1            
            if loss_mask[b][0] == 1:
                turn_start_pos = torch.cat([torch.tensor([0], device=loss_mask.device), turn_start_pos])
            
            valid_response_length = values[b].nonzero(as_tuple=True)[0].shape[0] - 1
            turn_end_pos = torch.cat([turn_end_pos, torch.tensor([valid_response_length - 1], device=loss_mask.device)])
            
            lastgaelam = 0
            
            # valid_positions = loss_mask[b].nonzero(as_tuple=True)[0]
            valid_positions = torch.cat([turn_start_pos, torch.tensor([valid_response_length - 1], device=loss_mask.device)])
            
            for i in range(len(valid_positions) - 1, -1, -1):
                if i == 0:
                    break
                last_pos = valid_positions[i-1]
                curr_pos = valid_positions[i]
 
                curr_turn_reward = token_level_rewards[b, curr_pos-1]
                curr_turn_value = values[b, curr_pos]
                 
                # curr_turn_reward = token_level_rewards[b, last_pos:curr_pos].sum()
                # curr_turn_value = values[b, last_pos+1:curr_pos+1].sum()
                
                # curr_turn_reward = token_level_rewards[b, last_pos:curr_pos].mean()
                # curr_turn_value = values[b, last_pos+1:curr_pos+1].mean()
                
                
                if i < len(valid_positions) - 1:
                    next_pos = valid_positions[i+1]
                    nextvalues = values[b, next_pos]
                    # nextvalues = values[b, curr_pos+1:next_pos+1].sum()
                    # nextvalues = values[b, curr_pos+1:next_pos+1].mean()
                else:
                    nextvalues = 0.0
                               
                delta = curr_turn_reward + turn_level_gamma * nextvalues - curr_turn_value
                lastgaelam = delta + turn_level_gamma * turn_level_lam * lastgaelam
                turn_level_adv[b, curr_pos] = lastgaelam

            # each token in the sequence has the same advantage
            for start, end in zip(turn_start_pos, turn_end_pos):
                if end < valid_response_length - 1:
                    adv_value = turn_level_adv[b, end+1]
                else:
                    adv_value = turn_level_adv[b, end]
                for pos in range(start, end + 1):
                    turn_level_adv[b, pos] = adv_value

        weighted_advantages = (1 - turn_level_weight) * advantages + turn_level_weight * turn_level_adv

        advantages = verl_F.masked_whiten(weighted_advantages, loss_mask)
    return advantages, returns

# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    non_zero_mask = (token_level_rewards != 0)
    scores = (token_level_rewards * non_zero_mask).sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss( 
    old_log_prob,
    log_prob,
    advantages,
    eos_mask,
    cliprange,
    detach_ratio=None,
    importance_sampling_level='token',
    turn_indices=None,
):
    """
    Computes PPO policy loss with optional detached importance sampling ratio.
    If detach_ratio=True, ratio is linearly decayed to 0 outside the clipping range.

    Args:
        old_log_prob: old log probabilities for each token
        log_prob: new log probabilities for each token  
        advantages: advantage values for each token
        eos_mask: mask for valid tokens (completion_mask)
        cliprange: clipping range for PPO
        detach_ratio: detachment strategy ('soft', 'hard', or None)
        importance_sampling_level: importance sampling strategy
            - 'token': each token uses its own importance weight
            - 'sequence': traditional sequence-level where all tokens share the same weight (average over sequence)
            - 'partial_sequence': partial-sequence-level GRPO where each token at position t uses 
                                 cumulative average of log_ratio from position 1 to t
            - 'turn': turn-level where all tokens within the same turn share the same importance weight
        turn_indices: tensor of shape (batch_size, max_indices) containing turn indices,
                     required when importance_sampling_level='turn'

    Returns:
        pg_loss: scalar tensor
        pg_clipfrac: float tensor
        ppo_kl: scalar tensor
    """
    log_ratio = log_prob - old_log_prob
    
    # Compute importance weights based on the specified level
    if importance_sampling_level == "token":
        print("="*80)
        print(f"[Debug] (default) token-level importance sampling")
        print("="*80)
        log_importance_weights = log_ratio
    elif importance_sampling_level == "sequence":
        print("="*80)
        print(f"[Debug] (GSPO) sequence-level importance sampling")
        print("="*80)
        # Traditional sequence-level importance sampling: all tokens share same weight (average over sequence)
        log_importance_weights = (log_ratio * eos_mask).sum(-1) / eos_mask.sum(-1).clamp(min=1.0)
        log_importance_weights = log_importance_weights.unsqueeze(-1)
    elif importance_sampling_level == "partial_sequence":
        print("="*80)
        print(f"[Debug] partial-sequence-level importance sampling")
        print("="*80)
        # Partial-Sequence-level importance sampling: 
        # For each token at position t, use cumulative average of log_ratio from position 1 to t
        # w_{i,t} = exp(1/t * Σ_{s=1}^t log_ratio_s)
        
        batch_size, seq_len = log_ratio.shape
        log_importance_weights = torch.zeros_like(log_ratio)
        
        for b in range(batch_size):
            mask = eos_mask[b]  # mask for this batch
            valid_positions = mask.nonzero(as_tuple=True)[0]  # positions where mask=1
            
            if len(valid_positions) == 0:
                continue
                
            # Extract log_ratios for valid positions only
            valid_log_ratios = log_ratio[b, valid_positions]  # shape: [num_valid_tokens]
            
            # Compute cumulative sum and divide by position indices to get cumulative averages
            cumsum = torch.cumsum(valid_log_ratios, dim=0)  # [num_valid_tokens]
            position_indices = torch.arange(1, len(valid_positions) + 1, 
                                          dtype=cumsum.dtype, device=cumsum.device)  # [1, 2, 3, ...]
            cumulative_averages = cumsum / position_indices  # [num_valid_tokens]
            
            # Assign back to the original positions
            log_importance_weights[b, valid_positions] = cumulative_averages
    elif importance_sampling_level == "turn":
        print("="*80)
        print(f"[Debug] turn-level importance sampling")
        print("="*80)
        # Turn-level importance sampling: all tokens within the same turn share the same importance weight
        # The weight for each turn is the average of log_ratios of all tokens within that turn
        
        if turn_indices is None:
            raise ValueError("turn_indices must be provided when importance_sampling_level='turn'")
        
        batch_size, seq_len = log_ratio.shape
        log_importance_weights = torch.zeros_like(log_ratio)
        
        for b in range(batch_size):
            mask = eos_mask[b]  # mask for this batch
            
            # Get turn indices for this batch sample - stored as flattened [start1, end1, start2, end2, ...]
            turn_indices_b = turn_indices[b]  # shape: (max_indices,)
            
            # Find valid turn indices (not equal to -1)
            valid_indices = turn_indices_b[turn_indices_b != -1]
            
            if len(valid_indices) == 0 or len(valid_indices) % 2 != 0:
                # If no valid indices or odd number of indices, fall back to token-level
                log_importance_weights[b] = log_ratio[b]
                continue
            
            # Parse pairs of (start, end) indices
            num_turns = len(valid_indices) // 2
            
            for turn_idx in range(num_turns):
                start_pos = valid_indices[turn_idx * 2].item()
                end_pos = valid_indices[turn_idx * 2 + 1].item()
                
                # Ensure positions are within sequence bounds
                start_pos = max(0, min(start_pos, seq_len - 1))
                end_pos = max(0, min(end_pos, seq_len - 1))
                
                if start_pos > end_pos:
                    continue
                
                # Get log_ratios for tokens in this turn that are valid (masked)
                turn_mask = mask[start_pos:end_pos + 1]
                turn_log_ratios = log_ratio[b, start_pos:end_pos + 1]
                
                if turn_mask.sum() > 0:
                    # Compute average log_ratio for this turn (only for valid tokens)
                    turn_avg_log_ratio = (turn_log_ratios * turn_mask).sum() / turn_mask.sum()
                    
                    # Assign this average to all tokens in the turn
                    log_importance_weights[b, start_pos:end_pos + 1] = turn_avg_log_ratio

    else:
        raise ValueError(
            f"Unknown importance sampling level: {importance_sampling_level}. Possible values are 'token', "
            "'sequence', 'partial_sequence', and 'turn'."
        )
    
    negative_approx_kl = log_importance_weights
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)
    
    if detach_ratio=='hard':
        print("="*80)
        print(f"[Debug] hard detach ratio with {importance_sampling_level}-level importance sampling and the cliprange is",cliprange)
        print("="*80)
        # Detach ratio but still apply clipping
        ratio_detached = ratio.detach()
        clipped_ratio_detached = torch.clamp(ratio_detached, 1.0 - cliprange, 1.0 + cliprange)

        pg_losses1 = -ratio_detached * advantages * log_prob
        pg_losses2 = -clipped_ratio_detached * advantages * log_prob

        pg_loss = verl_F.masked_mean(torch.max(pg_losses1, pg_losses2), eos_mask)
        pg_clipfrac = verl_F.masked_mean((pg_losses2 > pg_losses1).float(), eos_mask)
    elif detach_ratio=='soft':
        print("="*80)
        print(f"[Debug] soft detach ratio with {importance_sampling_level}-level importance sampling and the cliprange is",cliprange)
        print("="*80)
        ratio_detached = ratio.detach()

        # Define a smooth mask in the range [1 - 2ε, 1 - ε] ∪ [1 + ε, 1 + 2ε]
        lower = 1.0 - cliprange
        upper = 1.0 + cliprange
        lower_decay = 1.0 - 1.05 * cliprange 
        upper_decay = 1.0 + 1.3 * cliprange

        # Create smooth decay mask: multiplier ∈ [1, 0]
        decay_mask = torch.ones_like(ratio_detached)

        # Left decay: [1 - 2ε, 1 - ε]
        left_mask = (ratio_detached >= lower_decay) & (ratio_detached < lower)
        decay_mask[left_mask] = (ratio_detached[left_mask] - lower_decay) / cliprange

        # Right decay: [1 + ε, 1 + 2ε]
        right_mask = (ratio_detached > upper) & (ratio_detached <= upper_decay)
        decay_mask[right_mask] = (upper_decay - ratio_detached[right_mask]) / cliprange

        # Outside full decay region: ratio < 1 - 2ε or > 1 + 2ε → mask = 0
        decay_mask[ratio_detached < lower_decay] = 0.0
        decay_mask[ratio_detached > upper_decay] = 0.0

        # Apply soft decay mask
        effective_ratio = ratio_detached * decay_mask

        pg_losses = -advantages * log_prob * effective_ratio
        pg_loss = verl_F.masked_mean(pg_losses, eos_mask)
        pg_clipfrac = verl_F.masked_mean((decay_mask < 1.0).float(), eos_mask)

    else:
        # Standard PPO
        print("="*80)
        print(f"[Debug] normal detach ratio with {importance_sampling_level}-level importance sampling and the cliprange is",cliprange)
        print("="*80)
        
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

        pg_loss = verl_F.masked_mean(torch.max(pg_losses, pg_losses2), eos_mask)
        pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses).float(), eos_mask)

    return pg_loss, pg_clipfrac, ppo_kl


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
