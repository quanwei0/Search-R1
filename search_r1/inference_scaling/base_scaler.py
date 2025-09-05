"""
Base class for inference-time scaling algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import torch
import re
from verl import DataProto
from verl.utils.reward_score.qa_em_new import *

class BaseInferenceGenerator(ABC):
    """
    Base class for inference-time scaling algorithms.
    
    Each scaling algorithm should inherit from this and implement the scale_inference method.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the scaler with configuration.
        
        Args:
            config: Dictionary containing scaler-specific configuration
        """
        self.config = config
    
    @abstractmethod
    def generate(self, 
                       generation_manager,
                       gen_batch: DataProto,
                       initial_input_ids: torch.Tensor) -> DataProto:
        """
        Apply inference-time scaling to generation.
        
        Args:
            generation_manager: The LLMGenerationManager instance
            gen_batch: Input generation batch
            initial_input_ids: Initial input token IDs
            
        Returns:
            DataProto: Scaled generation output (may contain multiple candidates)
        """
        pass
    
    def _compute_trajectory_reward(self, candidate: DataProto, reward_fn) -> float:
        """
        Compute reward for a single trajectory.
        
        Args:
            candidate: Single generation candidate
            reward_fn: Reward function
            
        Returns:
            float: Computed reward
        """
        if reward_fn is None:
            return 0.0
        
        try:
            reward_dict = reward_fn(candidate)
            # Use mixed outcome reward as default selection criterion
            return reward_dict.get('mixed_outcome_reward', 
                                 reward_dict.get('answer_correctness', 0.0))
        except Exception as e:
            print(f"Warning: Failed to compute reward: {e}")
            return 0.0
    
    @staticmethod
    def compute_step_format_score(turn_text: str, search_count: int = 0) -> float:
        """
        Compute format score for intermediate step without ground truth.
        
        Args:
            turn_text: Text of the current turn
            search_count: Number of searches performed so far
            
        Returns:
            Format score for the step
        """
        # Check for proper tag structure
        format_score = 0.1 if mid_format_check(turn_text) else -0.2
        
        # Add search penalty
        search_penalty = 0.0
        if "<search>" in turn_text:
            turn_search_count = turn_text.count("<search>")
            search_penalty = -0.1 * (search_count + turn_search_count)
        
        return format_score + search_penalty
    
    @staticmethod
    def compute_final_format_score(final_turn_str: str) -> float:
        """
        Compute format score for final turn without ground truth.
        
        Args:
            final_turn_str: Text of the final turn
            
        Returns:
            Format score for final turn
        """
        if not final_format_check(final_turn_str):
            return -1.0
        else:
            return 0.2  # Good format but no answer
        
    def _split_turn_idx(self, batch: DataProto) -> DataProto:
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        loss_mask = batch.batch['info_mask'][:, -response_length:]       
        # values = batch.batch['values']

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

        # Save to batch meta_info for later use (e.g., in GAE)
        batch.meta_info['turn_indices'] = turn_indices

        batch_size = len(turn_indices)
        max_indices = 20  # Should be enough for most cases (3 turns = max 6 indices, with buffer)
        turn_indices_tensor = torch.full((batch_size, max_indices), -1, dtype=torch.long, device=loss_mask.device)
        
        # Fill in the actual turn indices for each sample
        
        for b, indices in enumerate(turn_indices):
            flattened_indices = []
            for start, end in indices:
                flattened_indices.extend([start, end])
            
            # Fill the tensor with actual indices
            num_indices = min(len(flattened_indices), max_indices)
            turn_indices_tensor[b, :num_indices] = torch.tensor(flattened_indices[:num_indices], dtype=torch.long, device=loss_mask.device)
        
        batch.batch['turn_indices'] = turn_indices_tensor
        
        return batch
    
    def _split_trajectories(self, batch, save_dir: Optional[str] = None, val_batch_idx: Optional[int] = None) -> DataProto:
        """
        Decode full trajectories and per-turn sequences from the batch, and store them
        into batch.meta_info for later use.

        Optionally, save them to disk if `save_dir` is provided.
        """
        full_texts = []
        prompt_texts = []
        turn_texts = []
        trajectories = []
        
        response_text_lengths = [] 
        turn_text_lengths = []
        num_turns = []

        turn_indices = batch.meta_info.get("turn_indices", [[] for _ in range(len(batch))])
        
        for i in range(len(batch)):
            data_item = batch[i]

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            attention_mask = data_item.batch['attention_mask']
            valid_prompt_length = attention_mask[:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            prompt_text = self.tokenizer.decode(valid_prompt_ids)
            prompt_texts.append(prompt_text)
            
            response_ids = data_item.batch['responses']
            valid_response_length = attention_mask[prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            full_ids = torch.cat((valid_prompt_ids, valid_response_ids))
            full_text = self.tokenizer.decode(full_ids)
            full_texts.append(full_text)
            response_text_lengths.append(valid_response_ids.shape[0])

            # Turn-level decoding
            turns = []
            turn_lengths = []
            for start, end in turn_indices[i]:
                turn_ids = response_ids[start:end + 1]
                turn_text = self.tokenizer.decode(turn_ids)
                turns.append(turn_text)
                turn_lengths.append(turn_ids.shape[0])
            turn_texts.append(turns)
            turn_text_lengths.append(turn_lengths)
            num_turns.append(len(turns))