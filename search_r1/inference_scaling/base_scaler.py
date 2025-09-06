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
    def compute_step_format_score(turn_list: list) -> float:
        """
        Compute format score for intermediate step without ground truth.
        
        Args:
            turn_text: Text of the current turn
            search_count: Number of searches performed so far
            
        Returns:
            Format score for the step
        """
        # Check for proper tag structure
        format_score = 0.1 if mid_format_check(turn_list[-1]) else -0.2

        # Add search penalty
        search_count = 0
        for turn_text in turn_list:
            if "<search>" in turn_text:
                search_count += 1
        search_penalty = -0.1 * search_count

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
        
    def turns_from_tokens(self, tokens, pad_id):
        """
        Returns a list of (start_idx, end_idx) ranges where each turn is:
        [non-pad tokens] + [immediately following pad tokens].
        Leading pads are skipped.
        """
        n = len(tokens)
        i = 0
        turns = []

        # skip leading pads (if any)
        while i < n and tokens[i] == pad_id:
            i += 1

        while i < n:
            start = i

            # consume non-pad stretch
            while i < n and tokens[i] != pad_id:
                i += 1

            # consume following pads (belong to the same turn)
            while i < n and tokens[i] == pad_id:
                i += 1

            end = i - 1
            turns.append((start, end))

        return turns