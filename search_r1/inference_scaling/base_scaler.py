"""
Base class for inference-time scaling algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import torch
from verl import DataProto


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
