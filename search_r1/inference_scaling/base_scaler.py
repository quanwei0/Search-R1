"""
Base class for inference-time scaling algorithms.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any
import torch
from verl import DataProto


class BaseInferenceScaler(ABC):
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
    def scale_inference(self, 
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
    
    @abstractmethod
    def select_best(self, candidates: List[DataProto], 
                   reward_fn=None, **kwargs) -> DataProto:
        """
        Select the best candidate from multiple generations.
        
        Args:
            candidates: List of generation candidates
            reward_fn: Optional reward function for selection
            **kwargs: Additional selection criteria
            
        Returns:
            DataProto: Best candidate
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


if __name__ == "__main__":
    """Simple test for BaseInferenceScaler interface."""
    print("Testing BaseInferenceScaler Interface...")
    
    try:
        # Test abstract class cannot be instantiated
        try:
            BaseInferenceScaler({})
            print("❌ Should not be able to instantiate abstract class")
        except TypeError:
            print("✅ Abstract class properly enforced")
        
        # Test that subclasses work
        class TestScaler(BaseInferenceScaler):
            def scale_inference(self, generation_manager, gen_batch, initial_input_ids):
                return []
            def select_best(self, candidates, reward_fn=None, **kwargs):
                return candidates[0] if candidates else None
        
        config = {'test': 'value'}
        test_scaler = TestScaler(config)
        assert test_scaler.config == config
        print("✅ Subclass implementation works")
        
        print("🎉 BaseInferenceScaler interface ready!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()