"""
Best-of-N inference scaling algorithm.
"""

from typing import Dict, List, Any, Optional
import torch
from verl import DataProto
from .base_scaler import BaseInferenceGenerator


class BestOfNGenerator(BaseInferenceGenerator):
    """
    Best-of-N sampling: Generate N independent full trajectories and select the best one.
    
    Two modes:
    1. Full trajectory BoN: Generate N complete multi-turn trajectories, select best
    2. Stepwise BoN: At each turn, generate N candidates, select best, continue
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Best-of-N scaler.
        
        Expected config:
            n_candidates: Number of candidates to generate (default: 4)
            selection_metric: Metric for selection ('reward', 'critic', 'length', 'random')
            temperature: Sampling temperature for generation (default: 1.0)
            use_stepwise: Whether to use stepwise BoN (default: False)
            critic_worker_group: Optional critic worker group for 'critic' selection
        """
        super().__init__(config)
        self.n_candidates = config.get('n_candidates', 4)
        self.selection_metric = config.get('selection_metric', 'reward')
        self.temperature = config.get('temperature', 1.0)
        self.use_stepwise = config.get('use_stepwise', False)
        
    def generate(self, generation_manager,
                       gen_batch: DataProto,
                       initial_input_ids: torch.Tensor,
                       reward_fn: Optional[callable] = None) -> List[DataProto]:
        """
        Generate N candidates using full trajectory or stepwise approach.
        
        Args:
            generation_manager: LLMGenerationManager instance
            gen_batch: Input generation batch  
            initial_input_ids: Initial input token IDs
            
        Returns:
            List[DataProto]: N generation candidates
        """

        """Generate N independent full trajectories and select best per sample."""
        
        self.generation_manager = generation_manager
        
        candidates = self.generation_manager._run_single_generation(gen_batch, initial_input_ids)
        
        if self.n_candidates == 1:
            return candidates
        
        output = reward_fn.compute_values(candidates)
        values = output.batch['values']  # Shape: (batch_size, seq_len)
        
        # Get the final non-zero value for each sample
        final_rewards = []
        for b in range(values.shape[0]):
            # Find last non-zero value in this sequence
            non_zero_indices = (values[b] != 0).nonzero(as_tuple=True)[0]
            if len(non_zero_indices) > 0:
                last_idx = non_zero_indices[-1]
                final_rewards.append(values[b, last_idx])
            else:
                final_rewards.append(torch.tensor(0.0, device=values.device))
        
        # Stack into tensor
        final_rewards = torch.stack(final_rewards)  # Shape: (batch_size * n_samples,)
        
        candidates.meta_info['n_candidates'] = self.n_candidates
        candidates.meta_info['rewards'] = final_rewards

        return candidates

    def _select_with_critic(self, candidates: List[DataProto]) -> DataProto:
        """Select best candidate using critic model."""
        if self.critic_wg is None:
            print("[Warning] Critic worker group not provided, falling back to first candidate")
            return candidates[0]
        
        best_candidate = candidates[0]
        best_value = float('-inf')
        
        print(f"[Critic Selection] Evaluating {len(candidates)} candidates...")
        
        for i, candidate in enumerate(candidates):
            try:
                # Compute critic value for this candidate
                values_output = self.critic_wg.compute_values(candidate)
                
                # Extract value score (usually the last token value or mean)
                values = values_output.batch['values']  # Shape: (batch_size, seq_len)
                
                # Use mean value as selection criterion
                mean_value = values.mean().item()
                
                print(f"[Critic Selection] Candidate {i+1} value: {mean_value:.4f}")
                
                if mean_value > best_value:
                    best_value = mean_value
                    best_candidate = candidate
                    
            except Exception as e:
                print(f"[Critic Selection] Error evaluating candidate {i+1}: {e}")
                continue
        
        print(f"[Critic Selection] Selected candidate with value: {best_value:.4f}")
        return best_candidate
    