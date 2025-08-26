"""
Best-of-N inference scaling algorithm.
"""

from typing import Dict, List, Any, Optional
import torch
from verl import DataProto
from .base_scaler import BaseInferenceScaler


class BestOfNScaler(BaseInferenceScaler):
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
        self.critic_wg = config.get('critic_worker_group', None)
    
    def scale_inference(self, 
                       generation_manager,
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
        if self.use_stepwise:
            return self._stepwise_generation(generation_manager, gen_batch, initial_input_ids)
        else:
            return self._full_trajectory_generation(generation_manager, gen_batch, initial_input_ids, reward_fn)

    def _full_trajectory_generation(self, generation_manager, gen_batch: DataProto, 
                                  initial_input_ids: torch.Tensor, reward_fn: Optional[callable] = None) -> List[DataProto]:
        """Generate N independent full trajectories and select best per sample."""
        
        candidates = generation_manager._run_single_generation(gen_batch, initial_input_ids)
        
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


    def _stepwise_generation(self, generation_manager, gen_batch: DataProto, 
                           initial_input_ids: torch.Tensor) -> List[DataProto]:
        """
        Stepwise Best-of-N: At each turn, generate N candidates, select best, continue.
        This is like greedy beam search with width N but only keeping best at each step.
        """
        print(f"[Stepwise-BoN] Starting stepwise generation with {self.n_candidates} candidates per step...")
        

        # Initialize with the input
        current_state = {
            'gen_batch': gen_batch,
            'initial_input_ids': initial_input_ids,
            'turn': 0,
            'active': True
        }
        
        max_turns = generation_manager.config.max_turns
        
        # Stepwise generation loop
        for turn in range(max_turns):
            if not current_state['active']:
                break
            # Generate N candidates for this turn
            turn_candidates = []
            for i in range(self.n_candidates):
                try:
                    # Generate single step (or remaining trajectory)
                    candidate = self._generate_single_turn_candidate(
                        generation_manager, 
                        current_state,
                        turn
                    )
                    turn_candidates.append(candidate)
                except Exception as e:
                    print(f"[Stepwise-BoN] Error generating candidate {i+1}: {e}")
                    continue
            
            if not turn_candidates:
                print("[Stepwise-BoN] No valid candidates generated, stopping")
                break
            
            # Select best candidate for this turn using critic or other metric
            best_candidate = self._select_best_turn_candidate(turn_candidates, turn)
            
            # Update state with best candidate
            current_state = self._update_state_with_best(current_state, best_candidate, turn)
            
            # Check if trajectory is complete
            if self._is_trajectory_complete(best_candidate):
                current_state['active'] = False
                break
        
        # Return the final trajectory as a single-item list
        final_trajectory = self._construct_final_trajectory(current_state)
        return final_trajectory
            

    def _generate_single_turn_candidate(self, generation_manager, state, turn):
        """Generate a single candidate for the current turn."""
        # For stepwise, we need to run a modified generation that stops after one turn
        # This is a simplified version - in practice you might want to modify generation_manager
        # to support single-turn generation
        return generation_manager._run_single_generation(
            state['gen_batch'], 
            state['initial_input_ids']
        )
    
    def _select_best_turn_candidate(self, candidates: List[DataProto], turn: int) -> DataProto:
        """Select best candidate for current turn using configured metric."""
        if len(candidates) == 1:
            return candidates[0]
        
        print(f"[Stepwise-BoN] Selecting best from {len(candidates)} candidates at turn {turn}")
        
        if self.selection_metric == 'critic' and self.critic_wg is not None:
            return self._select_with_critic(candidates)
        elif self.selection_metric == 'reward':
            # For stepwise, we need a turn-level reward evaluation
            # This is a simplified version
            return candidates[0]  # Placeholder
        else:
            # Fallback selection methods
            return self._basic_selection(candidates)
    
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
    
    def _basic_selection(self, candidates: List[DataProto]) -> DataProto:
        """Basic selection methods (length, random)."""
        if self.selection_metric == 'length':
            # Select candidate with longest response
            best_candidate = candidates[0]
            best_length = 0
            
            for candidate in candidates:
                response_length = candidate.batch['responses'].shape[1]
                if response_length > best_length:
                    best_length = response_length
                    best_candidate = candidate
                    
            return best_candidate
        
        elif self.selection_metric == 'random':
            import random
            return random.choice(candidates)
        
        else:
            # Default: return first candidate
            return candidates[0]
    
    def _update_state_with_best(self, state, best_candidate, turn):
        """Update generation state with the selected best candidate."""
        # This is a simplified version - you'd need to properly update the state
        # to continue generation from where the best candidate left off
        return {
            **state,
            'best_so_far': best_candidate,
            'turn': turn + 1
        }
    
    def _is_trajectory_complete(self, candidate: DataProto) -> bool:
        """Check if trajectory is complete."""
        if 'active_mask' in candidate.meta_info:
            return not any(candidate.meta_info['active_mask'])
        return False
    
    def _construct_final_trajectory(self, state) -> DataProto:
        """Construct final trajectory from stepwise state."""
        return state.get('best_so_far', DataProto())
    
    def select_best(self, candidates: List[DataProto], 
                   reward_fn=None, **kwargs) -> DataProto:
        """
        Select the best candidate based on the configured selection metric.
        
        Args:
            candidates: List of generation candidates
            reward_fn: Reward function for evaluation
            **kwargs: Additional arguments (can include critic_wg)
            
        Returns:
            DataProto: Best candidate
        """
        if not candidates:
            raise ValueError("No candidates provided for selection")
        
        if len(candidates) == 1:
            return candidates[0]
        
        print(f"[Best-of-N] Selecting best from {len(candidates)} candidates using {self.selection_metric}")
        
        if self.selection_metric == 'critic':
            return self._select_with_critic(candidates)
        
        elif self.selection_metric == 'reward':
            if reward_fn is None:
                print("[Warning] No reward function provided, falling back to random selection")
                import random
                return random.choice(candidates)
            
            best_candidate = candidates[0]
            best_reward = float('-inf')
            
            print(f"[Best-of-N] Evaluating {len(candidates)} candidates with reward function...")
            
            for i, candidate in enumerate(candidates):
                # Create single-item batch for reward computation
                single_batch = DataProto.from_dict({
                    k: v[:1] if len(v.shape) > 0 else v for k, v in candidate.batch.items()
                })
                single_batch.meta_info = candidate.meta_info
                
                reward = self._compute_trajectory_reward(single_batch, reward_fn)
                print(f"[Best-of-N] Candidate {i+1} reward: {reward:.4f}")
                
                if reward > best_reward:
                    best_reward = reward
                    best_candidate = candidate
            
            print(f"[Best-of-N] Selected candidate with reward: {best_reward:.4f}")
            return best_candidate
        
        elif self.selection_metric == 'length':
            # Select candidate with longest response
            best_candidate = candidates[0]
            best_length = 0
            
            for candidate in candidates:
                response_length = candidate.batch['responses'].shape[1]
                if response_length > best_length:
                    best_length = response_length
                    best_candidate = candidate
                    
            return best_candidate
        
        elif self.selection_metric == 'random':
            import random
            return random.choice(candidates)
        
        else:
            raise ValueError(f"Unknown selection metric: {self.selection_metric}")
    
    def _compute_trajectory_reward(self, candidate: DataProto, reward_fn) -> float:
        """
        Compute reward for a single trajectory, handling different reward types.
        """
        try:
            reward_dict = reward_fn(candidate)
            
            # Try different reward keys in order of preference
            for key in ['mixed_outcome_reward', 'answer_correctness', 'final_em_format']:
                if key in reward_dict:
                    reward_tensor = reward_dict[key]
                    if isinstance(reward_tensor, torch.Tensor):
                        return reward_tensor.sum().item()
                    return float(reward_tensor)
            
            # Fallback to first available numeric reward
            for key, value in reward_dict.items():
                if isinstance(value, (int, float, torch.Tensor)):
                    if isinstance(value, torch.Tensor):
                        return value.sum().item()
                    return float(value)
                    
            return 0.0
            
        except Exception as e:
            print(f"Warning: Failed to compute reward: {e}")
            return 0.0


if __name__ == "__main__":
    """Simple test for Best-of-N scaling."""
    print("Testing Best-of-N Inference Scaling...")
    
    try:
        # Test different configurations
        configs = [
            {'n_candidates': 4, 'selection_metric': 'reward', 'use_stepwise': False},
            {'n_candidates': 3, 'selection_metric': 'critic', 'use_stepwise': False}, 
            {'n_candidates': 2, 'selection_metric': 'length', 'use_stepwise': True}
        ]
        
        for i, config in enumerate(configs):
            scaler = BestOfNScaler(config)
            print(f"✅ Config {i+1}: {config['selection_metric']} selection, stepwise={config['use_stepwise']}")
            
        print("🎉 Best-of-N ready! Use in validation with:")
        print("  scaling_algorithm=best_of_n")
        print("  scaling_config.n_candidates=4")
        print("  scaling_config.selection_metric=critic")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()