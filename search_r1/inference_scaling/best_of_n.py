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
        """
        super().__init__(config)
        self.n_candidates = config.get('n_candidates', 4)
        self.selection_metric = config.get('selection_metric', 'reward')
        self.temperature = config.get('temperature', 1.0)
        self.use_step_rewards = config.get('use_step_rewards', True)
        self.step_reward_weight = config.get('step_reward_weight', 1.0)
        
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
        metrics = self.generation_manager.actor_rollout_wg.compute_log_prob_inference(candidates)

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

        if self.use_step_rewards:
            process_rewards = self._add_step_rewards(candidates)
            final_rewards += self.step_reward_weight * process_rewards

        candidates.meta_info['rewards'] = final_rewards

        candidates = candidates.union(metrics)
        
        return candidates

    def _add_step_rewards(self, candidates):
        """Add step rewards to the current scores.
        
        Args:
            state: Current state with scores
            meta_info: Metadata with generation history containing str_per_round
        """
        batch_size = candidates.batch['prompts'].shape[0]
        pad_token_id = self.generation_manager.tokenizer.pad_token_id  # 151643
        
        process_rewards = torch.zeros(batch_size, device=candidates.batch['prompts'].device)
        
        for i in range(batch_size):
            # Split turns based on responses_with_info_mask
            responses_mask = candidates.batch['responses_with_info_mask'][i]

            # Find turn boundaries by detecting transitions
            turn_str = []
            for start, end in self.turns_from_tokens(responses_mask, pad_token_id):
                turn_tokens = candidates.batch['responses'][i][start:end+1]
                turn_text = self.generation_manager.tokenizer.decode(
                    turn_tokens, skip_special_tokens=True
                )
                turn_str.append(turn_text)

            # Path is ongoing, update score with the newest turn
            if len(turn_str) > 0:
                # Compute final score for the last turn only
                final_score = self.compute_final_format_score(turn_str[-1])
                
                # Compute step scores for all turns except the last
                sum_step_scores = 0.0
                for t in range(1, len(turn_str)):  # From 1 to len(turn_str)-1 turns
                    step_score = self.compute_step_format_score(turn_str[:t])
                    sum_step_scores += step_score

                process_rewards[i] = final_score + sum_step_scores
                
        return process_rewards