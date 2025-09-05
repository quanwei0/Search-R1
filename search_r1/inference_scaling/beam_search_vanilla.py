"""
Beam Search Implementation for Search-R1
Implements beam search for multi-turn reasoning with search engine integration.
"""

import torch
from typing import Dict, List, Tuple, Optional, Any
from .base_scaler import BaseInferenceGenerator
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

# BeamState removed - we'll use gen_batch directly with additional fields


class BeamSearchVanillaGenerator(BaseInferenceGenerator):
    """Beam search generator for multi-turn reasoning with search."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize beam search generator.
        
        Args:
            config: Configuration dictionary containing:
                - beam_size: Number of beams to maintain
                - max_turns: Maximum number of reasoning turns
                - temperature: Sampling temperature
                - diversity_penalty: Penalty for similar beams
                - score_fn: Scoring function ('logprob', 'normalized_logprob', 'weighted', 'critic')
                - critic_worker_group: Worker group for value function scoring
        """
        super().__init__(config)
        self.n_candidates = config.get('n_candidates', 4)
        self.beam_width = config.get('beam_width', 2)  # How many expansions per beam
        self.budget = self.n_candidates * self.beam_width
        self.max_turns = config.get('max_turns', 10)
        self.temperature = config.get('temperature', 1.0)
        
        # Scoring parameters
        self.score_fn = config.get('score_fn', 'critic')  # Default to critic scoring
        self.critic_wg = config.get('critic_worker_group', None)  # Critic worker group for scoring
        # Beam search specific parameters
        self.filter_duplicates = config.get('filter_duplicates', True)
        self.lookahead_steps = config.get('lookahead_steps', 0)

    def generate(
        self,
        generation_manager,
        gen_batch,
        initial_input_ids: torch.Tensor,
        reward_fn: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Generate using beam search working directly with gen_batch.
        
        Args:
            generation_manager: The LLMGenerationManager instance
            gen_batch: Initial generation batch
            initial_input_ids: Initial input token IDs
            reward_fn: Optional reward function for final scoring
            
        Returns:
            Generation candidates with critic scores (similar to best_of_n)
        """
        self.generation_manager = generation_manager
        
        # Initialize gen_batch with additional beam search fields
        batch_size = gen_batch.batch['input_ids'].shape[0]
        state = self.generation_manager.create_generation_state(batch_size, initial_input_ids, self.generation_manager.config.max_start_length)
        rollings = gen_batch
        meta_info = {}
        
        # Track original batch size for proper candidate selection
        # gen_batch is already repeated n_candidates times
        original_batch_size = batch_size // self.n_candidates
        
        # Main beam search loop - expand one level at a time
        for turn in range(self.max_turns):
            # Check if all candidates are completed
            if not state.batch['active_mask'].sum():
                break
                
            # Repeat each candidate beam_width times
            # Note: This includes both active and inactive beams, but inactive ones
            # will be handled properly by _process_generation_turn and execute_predictions
            rollings = rollings.repeat(repeat_times=self.beam_width, interleave=True)
            state = state.repeat(repeat_times=self.beam_width, interleave=True)
            
            # Search one level - expand all active candidates
            rollings, meta_info = self.generation_manager._process_generation_turn(
                    state, rollings, is_final=False
                )
            
            # Compose batch for scoring after each turn
            left_side = {'input_ids': state.batch['left_input_ids']}
            right_side = {
                'responses': state.batch['responses'],
                'responses_with_info_mask': state.batch['responses_with_info_mask']
            }
            scoring_batch = self.generation_manager._compose_final_output(left_side, right_side, meta_info)
            
            # Score the composed batch with critic
            scoring_batch = self._score_batch_with_critic(scoring_batch, state, reward_fn)
            
            # Select top candidates based on scores
            rollings, state = self._select_top_candidates(rollings, state, original_batch_size)

        # Final generation turn without search
        if state.batch['active_mask'].sum():
            rollings, meta_info = self.generation_manager._process_generation_turn(
                state, rollings, is_final=True
            )
        
        # Compile final metadata
        meta_info['turns_stats'] = state.batch['turns_stats'].tolist()
        meta_info['active_mask'] = state.batch['active_mask'].tolist()
        meta_info['valid_action_stats'] = state.batch['valid_action_stats'].tolist()
        meta_info['valid_search_stats'] = state.batch['valid_search_stats'].tolist()
        meta_info['generation_history'] = state.meta_info['history']
        
        print("ACTIVE_TRAJ_NUM:", state.meta_info['active_num_history'])
        
        # Extract left_side and right_side for compose_final_output
        left_side = {'input_ids': state.batch['left_input_ids']}
        right_side = {
            'responses': state.batch['responses'],
            'responses_with_info_mask': state.batch['responses_with_info_mask']
        }
        
        final_candidates = self.generation_manager._compose_final_output(left_side, right_side, meta_info)
        
        if self.n_candidates == 1:
            return final_candidates
        
        # Apply critic scoring like in best_of_n
        if reward_fn:
            output = reward_fn.compute_values(final_candidates)
            values = output.batch['values']  # Shape: (batch_size, seq_len)
            
            # Get final rewards like in best_of_n
            final_rewards = []
            for b in range(values.shape[0]):
                non_zero_indices = (values[b] != 0).nonzero(as_tuple=True)[0]
                if len(non_zero_indices) > 0:
                    last_idx = non_zero_indices[-1]
                    final_rewards.append(values[b, last_idx])
                else:
                    final_rewards.append(torch.tensor(0.0, device=values.device))
            
            final_rewards = torch.stack(final_rewards)
            final_candidates.meta_info['n_candidates'] = self.n_candidates
            final_candidates.meta_info['rewards'] = final_rewards
        
        return final_candidates
    
    def batch_score(self, batch, reward_fn):
        """Score batch with critic value function."""
        if not reward_fn:
            return batch
        # Pad before critic
        batch_padded, pad_size = pad_dataproto_to_divisor(batch, reward_fn.world_size)

        output = reward_fn.compute_values(batch_padded)

        # Trim critic output back to original
        if pad_size:
            output = unpad_dataproto(output, pad_size)

        return output

    def _score_batch_with_critic(self, batch, state, reward_fn):
        """Score batch with critic value function."""
        if not reward_fn:
            return batch
            
        # Get critic values from the properly composed batch
        output = self.batch_score(batch, reward_fn)
        values = output.batch['values']  # Shape: (batch_size, seq_len)
        
        # Update scores in state
        for i in range(values.shape[0]):
            # Get final non-zero value as score
            non_zero_indices = (values[i] != 0).nonzero(as_tuple=True)[0]
            if len(non_zero_indices) > 0:
                last_idx = non_zero_indices[-1]
                critic_score = values[i, last_idx].item()
                state.batch['scores'][i] = critic_score
            else:
                state.batch['scores'][i] = 0.0

        return batch

    def _select_top_candidates(self, expanded_batch, state, original_batch_size):
        """Select top n_candidates candidates per original prompt.
        
        Args:
            expanded_batch: Batch after expansion with shape 
                (original_batch_size * n_candidates * beam_width, ...)
            state: State with scores matching expanded_batch size
            original_batch_size: Number of original prompts before any repetition
            
        Returns:
            Selected batch and state with shape (original_batch_size * n_candidates, ...)
        """
        scores = state.batch['scores']
        total_size = scores.shape[0]
        
        expanded_per_prompt = total_size // original_batch_size
        
        scores_reshaped = scores.view(original_batch_size, expanded_per_prompt)

        top_scores, top_indices_2d = torch.topk(scores_reshaped, self.n_candidates, dim=1, largest=True)
        
        prompt_offsets = torch.arange(original_batch_size, device=top_indices_2d.device) * expanded_per_prompt
        prompt_offsets = prompt_offsets.unsqueeze(1)  # Shape: (original_batch_size, 1)
        top_indices_1d = (top_indices_2d + prompt_offsets).flatten()  # Shape: (original_batch_size * n_candidates,)
        
        # Select top candidates
        new_batch = expanded_batch.select_idxs(top_indices_1d)
        new_state = state.select_idxs(top_indices_1d)
        
        return new_batch, new_state