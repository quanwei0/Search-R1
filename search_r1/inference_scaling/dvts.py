"""
DVTS (Diverse Via Tree Search) Implementation for Search-R1
Implements tree-structured beam search where each beam maintains its own candidates.
Each of the n_candidates beams expands to beam_width children and selects the best
child locally, ensuring diversity by keeping beams independent.
"""

import torch
from typing import Dict, List, Tuple, Optional, Any
from .base_scaler import BaseInferenceGenerator
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


class DVTSGenerator(BaseInferenceGenerator):
    """DVTS generator for multi-turn reasoning with beam-local selection."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DVTS generator.
        
        Args:
            config: Configuration dictionary containing:
                - n_candidates: Number of independent beams to maintain
                - beam_width: Number of expansions per beam (children per node)
                - max_turns: Maximum number of reasoning turns
                - temperature: Sampling temperature
                - score_fn: Scoring function ('logprob', 'normalized_logprob', 'weighted', 'critic')
                - critic_worker_group: Worker group for value function scoring
        """
        super().__init__(config)
        self.n_candidates = config.get('n_candidates', 4)
        self.beam_width = config.get('beam_width', 2)  # Children per beam
        self.max_turns = config.get('max_turns', 10)
        self.temperature = config.get('temperature', 1.0)
        
        # Scoring parameters
        self.score_fn = config.get('score_fn', 'critic')  # Default to critic scoring
        self.critic_wg = config.get('critic_worker_group', None)  # Critic worker group for scoring
        
        # DVTS specific parameters
        self.filter_duplicates = config.get('filter_duplicates', True)
        self.local_selection = True  # Always true for DVTS - select best within each beam

    def generate(
        self,
        generation_manager,
        gen_batch,
        initial_input_ids: torch.Tensor,
        reward_fn: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Generate using DVTS with beam-local selection.
        
        Each beam independently expands to beam_width children and selects 
        the best one locally, maintaining diversity across beams.
        
        Args:
            generation_manager: The LLMGenerationManager instance
            gen_batch: Initial generation batch
            initial_input_ids: Initial input token IDs
            reward_fn: Optional reward function for final scoring
            
        Returns:
            Generation candidates with critic scores (n_candidates per prompt)
        """
        self.generation_manager = generation_manager
        
        # Initialize gen_batch with additional DVTS fields
        batch_size = gen_batch.batch['input_ids'].shape[0]
        state = self.generation_manager.create_generation_state(
            batch_size, initial_input_ids, 
            self.generation_manager.config.max_start_length
        )
        rollings = gen_batch
        meta_info = {}
        
        # Track original batch size for proper candidate selection
        # gen_batch is already repeated n_candidates times
        original_batch_size = batch_size // self.n_candidates
        
        # Main DVTS loop - expand and select locally within each beam
        for turn in range(self.max_turns):
            # Check if all candidates are completed
            if not state.batch['active_mask'].sum():
                break
                
            # Expand each beam to beam_width children
            # This creates n_candidates * beam_width total candidates
            rollings = rollings.repeat(repeat_times=self.beam_width, interleave=True)
            state = state.repeat(repeat_times=self.beam_width, interleave=True)
            
            # Generate one turn for all expanded candidates
            rollings, meta_info = self.generation_manager._process_generation_turn(
                state, rollings, is_final=False
            )
            
            # Compose batch for scoring after each turn
            left_side = {'input_ids': state.batch['left_input_ids']}
            right_side = {
                'responses': state.batch['responses'],
                'responses_with_info_mask': state.batch['responses_with_info_mask']
            }
            scoring_batch = self.generation_manager._compose_final_output(
                left_side, right_side, meta_info
            )
            
            # Score the composed batch with critic
            scoring_batch = self._score_batch_with_critic(
                scoring_batch, state, reward_fn
            )
            
            # DVTS: Select best child locally within each beam
            rollings, state = self._select_best_per_beam(
                rollings, state, original_batch_size
            )

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
        
        final_candidates = self.generation_manager._compose_final_output(
            left_side, right_side, meta_info
        )
        
        if self.n_candidates == 1:
            return final_candidates
        
        # Apply critic scoring like in best_of_n
        if reward_fn:
            output = self.batch_score(final_candidates, reward_fn)
            values = output.batch['values']  # Shape: (batch_size, seq_len)
            
            # Get final rewards
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
        batch_padded, pad_size = pad_dataproto_to_divisor(
            batch, reward_fn.world_size
        )

        output = reward_fn.compute_values(batch_padded)

        # Trim critic output back to original
        if pad_size:
            output = unpad_dataproto(output, pad_size)

        return output

    def _score_batch_with_critic(self, batch, state, reward_fn):
        """Score batch with critic value function.
        
        Only updates scores for active beams (active_mask == 1).
        Completed beams keep their final scores.
        """
        if not reward_fn:
            return batch
        active_mask = state.batch['active_mask']
        
        # Find indices of active beams that need scoring
        active_indices = torch.where(active_mask == 1)[0]
        
        if len(active_indices) == 0:
            # No active beams to score
            return batch
        
        # Extract only active beams for scoring
        active_batch = batch.select_idxs(active_indices)
        
        # Score only the active beams
        active_output = self.batch_score(active_batch, reward_fn)
        active_values = active_output.batch['values']  # Shape: (num_active, seq_len)
        
        # Update scores for active beams
        for idx, beam_idx in enumerate(active_indices):
            # Get final non-zero value as score
            non_zero_indices = (active_values[idx] != 0).nonzero(as_tuple=True)[0]
            if len(non_zero_indices) > 0:
                last_idx = non_zero_indices[-1]
                critic_score = active_values[idx, last_idx].item()
                state.batch['scores'][beam_idx] = critic_score
            else:
                state.batch['scores'][beam_idx] = 0.0
        
        # Completed beams keep their final scores
        
        return batch

    def _select_best_per_beam(self, expanded_batch, state, original_batch_size):
        """DVTS selection: Select best child locally within each beam.
        
        For each of the n_candidates beams, select the best child among its 
        beam_width expansions. This ensures each beam maintains its own 
        independent search path.
        
        Args:
            expanded_batch: Batch after expansion with shape 
                (original_batch_size * n_candidates * beam_width, ...)
            state: State with scores matching expanded_batch size
            original_batch_size: Number of original prompts before any repetition
            
        Returns:
            Selected batch and state with shape (original_batch_size * n_candidates, ...)
        """
        scores = state.batch['scores']
        active_mask = state.batch['active_mask']
        total_size = scores.shape[0]
        
        # Total candidates per original prompt
        candidates_per_prompt = self.n_candidates * self.beam_width
        
        # Build selection indices
        all_indices = []
        
        for prompt_idx in range(original_batch_size):
            prompt_offset = prompt_idx * candidates_per_prompt
            
            # For each beam in this prompt
            for beam_idx in range(self.n_candidates):
                # Due to interleave=True in repeat, the children of beam_idx are at:
                # beam_idx * beam_width, beam_idx * beam_width + 1, ..., beam_idx * beam_width + beam_width - 1
                beam_start = prompt_offset + beam_idx * self.beam_width
                beam_end = beam_start + self.beam_width
                beam_indices = torch.arange(beam_start, beam_end, device=scores.device)
                
                # Get scores and active mask for this beam's children
                beam_scores = scores[beam_indices]
                beam_active = active_mask[beam_indices]
                
                # Check if the original beam (first child) is completed
                if beam_active[0] == 0:
                    # Original beam is completed, keep it
                    selected_idx = beam_indices[0]
                else:
                    # Select best active child within this beam
                    # Set inactive children's scores to -inf for selection
                    selection_scores = beam_scores.clone()
                    selection_scores[beam_active == 0] = float('-inf')
                    
                    # Find best child
                    if (selection_scores != float('-inf')).any():
                        best_child_idx = torch.argmax(selection_scores)
                        selected_idx = beam_indices[best_child_idx]
                    else:
                        # Fallback: no active children, keep the first one
                        selected_idx = beam_indices[0]
                
                all_indices.append(selected_idx)
        
        # Convert to tensor
        selected_indices = torch.tensor(all_indices, device=scores.device)
        
        # Select the best child from each beam
        new_batch = expanded_batch.select_idxs(selected_indices)
        new_state = state.select_idxs(selected_indices)
        
        return new_batch, new_state