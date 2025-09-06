"""
Beam Search Implementation for Search-R1
Implements beam search for multi-turn reasoning with search engine integration.
"""

import torch
import re
from typing import Dict, List, Tuple, Optional, Any
from .base_scaler import BaseInferenceGenerator
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

# BeamState removed - we'll use gen_batch directly with additional fields


class BeamSearchGenerator(BaseInferenceGenerator):
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
        
        # Beam search specific parameters
        self.filter_duplicates = config.get('filter_duplicates', True)
        self.lookahead_steps = config.get('lookahead_steps', 0)
        
        # Step reward weights
        self.use_step_rewards = config.get('use_step_rewards', True)
        self.step_reward_weight = config.get('step_reward_weight', 1.0)

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

            # Add step rewards if enabled
            if self.use_step_rewards:
                self._add_step_rewards(state, current_turn_id=turn)
            state.batch['Q_values'] = state.batch['scores'] + state.batch['process_rewards']
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
        
        # Apply critic scoring
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
            
            if self.use_step_rewards:
                self._add_step_rewards(state, current_turn_id=self.max_turns)
                final_candidates.meta_info['process_rewards'] = state.batch['process_rewards']
                
            final_candidates.meta_info['n_candidates'] = self.n_candidates
            final_candidates.meta_info['rewards'] = final_rewards + state.batch['process_rewards']
        
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

    def _select_top_candidates(self, expanded_batch, state, original_batch_size):
        """Select top n_candidates candidates per original prompt.
        
        Keeps completed beams (active_mask == 0) from the original n_candidates and 
        selects top-k only among active beams (both original and expanded).
        
        Args:
            expanded_batch: Batch after expansion with shape 
                (original_batch_size * n_candidates * beam_width, ...)
            state: State with scores matching expanded_batch size
            original_batch_size: Number of original prompts before any repetition
            
        Returns:
            Selected batch and state with shape (original_batch_size * n_candidates, ...)
        """
        scores = state.batch['Q_values']
        active_mask = state.batch['active_mask']
        total_size = scores.shape[0]
        
        expanded_per_prompt = total_size // original_batch_size  # n_candidates * beam_width
        
        # Reshape for per-prompt processing
        scores_reshaped = scores.view(original_batch_size, expanded_per_prompt)
        active_reshaped = active_mask.view(original_batch_size, expanded_per_prompt)
        
        # Build selection indices
        all_indices = []
        
        for i in range(original_batch_size):
            prompt_offset = i * expanded_per_prompt
            
            # The first n_candidates indices are the original beams (before expansion)
            # Indices [0, n_candidates) are original beam 0, [n_candidates, 2*n_candidates) are original beam 1, etc.
            # Due to interleave=True in repeat, the pattern is:
            # [orig_0_copy_0, orig_0_copy_1, ..., orig_1_copy_0, orig_1_copy_1, ...]
            
            # Since interleave=True, the original beams are at indices:
            # 0, beam_width, 2*beam_width, ..., (n_candidates-1)*beam_width
            original_beam_indices = torch.arange(self.n_candidates, device=scores.device) * self.beam_width
            
            # Check which original beams are completed
            original_active = active_reshaped[i][original_beam_indices]
            completed_original_mask = (original_active == 0)
            
            # Indices of completed original beams (in expanded space)
            completed_indices = original_beam_indices[completed_original_mask] + prompt_offset
            num_completed = len(completed_indices)
            
            if num_completed >= self.n_candidates:
                # All slots filled with completed beams, keep first n_candidates completed
                selected_indices = completed_indices[:self.n_candidates]
            else:
                # Need to select (n_candidates - num_completed) from all beams
                num_to_select = self.n_candidates - num_completed
                
                # Mask out completed original beams by setting their scores to -inf
                # (we want to keep them separately, not compete in top-k)
                prompt_scores = scores_reshaped[i].clone()
                prompt_scores[original_beam_indices[completed_original_mask]] = float('-inf')
                
                # Select top beams from remaining (active original + all expanded)
                if (prompt_scores != float('-inf')).any():
                    # Get more candidates than needed to filter out -inf
                    top_scores, top_indices = torch.topk(prompt_scores, min(expanded_per_prompt, expanded_per_prompt), largest=True)
                    # Filter out -inf scores and take only what we need
                    valid_mask = top_scores != float('-inf')
                    top_indices = top_indices[valid_mask][:num_to_select]
                    active_indices = top_indices + prompt_offset
                else:
                    # Fallback: no active beams, take first available
                    active_indices = torch.arange(num_to_select, device=scores.device) + prompt_offset
                
                # Combine completed and active indices
                if num_completed > 0:
                    selected_indices = torch.cat([completed_indices, active_indices])[:self.n_candidates]
                else:
                    selected_indices = active_indices[:self.n_candidates]
            
            all_indices.append(selected_indices)
        
        # Flatten all indices
        top_indices_1d = torch.cat(all_indices)
        
        # Select top candidates
        new_batch = expanded_batch.select_idxs(top_indices_1d)
        new_state = state.select_idxs(top_indices_1d)
        
        return new_batch, new_state

    def _add_step_rewards(self, state, current_turn_id):
        """Add step rewards to the current scores.
        
        Args:
            state: Current state with scores
            meta_info: Metadata with generation history containing str_per_round
            current_turn_id: The current turn number (0-indexed)
        """
        batch_size = state.batch['scores'].shape[0]
        pad_token_id = self.generation_manager.tokenizer.pad_token_id  # 151643
        current_turn_id += 1  # Convert to 1-indexed for easier comparison

        for i in range(batch_size):
            # Split turns based on responses_with_info_mask
            responses_mask = state.batch['responses_with_info_mask'][i]
            
            # Find turn boundaries by detecting transitions
            turn_str = []
            for start, end in self.turns_from_tokens(responses_mask, pad_token_id):
                turn_tokens = state.batch['responses'][i][start:end+1]
                turn_text = self.generation_manager.tokenizer.decode(
                    turn_tokens, skip_special_tokens=True
                )
                turn_str.append(turn_text)
            
            # Check if trajectory is already finished
            if len(turn_str) < current_turn_id:
                # Trajectory already finished, don't update score
                continue
            
            # Path is ongoing, update score with the newest turn
            if len(turn_str) > 0:
                newest_turn = turn_str[-1]  # Get the latest turn
                
                # Check if this is final turn (has <answer> tag)
                is_final = '<answer>' in newest_turn and '</answer>' in newest_turn
                
                if is_final:
                    # Compute final turn score
                    final_score = self.compute_final_format_score(newest_turn)
                    state.batch['process_rewards'][i] += self.step_reward_weight * final_score
                else:
                    # Count search operations in the newest turn
                    search_count = ''.join(turn_str[:-1]).count("<search>") if len(turn_str) > 1 else 0
                    # Compute step format score
                    step_score = self.compute_step_format_score(newest_turn, search_count)
                    state.batch['process_rewards'][i] += self.step_reward_weight * step_score
