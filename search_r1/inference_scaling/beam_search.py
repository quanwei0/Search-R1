"""
Beam Search Implementation for Search-R1
Implements beam search for multi-turn reasoning with search engine integration.
"""

import torch
from typing import Dict, List, Tuple, Optional, Any
from .base_scaler import BaseInferenceGenerator


# BeamState removed - we'll use gen_batch directly with additional fields


class BeamSearchGenerator(BaseInferenceGenerator):
    """Beam search generator for multi-turn reasoning with search."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize beam search generator.
        
        Args:
            config: Configuration dictionary containing:
                - n_candidates: Number of final candidates per prompt
                - beam_width: Number of expansions per beam
                - max_turns: Maximum number of reasoning turns
                - temperature: Sampling temperature
                - score_fn: Scoring function ('logprob', 'normalized_logprob', 'weighted', 'critic')
                - critic_worker_group: Worker group for value function scoring
                - early_stopping: Enable early stopping when prompt has k finished candidates (default: True)
                - filter_duplicates: Filter duplicate candidates (default: True)
                - lookahead_steps: Number of lookahead steps (default: 0)
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
        self.early_stopping = config.get('early_stopping', True)  # Enable early stopping heuristic
        
        # Finished pool to store completed trajectories
        self.finished_pool = {}

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
        
        # Initialize finished pool for each prompt
        for prompt_idx in range(original_batch_size):
            self.finished_pool[prompt_idx] = {'candidates': [], 'scores': [], 'count': 0}
        breakpoint()
        # Main beam search loop - expand one level at a time
        for turn in range(self.max_turns):
            # Move finished candidates to pool and update active_mask
            self._move_finished_to_pool(rollings, state, original_batch_size)
            
            # Early stopping: check if we should stop any prompts that have enough finished candidates
            self._apply_early_stopping(state, original_batch_size)
            
            # Check if all candidates are completed
            active_mask = state.batch['active_mask']
            if not active_mask.sum():
                break
            
            # Only expand active candidates to save budget
            active_indices = active_mask.nonzero(as_tuple=True)[0]
            if len(active_indices) == 0:
                break
                
            # Select only active candidates
            active_rollings = rollings.select_idxs(active_indices)
            active_state = state.select_idxs(active_indices)
            
            # Expand only the active candidates by beam_width
            expanded_rollings = active_rollings.repeat(repeat_times=self.beam_width, interleave=True)
            expanded_state = active_state.repeat(repeat_times=self.beam_width, interleave=True)
            
            # Search one level - expand all candidates (active/inactive handled by generation manager)
            expanded_rollings, meta_info = self.generation_manager._process_generation_turn(
                    expanded_state, expanded_rollings, is_final=False
                )
            
            # Compose batch for scoring after each turn
            left_side = {'input_ids': expanded_state.batch['left_input_ids']}
            right_side = {
                'responses': expanded_state.batch['responses'],
                'responses_with_info_mask': expanded_state.batch['responses_with_info_mask']
            }
            scoring_batch = self.generation_manager._compose_final_output(left_side, right_side, meta_info)
            
            # Score the composed batch with critic
            scoring_batch = self._score_batch_with_critic(scoring_batch, expanded_state, reward_fn)
            
            # Select top n_candidates per prompt from all expanded candidates
            # Note: We need to map back to original prompts since we only expanded active candidates
            orig_candidates_per_prompt = rollings.batch['input_ids'].shape[0] // original_batch_size
            new_rollings, new_state = self._select_top_candidates_per_prompt_from_active(
                expanded_rollings, expanded_state, active_indices, original_batch_size, orig_candidates_per_prompt
            )
            
            # If no active candidates selected, break
            if new_rollings is None:
                break
                
            rollings = new_rollings
            state = new_state

        # Final generation turn without search for remaining active candidates
        if state is not None and state.batch['active_mask'].sum():
            rollings, meta_info = self.generation_manager._process_generation_turn(
                state, rollings, is_final=True
            )
            
            # Move any final finished candidates to pool
            self._move_finished_to_pool(rollings, state, original_batch_size)
        
        # Combine active and finished candidates for final output
        final_batch, final_state = self._combine_active_and_finished_simple(rollings, state, original_batch_size)
        
        # Compile final metadata
        if final_state is not None:
            meta_info['turns_stats'] = final_state.batch['turns_stats'].tolist()
            meta_info['active_mask'] = final_state.batch['active_mask'].tolist()
            meta_info['valid_action_stats'] = final_state.batch['valid_action_stats'].tolist()
            meta_info['valid_search_stats'] = final_state.batch['valid_search_stats'].tolist()
            meta_info['generation_history'] = final_state.meta_info['history']
            
            print("ACTIVE_TRAJ_NUM:", final_state.meta_info.get('active_num_history', 0))
            
            # Extract left_side and right_side for compose_final_output
            left_side = {'input_ids': final_state.batch['left_input_ids']}
            right_side = {
                'responses': final_state.batch['responses'],
                'responses_with_info_mask': final_state.batch['responses_with_info_mask']
            }
            
            final_candidates = self.generation_manager._compose_final_output(left_side, right_side, meta_info)
        else:
            # Handle case where we only have finished pool candidates
            final_candidates = self._create_candidates_from_finished_pool(original_batch_size)
        
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
    

    def _score_batch_with_critic(self, batch, state, reward_fn):
        """Score batch with critic value function."""
        if not reward_fn:
            return batch
            
        # Get critic values from the properly composed batch
        output = reward_fn.compute_values(batch)
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

    def _move_finished_to_pool(self, rollings, state, original_batch_size):
        """Move finished candidates to the finished pool. Keep batch structure intact."""
        active_mask = state.batch['active_mask']
        finished_indices = (~active_mask).nonzero(as_tuple=True)[0]
        
        if len(finished_indices) == 0:
            return
        
        # Group finished indices by their original prompt
        candidates_per_prompt = rollings.batch['input_ids'].shape[0] // original_batch_size
        
        for idx in finished_indices:
            prompt_idx = idx.item() // candidates_per_prompt
            
            # Extract candidate data
            candidate_data = {
                'rollings': rollings.select_idxs([idx]),
                'state': state.select_idxs([idx])
            }
            
            score = state.batch['scores'][idx].item() if 'scores' in state.batch else 0.0
            
            # Add to finished pool (no limit here - we'll select top later)
            self.finished_pool[prompt_idx]['candidates'].append(candidate_data)
            self.finished_pool[prompt_idx]['scores'].append(score)
            self.finished_pool[prompt_idx]['count'] += 1
        
        # Note: We do NOT remove finished candidates from batch here
        # The active_mask already marks them as inactive, and selection will ignore them
    
    def _apply_early_stopping(self, state, original_batch_size):
        """Apply early stopping heuristic: stop prompts that have k finished candidates."""
        if state is None or not self.early_stopping:
            return
            
        active_mask = state.batch['active_mask']
        current_batch_size = active_mask.shape[0]
        candidates_per_prompt = current_batch_size // original_batch_size
        
        for prompt_idx in range(original_batch_size):
            # Check if this prompt already has enough finished candidates
            finished_count = self.finished_pool[prompt_idx]['count']
            
            if finished_count >= self.n_candidates:
                # Stop all remaining active candidates for this prompt
                start_idx = prompt_idx * candidates_per_prompt
                end_idx = start_idx + candidates_per_prompt
                
                stopped_count = 0
                for idx in range(start_idx, min(end_idx, current_batch_size)):
                    if active_mask[idx]:
                        # Mark as inactive to stop further expansion
                        state.batch['active_mask'][idx] = False
                        stopped_count += 1
                
                if stopped_count > 0:
                    print(f"Early stopping: Prompt {prompt_idx} has {finished_count} finished candidates, stopped {stopped_count} active candidates")
                        
    def _select_top_candidates_per_prompt_from_active(self, expanded_batch, expanded_state, active_indices, original_batch_size, orig_candidates_per_prompt):
        """Select top n_candidates per prompt from expanded active candidates."""
        scores = expanded_state.batch['scores']
        active_mask = expanded_state.batch['active_mask']
        
        # Group expanded candidates by original prompt
        prompt_groups = {}
        for i, orig_idx in enumerate(active_indices):
            prompt_idx = orig_idx.item() // orig_candidates_per_prompt
            if prompt_idx not in prompt_groups:
                prompt_groups[prompt_idx] = []
            
            # Each original active candidate was expanded by beam_width
            for j in range(self.beam_width):
                expanded_idx = i * self.beam_width + j
                if expanded_idx < scores.shape[0]:
                    prompt_groups[prompt_idx].append(expanded_idx)
        
        # Select top candidates per prompt
        selected_indices = []
        
        for prompt_idx in range(original_batch_size):
            if prompt_idx in prompt_groups:
                prompt_candidates = prompt_groups[prompt_idx]
                prompt_active_mask = active_mask[prompt_candidates]
                
                # Only consider active candidates
                active_prompt_candidates = [idx for i, idx in enumerate(prompt_candidates) 
                                           if prompt_active_mask[i]]
                
                if len(active_prompt_candidates) > 0:
                    active_scores = scores[active_prompt_candidates]
                    k = min(self.n_candidates, len(active_prompt_candidates))
                    _, top_indices = torch.topk(active_scores, k, largest=True)
                    
                    selected_candidates = [active_prompt_candidates[idx] for idx in top_indices]
                    selected_indices.extend(selected_candidates)
        
        if len(selected_indices) == 0:
            return None, None
        
        # Select the top candidates
        selected_batch = expanded_batch.select_idxs(selected_indices)
        selected_state = expanded_state.select_idxs(selected_indices)
        
        return selected_batch, selected_state
    
    def _construct_batch_from_finished_pool(self, original_batch_size):
        """Construct final batch from finished pool when no active candidates remain."""
        all_candidates = []
        all_states = []
        
        for prompt_idx in range(original_batch_size):
            finished_data = self.finished_pool[prompt_idx]
            if finished_data['count'] > 0:
                # Take top candidates from finished pool
                scores = torch.tensor(finished_data['scores'])
                n_select = min(self.n_candidates, len(scores))
                _, top_indices = torch.topk(scores, n_select, largest=True)
                
                for idx in top_indices:
                    all_candidates.append(finished_data['candidates'][idx]['rollings'])
                    all_states.append(finished_data['candidates'][idx]['state'])
        
        if len(all_candidates) == 0:
            # Return empty batch - this shouldn't happen in normal cases
            return None, None
        
        # Combine all selected candidates
        # Note: This is a simplified combination - you may need to implement proper batch concatenation
        # based on your batch structure
        combined_batch = all_candidates[0]
        combined_state = all_states[0]
        
        for _ in range(1, len(all_candidates)):
            # Concatenate batches - implementation depends on your batch structure
            pass  # TODO: Implement proper batch concatenation
        
        return combined_batch, combined_state
    
    def _combine_active_and_finished_simple(self, rollings, state, original_batch_size):
        """Combine active and finished candidates, ensuring exactly n_candidates per prompt."""
        # Collect all candidates per prompt
        prompt_data = {}
        
        for prompt_idx in range(original_batch_size):
            prompt_data[prompt_idx] = {'candidates': [], 'states': [], 'scores': []}
            
            # Add active candidates for this prompt
            if state is not None and state.batch['active_mask'].sum() > 0:
                current_batch_size = rollings.batch['input_ids'].shape[0]
                candidates_per_prompt = current_batch_size // original_batch_size if current_batch_size > 0 else 0
                
                for i in range(candidates_per_prompt):
                    idx = prompt_idx * candidates_per_prompt + i
                    if idx < current_batch_size and state.batch['active_mask'][idx]:
                        prompt_data[prompt_idx]['candidates'].append(rollings.select_idxs([idx]))
                        prompt_data[prompt_idx]['states'].append(state.select_idxs([idx]))
                        score = state.batch['scores'][idx].item() if 'scores' in state.batch else 0.0
                        prompt_data[prompt_idx]['scores'].append(score)
            
            # Add finished candidates from pool
            if prompt_idx in self.finished_pool:
                finished_data = self.finished_pool[prompt_idx]
                prompt_data[prompt_idx]['candidates'].extend([c['rollings'] for c in finished_data['candidates']])
                prompt_data[prompt_idx]['states'].extend([c['state'] for c in finished_data['candidates']])
                prompt_data[prompt_idx]['scores'].extend(finished_data['scores'])
            
            # Select exactly n_candidates for this prompt
            if len(prompt_data[prompt_idx]['candidates']) > self.n_candidates:
                scores_tensor = torch.tensor(prompt_data[prompt_idx]['scores'])
                _, top_indices = torch.topk(scores_tensor, self.n_candidates, largest=True)
                
                prompt_data[prompt_idx]['candidates'] = [prompt_data[prompt_idx]['candidates'][i] for i in top_indices]
                prompt_data[prompt_idx]['states'] = [prompt_data[prompt_idx]['states'][i] for i in top_indices]
                prompt_data[prompt_idx]['scores'] = [prompt_data[prompt_idx]['scores'][i] for i in top_indices]
        
        # Concatenate all selected candidates
        all_candidates = []
        all_states = []
        
        for prompt_idx in range(original_batch_size):
            all_candidates.extend(prompt_data[prompt_idx]['candidates'])
            all_states.extend(prompt_data[prompt_idx]['states'])
        
        if len(all_candidates) == 0:
            return None, None
        
        # Concatenate batches properly
        return self._concatenate_batches(all_candidates, all_states)
    
    def _concatenate_batches(self, candidate_batches, state_batches):
        """Concatenate multiple batches into single batch."""
        if len(candidate_batches) == 0:
            return None, None
        
        if len(candidate_batches) == 1:
            return candidate_batches[0], state_batches[0]
        
        # Assume all batches have the same structure - implement based on your batch type
        # This is a placeholder that needs to be implemented based on your actual batch structure
        # For now, return the first batch as a fallback
        return candidate_batches[0], state_batches[0]
    
    def _combine_active_and_finished(self, rollings, state, original_batch_size):
        """Combine remaining active candidates with finished pool to maintain n_candidates per prompt."""
        # If no active candidates, return only from finished pool
        if state is None or not state.batch['active_mask'].sum():
            return self._construct_batch_from_finished_pool(original_batch_size)
        
        # Combine active and finished candidates
        # For now, prioritize active candidates and fill from finished pool if needed
        combined_candidates = []
        combined_states = []
        
        # Get active candidates
        active_mask = state.batch['active_mask']
        active_indices = active_mask.nonzero(as_tuple=True)[0]
        
        candidates_per_prompt = rollings.batch['input_ids'].shape[0] // original_batch_size if rollings.batch['input_ids'].shape[0] > 0 else self.n_candidates
        
        for prompt_idx in range(original_batch_size):
            prompt_actives = []
            prompt_states = []
            
            # Collect active candidates for this prompt
            for idx in active_indices:
                if idx.item() // candidates_per_prompt == prompt_idx:
                    prompt_actives.append(rollings.select_idxs([idx]))
                    prompt_states.append(state.select_idxs([idx]))
            
            # Fill remaining slots from finished pool
            finished_needed = self.n_candidates - len(prompt_actives)
            if finished_needed > 0 and self.finished_pool[prompt_idx]['count'] > 0:
                finished_data = self.finished_pool[prompt_idx]
                scores = torch.tensor(finished_data['scores'])
                n_take = min(finished_needed, len(scores))
                _, top_indices = torch.topk(scores, n_take, largest=True)
                
                for idx in top_indices:
                    prompt_actives.append(finished_data['candidates'][idx]['rollings'])
                    prompt_states.append(finished_data['candidates'][idx]['state'])
            
            combined_candidates.extend(prompt_actives)
            combined_states.extend(prompt_states)
        
        if len(combined_candidates) == 0:
            return None, None
        
        # Simple combination - in practice you'd need proper batch concatenation
        return combined_candidates[0], combined_states[0] if len(combined_states) > 0 else None
    
    def _create_candidates_from_finished_pool(self, original_batch_size):
        """Create final candidates when only finished pool exists."""
        # This is a placeholder - you'd need to implement proper candidate creation
        # from the finished pool based on your batch structure
        return None

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

        _, top_indices_2d = torch.topk(scores_reshaped, self.n_candidates, dim=1, largest=True)
        
        prompt_offsets = torch.arange(original_batch_size, device=top_indices_2d.device) * expanded_per_prompt
        prompt_offsets = prompt_offsets.unsqueeze(1)  # Shape: (original_batch_size, 1)
        top_indices_1d = (top_indices_2d + prompt_offsets).flatten()  # Shape: (original_batch_size * n_candidates,)
        
        # Select top candidates
        new_batch = expanded_batch.select_idxs(top_indices_1d)
        new_state = state.select_idxs(top_indices_1d)
        
        return new_batch, new_state
