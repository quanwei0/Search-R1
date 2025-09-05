import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3
    # Inference scaling configuration
    use_inference_scaling: bool = False
    scaling_config: dict = None

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))
        
        # Initialize inference scaling if enabled
        self.inference_scaler = None
        if config.use_inference_scaling:
            self._init_inference_scaler()
    
    def _init_inference_scaler(self):
        """Initialize the appropriate inference scaling algorithm."""
        scaling_config = self.config.scaling_config or {}
        if scaling_config.algorithm == "bon":
            from ..inference_scaling import BestOfNGenerator
            self.inference_scaler = BestOfNGenerator(scaling_config)
        elif scaling_config.algorithm == "beam_search":
            from ..inference_scaling import BeamSearchGenerator 
            self.inference_scaler = BeamSearchGenerator(scaling_config)
        elif scaling_config.algorithm == "beam_search_vanilla":
            from ..inference_scaling import BeamSearchVanillaGenerator  
            self.inference_scaler = BeamSearchVanillaGenerator(scaling_config)
        elif scaling_config.algorithm == "beam_search_tree":
            from ..inference_scaling import BeamSearchTreeGenerator  
            self.inference_scaler = BeamSearchTreeGenerator(scaling_config)
        elif scaling_config.algorithm == "beam_search_tree2":
            from ..inference_scaling import BeamSearchTree2Generator  
            self.inference_scaler = BeamSearchTree2Generator(scaling_config)
        elif scaling_config.algorithm == "dvts":
            from ..inference_scaling import DVTSGenerator  
            self.inference_scaler = DVTSGenerator(scaling_config)
        elif scaling_config.algorithm == "mcts":
            from ..inference_scaling import MCTSGenerator
            self.inference_scaler = MCTSGenerator(scaling_config)
        else:
            raise ValueError(f"Unknown scaling algorithm: {scaling_config.algorithm}")

        print(f"[Generation] Initialized {scaling_config.algorithm} with config: {scaling_config}")
        
    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> torch.Tensor:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = [resp.split('</search>')[0] + '</search>'
                 if '</search>' in resp 
                 else resp.split('</answer>')[0] + '</answer>'
                 if '</answer>' in resp 
                 else resp
                 for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""

        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print(f"[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG, {next_obs_ids.shape[1]} & {self.config.max_obs_length}")            
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Concatenate and handle padding
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            rollings.batch['input_ids'],
            cur_responses,
            next_obs_ids
        ])

        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)

        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)

        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None) -> Dict:
        """Update right side state."""
        if next_obs_ids != None:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask = self._info_masked_concatenate_with_padding(
                    right_side['responses'],
                    right_side['responses_with_info_mask'],
                    cur_responses,
                    pad_to_left=False
                )
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        return {'responses': responses[:, :max_len], 'responses_with_info_mask': responses_with_info_mask[:, :max_len]}

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus

        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)

        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}

        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] for k, v in padded_output.batch.items()}

        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size]
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta

        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor, reward_fn=None) -> Tuple[Dict, Dict]:
        """
        Run main LLM generation loop with optional inference scaling.
        
        Args:
            gen_batch: Generation batch
            initial_input_ids: Initial input token IDs
            reward_fn: Optional reward function for candidate selection
            
        Returns:
            Generation output (single best candidate if scaling is used)
        """
        # Use inference scaling if enabled
        if self.config.use_inference_scaling and self.inference_scaler is not None:
            results = self.inference_scaler.generate(
                generation_manager=self,
                gen_batch=gen_batch,
                initial_input_ids=initial_input_ids,
                reward_fn=reward_fn,
            )
        else:
            results = self._run_single_generation(gen_batch, initial_input_ids)
        # Original generation logic
        return results
    
    def create_generation_state(self, batch_size, initial_input_ids, max_start_length):
        """Create a generation state using DataProto."""
        state = DataProto.from_dict({
            # Left side - prompt
            'left_input_ids': initial_input_ids[:, -max_start_length:],
            
            # Right side - responses
            'responses': initial_input_ids[:, []],
            'responses_with_info_mask': initial_input_ids[:, []],
            
            # Statistics
            'active_mask': torch.ones(batch_size, dtype=torch.bool),
            'turns_stats': torch.ones(batch_size, dtype=torch.int),
            'valid_action_stats': torch.zeros(batch_size, dtype=torch.int),
            'valid_search_stats': torch.zeros(batch_size, dtype=torch.int),
            
            # For beam search
            'scores': torch.zeros(batch_size, dtype=torch.float),
            'process_rewards': torch.zeros(batch_size, dtype=torch.float),
            'completed': torch.zeros(batch_size, dtype=torch.bool)
        })
        
        # Initialize meta_info
        state.meta_info['active_num_history'] = [state.batch['active_mask'].sum().item()]
        state.meta_info['history'] = {
            'ids_per_round': [],
            'str_per_round': []
        }
        
        return state
    
    def update_generation_stats(self, state, curr_active_mask, valid_action, is_search):
        """Update statistics in the generation state."""
        state.batch['active_mask'] = state.batch['active_mask'] * curr_active_mask
        state.meta_info['active_num_history'].append(state.batch['active_mask'].sum().item())
        state.batch['turns_stats'][curr_active_mask] += 1
        state.batch['valid_action_stats'] += torch.tensor(valid_action, dtype=torch.int)
        state.batch['valid_search_stats'] += torch.tensor(is_search, dtype=torch.int)
        
    def add_to_generation_history(self, state, response_ids, response_str, obs_ids=None, obs_str=None):
        """Add generation round to history in the state."""
        state.meta_info['history']['ids_per_round'].append(response_ids.clone())
        state.meta_info['history']['str_per_round'].append(response_str.copy())
        if obs_str is not None and len(obs_str) > 0:
            state.meta_info['history']['str_per_round'].append(obs_str.copy())
            state.meta_info['history']['ids_per_round'].append(obs_ids.clone())

    def _generate_candidates(self, rollings, active_mask):
        """Generate candidate responses for current state.
        
        Args:
            rollings: Current rolling state
            active_mask: Active trajectories mask
            beam_size: Number of candidates to generate per trajectory
            
        Returns:
            Tuple of (responses_ids, responses_str, meta_info)
        """
        # Prepare input
        rollings.batch = self.tensor_fn.cut_to_effective_len(
            rollings.batch,
            keys=['input_ids', 'attention_mask', 'position_ids']
        )
        
        # Generate with active mask
        rollings_active = DataProto.from_dict({
            k: v[active_mask] for k, v in rollings.batch.items()
        })
        
        # Generate sequences (can be extended for beam search)
        gen_output = self._generate_with_gpu_padding(rollings_active)
        
        # Process outputs
        meta_info = gen_output.meta_info
        responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
        responses_ids, responses_str = self.tensor_fn._example_level_pad(
            responses_ids, responses_str, active_mask
        )
        
        return responses_ids, responses_str, meta_info
    
    def _execute_turn(self, responses_str, active_mask, do_search=True):
        """Execute a turn and get environment feedback.
        
        Args:
            responses_str: Generated response strings
            active_mask: Active trajectories mask  
            do_search: Whether to execute search queries
            
        Returns:
            Tuple of (next_obs, dones, valid_action, is_search, next_obs_ids)
        """
        # Execute predictions in environment
        next_obs, dones, valid_action, is_search = self.execute_predictions(
            responses_str, self.tokenizer.pad_token, active_mask, do_search=do_search
        )
        
        # Process observations to token IDs if needed
        next_obs_ids = self._process_next_obs(next_obs) if do_search else None
        
        return next_obs, dones, valid_action, is_search, next_obs_ids
    
    def _process_generation_turn(self, state: DataProto, rollings, is_final=False):
        """Process a single generation turn.
        
        Args:
            state: Current generation state
            rollings: Rolling generation batch
            is_final: Whether this is the final turn
            
        Returns:
            Updated rollings and meta_info
        """
        # Generate candidates
        responses_ids, responses_str, meta_info = self._generate_candidates(
            rollings, state.batch['active_mask']
        )
        
        # Execute turn
        next_obs, dones, valid_action, is_search, next_obs_ids = self._execute_turn(
            responses_str, state.batch['active_mask'], do_search=not is_final
        )
        
        # Update state
        curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
        self.update_generation_stats(state, curr_active_mask, valid_action, is_search)
        
        # Update history
        if is_final:
            self.add_to_generation_history(state, responses_ids, responses_str)
        else:
            self.add_to_generation_history(state, responses_ids, responses_str, next_obs_ids, next_obs)
        
        # Update generation context
        if not is_final:
            rollings = self._update_rolling_state(rollings, responses_ids, next_obs_ids)
            # Update right_side in state
            right_side = {
                'responses': state.batch['responses'],
                'responses_with_info_mask': state.batch['responses_with_info_mask']
            }
            updated_right = self._update_right_side(right_side, responses_ids, next_obs_ids)
            state.batch['responses'] = updated_right['responses']
            state.batch['responses_with_info_mask'] = updated_right['responses_with_info_mask']
        else:
            # Update right_side in state for final turn
            right_side = {
                'responses': state.batch['responses'],
                'responses_with_info_mask': state.batch['responses_with_info_mask']
            }
            updated_right = self._update_right_side(right_side, responses_ids)
            state.batch['responses'] = updated_right['responses']
            state.batch['responses_with_info_mask'] = updated_right['responses_with_info_mask']
            
        return rollings, meta_info
    
    def _run_single_generation(self, gen_batch, initial_input_ids: torch.Tensor):
        """Run single generation (refactored for beam search compatibility)."""

        # Initialize generation state
        batch_size = gen_batch.batch['input_ids'].shape[0]
        state = self.create_generation_state(batch_size, initial_input_ids, self.config.max_start_length)
        rollings = gen_batch
        meta_info = {}

        # Main generation loop for intermediate turns
        for step in range(self.config.max_turns):
            if not state.batch['active_mask'].sum():
                break
                
            # Process turn with search
            rollings, meta_info = self._process_generation_turn(
                state, rollings, is_final=False
            )

        # Final generation turn (without search)
        if state.batch['active_mask'].sum():
            rollings, meta_info = self._process_generation_turn(
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
        
        return self._compose_final_output(left_side, right_side, meta_info)

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']

        # Combine input IDs
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'],
            right_side['responses']
        ], dim=1)

        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids']),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)

        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )

        final_output = DataProto.from_dict(final_output)
        final_output.meta_info.update(meta_info)

        return final_output

    def execute_predictions(self, predictions: List[str], pad_token: str, active_mask=None, do_search=True) -> List[str]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            
        Returns:
            List of observation strings
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        next_obs, dones, valid_action, is_search = [], [], [], []

        search_queries = [content for action, content in zip(cur_actions, contents) if action == 'search']
        if do_search:
            search_results = self.batch_search(search_queries)
            assert len(search_results) == sum([1 for action in cur_actions if action == 'search'])
        else:
            search_results = [''] * sum([1 for action in cur_actions if action == 'search'])

        # tag_str = "\n\n<information></information>\n\n"
        # num_tag_str = self.tokenizer(tag_str, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[1]
        max_tokens = self.config.max_obs_length - 10

        def truncate_by_tokens(text: str) -> str:
            ids = self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,  # Prevents adding special tokens
            )["input_ids"]
            ids = ids[0, :max_tokens]
            return self.tokenizer.decode(ids, skip_special_tokens=True)

        search_results = [truncate_by_tokens(r) for r in search_results]

        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):

            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                else:
                    next_obs.append(f'\nMy previous action is invalid. \
If I want to search, I should put the query between <search> and </search>. \
If I want to give the final answer, I should put the answer between <answer> and </answer>. Let me try again.\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)

        assert len(search_results) == 0

        return next_obs, dones, valid_action, is_search

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        """
        Process (text-based) predictions from llm into actions and validity flags.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, validity flags list)
        """
        actions = []
        contents = []

        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                pattern = r'<(search|answer)>(.*?)</\1>'
                match = re.search(pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()  # Return only the content inside the tags
                    action = match.group(1)
                else:
                    content = ''
                    action = None
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")

            actions.append(action)
            contents.append(content)

        return actions, contents

    def batch_search(self, queries: List[str] = None) -> str:
        """
        Batchified search for queries.
        Args:
            queries: queries to call the search engine
        Returns:
            search results which is concatenated into a string
        """
        results = self._batch_search(queries)['result']

        return [self._passages2string(result) for result in results]

    def _batch_search(self, queries):

        payload = {
            "queries": queries,
            "topk": self.config.topk,
            "return_scores": True
        }

        return requests.post(self.config.search_url, json=payload).json()

    def _passages2string(self, retrieval_result):
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):

            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"

        return format_reference
