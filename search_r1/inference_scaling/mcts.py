"""
Monte Carlo Tree Search (MCTS) Implementation for Search-R1
Implements MCTS for multi-turn reasoning with search engine integration.
"""

import torch
import math
import random
import copy
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from .base_scaler import BaseInferenceGenerator
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto


@dataclass
class MCTSNode:
    """A node in the MCTS tree."""
    state: DataProto  # Current generation state
    rollings: DataProto  # Current rolling batch
    action: Optional[str] = None  # Action that led to this node
    parent: Optional['MCTSNode'] = None
    children: Dict[str, 'MCTSNode'] = field(default_factory=dict)
    
    # Statistics
    visit_count: int = 0
    value_sum: float = 0.0
    value: float = 0.0  # Average value
    prior: float = 1.0  # Prior probability (uniform by default)
    
    # Flags
    is_terminal: bool = False
    is_fully_expanded: bool = False
    
    @property
    def q_value(self) -> float:
        """Get Q-value (average value) of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def uct_value(self, c: float = 1.414, epsilon: float = 1e-8) -> float:
        """Calculate UCT value for node selection."""
        if self.visit_count == 0:
            return float('inf')
        
        exploitation = self.q_value
        exploration = c * math.sqrt(math.log(self.parent.visit_count + 1) / (self.visit_count + epsilon))
        return exploitation + exploration
    
    def add_child(self, action: str, child_state: DataProto, child_rollings: DataProto) -> 'MCTSNode':
        """Add a child node."""
        child = MCTSNode(
            state=child_state,
            rollings=child_rollings,
            action=action,
            parent=self
        )
        self.children[action] = child
        return child
    
    def select_best_child(self, c: float = 1.414) -> 'MCTSNode':
        """Select the best child using UCT."""
        return max(self.children.values(), key=lambda n: n.uct_value(c))
    
    def select_most_visited_child(self) -> 'MCTSNode':
        """Select the most visited child (for final selection)."""
        return max(self.children.values(), key=lambda n: n.visit_count)


class MCTSGenerator(BaseInferenceGenerator):
    """MCTS generator for multi-turn reasoning with search."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize MCTS generator.
        
        Args:
            config: Configuration dictionary containing:
                - num_simulations: Number of MCTS simulations to run
                - max_turns: Maximum number of reasoning turns
                - c_puct: Exploration constant for UCT
                - num_expand_actions: Number of actions to expand at each node
                - temperature: Temperature for final action selection
                - use_critic: Whether to use critic for value estimation
                - critic_worker_group: Worker group for critic scoring
                - value_discount: Discount factor for values across turns
        """
        super().__init__(config)
        
        # MCTS parameters
        self.num_simulations = config.get('num_simulations', 100)
        self.max_turns = config.get('max_turns', 10)
        self.c_puct = config.get('c_puct', 1.414)
        self.num_expand_actions = config.get('num_expand_actions', 3)
        self.temperature = config.get('temperature', 1.0)
        self.value_discount = config.get('value_discount', 0.99)
        
        # Scoring parameters
        self.use_critic = config.get('use_critic', True)
        self.critic_wg = config.get('critic_worker_group', None)
        
        # Tree reuse
        self.reuse_tree = config.get('reuse_tree', False)
        self.tree_cache = {}
        
    def generate(
        self,
        generation_manager,
        gen_batch: DataProto,
        initial_input_ids: torch.Tensor,
        reward_fn: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Generate using MCTS.
        
        Args:
            generation_manager: The LLMGenerationManager instance
            gen_batch: Initial generation batch
            initial_input_ids: Initial input token IDs
            reward_fn: Optional reward function for value estimation
            
        Returns:
            Best trajectory found by MCTS
        """
        self.generation_manager = generation_manager
        self.reward_fn = reward_fn
        
        # Initialize root node
        batch_size = gen_batch.batch['input_ids'].shape[0]
        if batch_size > 1:
            print(f"[MCTS] Warning: Batch size {batch_size} > 1. Processing sequentially.")
        
        # Process each item in batch separately (MCTS is inherently sequential)
        all_results = []
        for b_idx in range(batch_size):
            single_batch = gen_batch.select_idxs([b_idx])
            single_input_ids = initial_input_ids[b_idx:b_idx+1]
            
            result = self._run_mcts_for_single_item(single_batch, single_input_ids)
            all_results.append(result)
        
        # Combine results
        if batch_size == 1:
            return all_results[0]
        else:
            # Stack results from multiple items
            return self._combine_results(all_results)
    
    def _run_mcts_for_single_item(
        self,
        gen_batch: DataProto,
        initial_input_ids: torch.Tensor
    ) -> DataProto:
        """Run MCTS for a single item."""
        breakpoint()
        # Create root state
        root_state = self.generation_manager.create_generation_state(
            1, initial_input_ids, self.generation_manager.config.max_start_length
        )
        
        # Initialize root node
        root = MCTSNode(
            state=root_state,
            rollings=gen_batch
        )
        
        # Run MCTS simulations
        for sim in range(self.num_simulations):
            if sim % 10 == 0:
                print(f"[MCTS] Simulation {sim}/{self.num_simulations}")
            
            # 1. Selection: Select leaf node
            node = self._select(root)
            
            # 2. Expansion: Expand if not terminal
            if not node.is_terminal and not node.is_fully_expanded:
                node = self._expand(node)
            
            # 3. Simulation: Rollout from node
            value = self._simulate(node)
            
            # 4. Backpropagation: Update values
            self._backpropagate(node, value)
        
        # Select best trajectory based on visit counts
        best_path = self._extract_best_path(root)
        return self._path_to_output(best_path)
    
    def _select(self, root: MCTSNode) -> MCTSNode:
        """Select a leaf node using UCT."""
        node = root
        
        while node.children and not node.is_terminal:
            # If not fully expanded, return for expansion
            if not node.is_fully_expanded:
                return node
            
            # Select best child using UCT
            node = node.select_best_child(self.c_puct)
        
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand a node by generating new actions."""
        
        # Check if already terminal
        if not node.state.batch['active_mask'].any():
            node.is_terminal = True
            node.is_fully_expanded = True
            return node
        
        # Generate candidate actions
        actions = self._generate_actions(node)
        
        if not actions:
            node.is_fully_expanded = True
            return node
        
        # Add children for each action
        for action_data in actions:
            action_key = self._action_to_key(action_data['response_str'])
            if action_key not in node.children:
                # Create new state after taking this action
                # Deep copy the DataProto objects
                new_state = copy.deepcopy(node.state)
                new_rollings = copy.deepcopy(node.rollings)
                new_state, new_rollings = self._apply_action(
                    new_state,
                    new_rollings,
                    action_data
                )
                
                child = node.add_child(action_key, new_state, new_rollings)
                
                # Check if terminal
                if not new_state.batch['active_mask'].any() or action_data.get('is_final', False):
                    child.is_terminal = True
        
        node.is_fully_expanded = len(node.children) >= min(self.num_expand_actions, len(actions))
        
        # Return a random child for simulation
        if node.children:
            return random.choice(list(node.children.values()))
        return node
    
    def _generate_actions(self, node: MCTSNode) -> List[Dict]:
        """Generate possible actions from current state."""
        
        # Use generation manager to get candidate responses
        state = node.state
        rollings = node.rollings
        
        # Generate multiple candidates by sampling
        actions = []
        for _ in range(self.num_expand_actions):
            # Generate one response
            response_ids, response_str, meta_info = self.generation_manager._generate_candidates(
                rollings, state.batch['active_mask']
            )
            
            # Check if this is a final action (answer)
            is_final = any('</answer>' in s for s in response_str)
            
            actions.append({
                'response_ids': response_ids,
                'response_str': response_str,
                'meta_info': meta_info,
                'is_final': is_final
            })
        
        return actions
    
    def _apply_action(
        self, 
        state: DataProto, 
        rollings: DataProto,
        action_data: Dict
    ) -> Tuple[DataProto, DataProto]:
        """Apply an action to create a new state."""
        
        # Execute the action to get environment feedback
        next_obs, dones, valid_action, is_search, next_obs_ids = self.generation_manager._execute_turn(
            action_data['response_str'], 
            state.batch['active_mask'],
            do_search=not action_data.get('is_final', False)
        )
        
        # Update state
        curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
        self.generation_manager.update_generation_stats(
            state, curr_active_mask, valid_action, is_search
        )
        
        # Update history
        if action_data.get('is_final', False):
            self.generation_manager.add_to_generation_history(
                state, action_data['response_ids'], action_data['response_str']
            )
        else:
            self.generation_manager.add_to_generation_history(
                state, action_data['response_ids'], action_data['response_str'], 
                next_obs_ids, next_obs
            )
        
        # Update generation context
        if not action_data.get('is_final', False) and next_obs_ids is not None:
            rollings = self.generation_manager._update_rolling_state(
                rollings, action_data['response_ids'], next_obs_ids
            )
            
            # Update right_side in state
            right_side = {
                'responses': state.batch['responses'],
                'responses_with_info_mask': state.batch['responses_with_info_mask']
            }
            updated_right = self.generation_manager._update_right_side(
                right_side, action_data['response_ids'], next_obs_ids
            )
            state.batch['responses'] = updated_right['responses']
            state.batch['responses_with_info_mask'] = updated_right['responses_with_info_mask']
        
        return state, rollings
    
    def _simulate(self, node: MCTSNode) -> float:
        """Simulate from a node to estimate its value."""
        
        # If terminal, evaluate directly
        if node.is_terminal:
            return self._evaluate_terminal(node)
        
        # Otherwise, do a rollout
        sim_state = copy.deepcopy(node.state)
        sim_rollings = copy.deepcopy(node.rollings)
        
        # Simple rollout: take random actions until terminal
        total_value = 0.0
        discount = 1.0
        
        for turn in range(self.max_turns):
            if not sim_state.batch['active_mask'].any():
                break
            
            # Generate one action
            actions = self._generate_actions(MCTSNode(sim_state, sim_rollings))
            if not actions:
                break
            
            action = random.choice(actions)
            sim_state, sim_rollings = self._apply_action(sim_state, sim_rollings, action)
            
            # Add discounted value
            if self.use_critic and self.reward_fn:
                value = self._evaluate_state(sim_state, sim_rollings)
                total_value += discount * value
                discount *= self.value_discount
            
            if action.get('is_final', False):
                break
        
        # Final evaluation
        if self.use_critic and self.reward_fn:
            final_value = self._evaluate_terminal_state(sim_state, sim_rollings)
            total_value += discount * final_value
        
        return total_value
    
    def _evaluate_state(self, state: DataProto, rollings: DataProto) -> float:
        """Evaluate a non-terminal state using critic."""
        if not self.reward_fn:
            return 0.0
        
        # Compose batch for scoring
        left_side = {'input_ids': state.batch['left_input_ids']}
        right_side = {
            'responses': state.batch['responses'],
            'responses_with_info_mask': state.batch['responses_with_info_mask']
        }
        scoring_batch = self.generation_manager._compose_final_output(left_side, right_side, {})
        
        # Score with critic
        output = self._batch_score(scoring_batch)
        if output is None:
            return 0.0
        
        values = output.batch.get('values')
        if values is None:
            return 0.0
        
        # Get last non-zero value
        non_zero_indices = (values[0] != 0).nonzero(as_tuple=True)[0]
        if len(non_zero_indices) > 0:
            return values[0, non_zero_indices[-1]].item()
        
        return 0.0
    
    def _evaluate_terminal_state(self, state: DataProto, rollings: DataProto) -> float:
        """Evaluate a terminal state."""
        return self._evaluate_state(state, rollings)
    
    def _evaluate_terminal(self, node: MCTSNode) -> float:
        """Evaluate a terminal node."""
        return self._evaluate_terminal_state(node.state, node.rollings)
    
    def _backpropagate(self, node: MCTSNode, value: float):
        """Backpropagate value up the tree."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            current.value = current.value_sum / current.visit_count
            # Apply discount for parent
            value *= self.value_discount
            current = current.parent
    
    def _extract_best_path(self, root: MCTSNode) -> List[MCTSNode]:
        """Extract the best path from root to leaf."""
        path = [root]
        node = root
        
        while node.children:
            # Select most visited child
            node = node.select_most_visited_child()
            path.append(node)
            
            if node.is_terminal:
                break
        
        return path
    
    def _path_to_output(self, path: List[MCTSNode]) -> DataProto:
        """Convert a path to final output format."""
        if not path:
            return None
        
        # Get the final node
        final_node = path[-1]
        state = final_node.state
        
        # Compile metadata
        meta_info = {
            'turns_stats': state.batch['turns_stats'].tolist(),
            'active_mask': state.batch['active_mask'].tolist(),
            'valid_action_stats': state.batch['valid_action_stats'].tolist(),
            'valid_search_stats': state.batch['valid_search_stats'].tolist(),
            'generation_history': state.meta_info.get('history', {}),
            'mcts_stats': {
                'num_simulations': self.num_simulations,
                'tree_depth': len(path) - 1,
                'final_value': final_node.value,
                'visit_count': final_node.visit_count
            }
        }
        
        # Compose final output
        left_side = {'input_ids': state.batch['left_input_ids']}
        right_side = {
            'responses': state.batch['responses'],
            'responses_with_info_mask': state.batch['responses_with_info_mask']
        }
        
        return self.generation_manager._compose_final_output(left_side, right_side, meta_info)
    
    def _batch_score(self, batch: DataProto) -> Optional[DataProto]:
        """Score batch with critic value function."""
        if not self.reward_fn:
            return None
        
        try:
            # Pad for multi-GPU if needed
            batch_padded, pad_size = pad_dataproto_to_divisor(batch, self.reward_fn.world_size)
            output = self.reward_fn.compute_values(batch_padded)
            
            # Unpad if needed
            if pad_size:
                output = unpad_dataproto(output, pad_size)
            
            return output
        except Exception as e:
            print(f"[MCTS] Warning: Failed to score with critic: {e}")
            return None
    
    def _action_to_key(self, response_str: List[str]) -> str:
        """Convert action to a hashable key."""
        # Use first 100 chars of response as key
        if response_str and response_str[0]:
            return response_str[0][:100]
        return "empty"
    
    def _combine_results(self, results: List[DataProto]) -> DataProto:
        """Combine results from multiple items."""
        # Stack all results
        combined_batch = {}
        
        # Get keys from first result
        first_result = results[0]
        for key in first_result.batch.keys():
            tensors = [r.batch[key] for r in results]
            combined_batch[key] = torch.cat(tensors, dim=0)
        
        combined = DataProto.from_dict(combined_batch)
        
        # Combine metadata
        combined.meta_info = {
            'mcts_stats': [r.meta_info.get('mcts_stats', {}) for r in results],
            'generation_history': [r.meta_info.get('generation_history', {}) for r in results]
        }
        
        return combined