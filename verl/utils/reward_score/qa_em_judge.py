from typing import List

from vllm_serve.vllm_client import VLLMClient, JudgeEvaluator, DataProcessor


# ============================================================================
# Main Scoring Function
# ============================================================================

def compute_score_step_retrieval_format_judge(mid_turn_str: List[str], final_turn_str: str, solution_str: str) -> List[float]:
    """Compute step retrieval format judge scores for turns.
    
    Args:
        mid_turn_str: Middle turn text(s)
        final_turn_str: Final turn text
        solution_str: Complete solution string
        
    Returns:
        List of scores for each turn (excluding final turn)
    """
    
    num_turns_minus_1 = len(mid_turn_str)
    
    # Initialize client and evaluator
    client = VLLMClient()
    judge_evaluator = JudgeEvaluator()
    data_processor = DataProcessor()
    
    # Extract prompt from solution string
    prompt = data_processor.extract_prompt_from_chat_format(solution_str)
    
    # Prepare turns list
    if isinstance(mid_turn_str, str):
        turns = [mid_turn_str, final_turn_str]
    else:
        turns = list(mid_turn_str) + [final_turn_str]
    
    # Generate judge prompt and get evaluation
    judge_prompt = judge_evaluator.create_judge_prompt(prompt, turns=turns)
    result = client.generate_text(judge_prompt)
    
    if result:
        # Extract scores for each turn
        all_scores = judge_evaluator.extract_turn_scores_from_judge_response(result, len(turns))
        print(f"All scores extracted: {all_scores}")
        
        # Return only the scores for mid turns (excluding final turn)
        mid_turn_scores = all_scores[:num_turns_minus_1] if len(all_scores) >= num_turns_minus_1 else [0.0] * num_turns_minus_1
        print(f"Mid turn scores (final output): {mid_turn_scores}")
        
        return mid_turn_scores
    else:
        # Return default scores if evaluation failed
        default_scores = [0.0] * num_turns_minus_1
        print(f"Evaluation failed, returning default scores: {default_scores}")
        return default_scores


# ============================================================================
# Alternative function name for compatibility
# ============================================================================

def compute_step_retrieval_format_judge_score(mid_turn_str: List[str], final_turn_str: str, solution_str: str) -> List[float]:
    """Alternative function name for compatibility."""
    return compute_score_step_retrieval_format_judge(mid_turn_str, final_turn_str, solution_str)