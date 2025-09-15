import asyncio
from typing import List, Union, Dict, Any

from vllm_serve.vllm_client import VLLMClient, JudgeEvaluator, DataProcessor
from vllm_serve.async_vllm_client import AsyncVLLMClient, run_batch


# ============================================================================
# Helper Functions
# ============================================================================

async def _compute_async_batch_scores(
    batch_mid_turns: List[List[str]],
    batch_final_turns: List[str],
    batch_solutions: List[str],
    batch_ground_truths: List[Dict[str, Union[str, List[str], Any]]],
    host: str,
    port: int,
    judge_model_name: str,
    judge_mode: str = 'outcome',
) -> Union[List[List[float]], List[float]]:
    """Compute scores for batch of items using async client for better performance."""
    print(f"Processing batch of {len(batch_solutions)} items with async client...")

    # Initialize async client and evaluator
    client = AsyncVLLMClient(host=host, port=port, model=judge_model_name)
    judge_evaluator = JudgeEvaluator()
    data_processor = DataProcessor()

    try:
        # Prepare samples for async batch processing
        batch_samples = []
        batch_num_turns = []
        for mid_turns, final_turn, solution, ground_truth in zip(
            batch_mid_turns, batch_final_turns, batch_solutions, batch_ground_truths
        ):
            prompt = data_processor.extract_prompt_from_chat_format(solution)

            # Prepare turns list for this item
            if isinstance(mid_turns, str):
                turns = [mid_turns, final_turn]
            else:
                turns = list(mid_turns) + [final_turn]

            # Extract ground truth target and convert to list then string
            ground_truths_list = list(ground_truth['target'])
            ground_truth_str = ", ".join(str(item) for item in ground_truths_list) if len(ground_truths_list) > 0 else ""
                
            batch_samples.append((prompt, turns, ground_truth_str))
            batch_num_turns.append(len(turns))  # Store the actual number of turns

        # Use async batch processing
        batch_judge_texts = await run_batch(
            batch_samples,
            client=client,
            concurrency=min(64, len(batch_samples)),  # Limit concurrency
            max_tokens=2048,
            max_retries=2,
            judge_mode=judge_mode,
        )

        # Extract scores for each item in the batch
        batch_scores = []
        for i, (judge_text, num_turns) in enumerate(zip(batch_judge_texts, batch_num_turns)):
            if i < 2:
                prompt, turns, ground_truth_str = batch_samples[i]
                print(f"Sample {i+1}:")
                print(f"Prompt: {prompt}")
                print(f"Response: {turns}")
                print(f"Ground Truth: {ground_truth_str}")
                print(f"Judge text:\n{judge_text}")
                print("-" * 50)
            if judge_mode == 'outcome':
                # For outcome mode, return single score per item
                if judge_text:
                    score = judge_evaluator.extract_outcome_score_from_judge_response(judge_text)
                    batch_scores.append(score)
                else:
                    batch_scores.append(0.0)
            else:
                # For turn mode, return scores for all turns
                if judge_text:
                    turn_scores = judge_evaluator.extract_turn_scores_from_judge_response(
                        judge_text, num_turns
                    )
                else:
                    turn_scores = [0.0] * num_turns

                batch_scores.append(turn_scores)

        print(f"Async batch processing completed. Results: {len(batch_scores)} items")
        return batch_scores
        
    finally:
        # Essential cleanup to prevent connection buildup
        if hasattr(client, 'client'):
            try:
                await client.client.aclose()
            except:
                pass


def _compute_sync_batch_scores(
    batch_mid_turns: List[List[str]],
    batch_final_turns: List[str],
    batch_solutions: List[str],
    batch_ground_truths: List[Dict[str, Union[str, List[str], Any]]],
    host: str,
    port: int,
    judge_model_name: str,
    judge_mode: str = 'outcome',
) -> Union[List[List[float]], List[float]]:
    """Compute scores for batch of items using sync client."""
    print(f"Processing batch of {len(batch_solutions)} items with sync client...")

    # Initialize client and evaluator
    client = VLLMClient(host=host, port=port, model=judge_model_name)
    judge_evaluator = JudgeEvaluator()
    data_processor = DataProcessor()

    # Extract prompts from all solution strings
    prompts = []
    turns_list = []
    ground_truths = []
    batch_num_turns = []

    for mid_turns, final_turn, solution, ground_truth in zip(
        batch_mid_turns, batch_final_turns, batch_solutions, batch_ground_truths
    ):
        prompt = data_processor.extract_prompt_from_chat_format(solution)
        prompts.append(prompt)
        
        # Extract ground truth target and convert to list then string
        ground_truths_list = list(ground_truth['target'])
        ground_truth_str = ", ".join(str(item) for item in ground_truths_list) if len(ground_truths_list) > 0 else ""
        ground_truths.append(ground_truth_str)

        # Prepare turns list for this item
        if isinstance(mid_turns, str):
            turns = [mid_turns, final_turn]
        else:
            turns = list(mid_turns) + [final_turn]

        turns_list.append(turns)
        batch_num_turns.append(len(turns))  # Store the actual number of turns

    # Process each item individually since batch methods don't exist
    results = []
    for prompt, turns, ground_truth in zip(prompts, turns_list, ground_truths):
        if judge_mode == 'outcome':
            judge_prompt = judge_evaluator.create_outcome_judge_prompt(prompt, turns, ground_truth)
        else:  # turn_level or step_retrieval_format
            judge_prompt = judge_evaluator.create_turn_judge_prompt(prompt, turns, ground_truth)
        result = client.generate_text(judge_prompt)
        results.append(result)

    # Extract scores for each item in the batch
    batch_scores = []
    for result, num_turns in zip(results, batch_num_turns):
        if judge_mode == 'outcome':
            # For outcome mode, return single score per item
            if result:
                score = judge_evaluator.extract_outcome_score_from_judge_response(result)
                batch_scores.append(score)
            else:
                batch_scores.append(0.0)
        else:
            # For turn mode, return scores for all turns
            if result:
                turn_scores = judge_evaluator.extract_turn_scores_from_judge_response(
                    result, num_turns
                )
            else:
                turn_scores = [0.0] * num_turns

            batch_scores.append(turn_scores)

    print(f"Sync batch processing completed. Results: {len(batch_scores)} items")
    return batch_scores


# ============================================================================
# Main Scoring Function  
# ============================================================================

def compute_score_judge(
    batch_mid_turns: List[List[str]],
    batch_final_turns: List[str],
    batch_solutions: List[str],
    batch_ground_truths: List[Dict[str, Union[str, List[str], Any]]],
    host: str,
    port: int,
    judge_model_name: str,
    judge_mode: str = 'outcome',
    use_async: bool = False,
) -> Union[List[float], List[List[float]]]:
    """Compute judge scores for a batch of solutions.
    
    Args:
        judge_mode: 'outcome' for single scores, 'turn' for turn-level scores
    """
    args = (batch_mid_turns, batch_final_turns, batch_solutions, batch_ground_truths, host, port, judge_model_name)
    
    if use_async:
        return asyncio.run(_compute_async_batch_scores(*args, judge_mode=judge_mode))
    else:
        return _compute_sync_batch_scores(*args, judge_mode=judge_mode)


# Backward compatibility aliases
def compute_score_judge_outcome(
    batch_mid_turns: List[List[str]],
    batch_final_turns: List[str],
    batch_solutions: List[str],
    batch_ground_truths: List[Dict[str, Union[str, List[str], Any]]],
    host: str,
    port: int,
    judge_model_name: str,
    use_async: bool = False,
) -> List[float]:
    """Compute judge outcome scores for a batch of solutions."""
    return compute_score_judge(
        batch_mid_turns, batch_final_turns, batch_solutions, batch_ground_truths,
        host, port, judge_model_name, judge_mode='outcome', use_async=use_async
    )


def compute_score_judge_turn(
    batch_mid_turns: List[List[str]],
    batch_final_turns: List[str],
    batch_solutions: List[str],
    batch_ground_truths: List[Dict[str, Union[str, List[str], Any]]],
    host: str,
    port: int,
    judge_model_name: str,
    use_async: bool = False,
) -> List[List[float]]:
    """Compute judge turn-level scores for a batch of multi-turn solutions."""
    return compute_score_judge(
        batch_mid_turns, batch_final_turns, batch_solutions, batch_ground_truths,
        host, port, judge_model_name, judge_mode='turn', use_async=use_async
    )