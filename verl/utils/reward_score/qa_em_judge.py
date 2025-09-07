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
) -> List[List[float]]:
    """Compute scores for batch of items using async client for better performance."""
    print(f"Processing batch of {len(batch_solutions)} items with async client...")

    # Initialize async client and evaluator
    client = AsyncVLLMClient(host=host, port=port, model=judge_model_name)
    judge_evaluator = JudgeEvaluator()
    data_processor = DataProcessor()

    # Prepare samples for async batch processing
    samples = []
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
            
        samples.append((prompt, turns, ground_truth_str))

    # Use async batch processing
    judge_texts = await run_batch(
        samples,
        client=client,
        concurrency=min(64, len(samples)),  # Limit concurrency
        max_tokens=2048,
        max_retries=2,
    )

    # Extract scores for each item in the batch
    batch_mid_scores = []
    for judge_text, mid_turns in zip(judge_texts, batch_mid_turns):
        num_turns = (
            len(mid_turns) + 1 if isinstance(mid_turns, list) else 2
        )  # +1 for final turn

        if judge_text:
            all_scores = judge_evaluator.extract_turn_scores_from_judge_response(
                judge_text, num_turns
            )
            num_turns_minus_1 = len(mid_turns) if isinstance(mid_turns, list) else 1
            mid_scores = (
                all_scores[:num_turns_minus_1]
                if len(all_scores) >= num_turns_minus_1
                else [0.0] * num_turns_minus_1
            )
        else:
            num_turns_minus_1 = len(mid_turns) if isinstance(mid_turns, list) else 1
            mid_scores = [0.0] * num_turns_minus_1

        batch_mid_scores.append(mid_scores)

    print(f"Async batch processing completed. Results: {len(batch_mid_scores)} items")
    return batch_mid_scores


def _compute_sync_batch_scores(
    batch_mid_turns: List[List[str]],
    batch_final_turns: List[str],
    batch_solutions: List[str],
    batch_ground_truths: List[Dict[str, Union[str, List[str], Any]]],
    host: str,
    port: int,
    judge_model_name: str,
) -> List[List[float]]:
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

    # Process each item individually since batch methods don't exist
    results = []
    for prompt, turns, ground_truth in zip(prompts, turns_list, ground_truths):
        judge_prompt = judge_evaluator.create_judge_prompt(prompt, turns, ground_truth)
        result = client.generate_text(judge_prompt)
        results.append(result)

    # Extract scores for each item in the batch
    batch_mid_scores = []
    for result, mid_turns in zip(results, batch_mid_turns):
        num_turns = len(mid_turns) + 1 if isinstance(mid_turns, list) else 2

        if result:
            all_scores = judge_evaluator.extract_turn_scores_from_judge_response(
                result, num_turns
            )
            num_turns_minus_1 = len(mid_turns) if isinstance(mid_turns, list) else 1
            mid_scores = (
                all_scores[:num_turns_minus_1]
                if len(all_scores) >= num_turns_minus_1
                else [0.0] * num_turns_minus_1
            )
        else:
            num_turns_minus_1 = len(mid_turns) if isinstance(mid_turns, list) else 1
            mid_scores = [0.0] * num_turns_minus_1

        batch_mid_scores.append(mid_scores)

    print(f"Sync batch processing completed. Results: {len(batch_mid_scores)} items")
    return batch_mid_scores


# ============================================================================
# Main Scoring Function  
# ============================================================================


def compute_score_step_retrieval_format_judge(
    batch_mid_turns: List[List[str]],
    batch_final_turns: List[str],
    batch_solutions: List[str],
    batch_ground_truths: List[Dict[str, Union[str, List[str], Any]]],
    host: str,
    port: int,
    judge_model_name: str,
    use_async: bool = False,
) -> List[List[float]]:
    """Compute step retrieval format judge scores for turns.

    Args:
        batch_mid_turns: List of middle turn texts for each item
        batch_final_turns: List of final turn texts
        batch_solutions: List of complete solution strings
        batch_ground_truths: List of ground truth dicts with 'target' key
        host: VLLM server host
        port: VLLM server port
        judge_model_name: Model name for the judge
        use_async: Whether to use async batch processing (recommended for batches)

    Returns:
        List of lists of scores for each item's turns (excluding final turn)
    """

    # Always batch processing - choose sync or async
    if use_async:
        # Use async batch processing for better performance
        return asyncio.run(
            _compute_async_batch_scores(
                batch_mid_turns,
                batch_final_turns,
                batch_solutions,
                batch_ground_truths,
                host,
                port,
                judge_model_name,
            )
        )
    else:
        # Use sync batch processing
        return _compute_sync_batch_scores(
            batch_mid_turns,
            batch_final_turns,
            batch_solutions,
            batch_ground_truths,
            host,
            port,
            judge_model_name,
        )
