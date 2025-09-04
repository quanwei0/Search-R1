import asyncio
from typing import List

from vllm_serve.vllm_client import VLLMClient, JudgeEvaluator, DataProcessor
from vllm_serve.async_vllm_client import AsyncVLLMClient, run_batch


# ============================================================================
# Main Scoring Function
# ============================================================================


def compute_score_step_retrieval_format_judge(
    mid_turn_str,
    final_turn_str,
    solution_str,
    ground_truth_str,
    host: str,
    port: int,
    judge_model_name: str,
    use_async: bool = False,
):
    """Compute step retrieval format judge scores for turns.

    Args:
        mid_turn_str: Middle turn text(s) - can be single item or batch
        final_turn_str: Final turn text - can be single item or batch
        solution_str: Complete solution string - can be single item or batch
        ground_truth_str: Ground truth answer(s) - can be single item or batch
        host: VLLM server host
        port: VLLM server port
        judge_model_name: Model name for the judge
        use_async: Whether to use async batch processing (recommended for batches)

    Returns:
        List of scores for each turn (excluding final turn)
    """

    # Check if this is batch processing
    is_batch = isinstance(solution_str, list)

    if is_batch and use_async:
        # Use async batch processing for better performance
        return asyncio.run(
            _compute_async_batch_scores(
                mid_turn_str,
                final_turn_str,
                solution_str,
                ground_truth_str,
                host,
                port,
                judge_model_name,
            )
        )
    elif is_batch:
        # Use sync batch processing
        return _compute_sync_batch_scores(
            mid_turn_str,
            final_turn_str,
            solution_str,
            ground_truth_str,
            host,
            port,
            judge_model_name,
        )
    else:
        # Single item processing
        return _compute_single_scores(
            mid_turn_str,
            final_turn_str,
            solution_str,
            ground_truth_str,
            host,
            port,
            judge_model_name,
        )


def _compute_single_scores(
    mid_turn_str: List[str],
    final_turn_str: str,
    solution_str: str,
    ground_truth_str: str,
    host: str,
    port: int,
    judge_model_name: str,
) -> List[float]:
    """Compute scores for a single item using sync client."""
    num_turns_minus_1 = len(mid_turn_str)

    # Initialize client and evaluator
    client = VLLMClient(host=host, port=port, model=judge_model_name)
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
    judge_prompt = judge_evaluator.create_judge_prompt(prompt, turns, ground_truth_str)
    result = client.generate_text(judge_prompt)

    if result:
        # Extract scores for each turn
        all_scores = judge_evaluator.extract_turn_scores_from_judge_response(
            result, len(turns)
        )
        print(f"All scores extracted: {all_scores}")

        # Return only the scores for mid turns (excluding final turn)
        mid_turn_scores = (
            all_scores[:num_turns_minus_1]
            if len(all_scores) >= num_turns_minus_1
            else [0.0] * num_turns_minus_1
        )
        print(f"Mid turn scores (final output): {mid_turn_scores}")

        return mid_turn_scores
    else:
        # Return default scores if evaluation failed
        default_scores = [0.0] * num_turns_minus_1
        print(f"Evaluation failed, returning default scores: {default_scores}")
        return default_scores


async def _compute_async_batch_scores(
    mid_turn_str_list: List[List[str]],
    final_turn_str_list: List[str],
    solution_str_list: List[str],
    ground_truth_str_list: List[str],
    host: str,
    port: int,
    judge_model_name: str,
) -> List[List[float]]:
    """Compute scores for batch of items using async client for better performance."""
    print(f"Processing batch of {len(solution_str_list)} items with async client...")

    # Initialize async client and evaluator
    client = AsyncVLLMClient(host=host, port=port, model=judge_model_name)
    judge_evaluator = JudgeEvaluator()
    data_processor = DataProcessor()

    # Prepare samples for async batch processing
    samples = []
    for mid_turns, final_turn, solution, ground_truth in zip(
        mid_turn_str_list, final_turn_str_list, solution_str_list, ground_truth_str_list
    ):
        prompt = data_processor.extract_prompt_from_chat_format(solution)

        # Prepare turns list for this item
        if isinstance(mid_turns, str):
            turns = [mid_turns, final_turn]
        else:
            turns = list(mid_turns) + [final_turn]

        samples.append((prompt, turns, ground_truth))

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
    for judge_text, mid_turns in zip(judge_texts, mid_turn_str_list):
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
    mid_turn_str_list: List[List[str]],
    final_turn_str_list: List[str],
    solution_str_list: List[str],
    ground_truth_str_list: List[str],
    host: str,
    port: int,
    judge_model_name: str,
) -> List[List[float]]:
    """Compute scores for batch of items using sync client."""
    print(f"Processing batch of {len(solution_str_list)} items with sync client...")

    # Initialize client and evaluator
    client = VLLMClient(host=host, port=port, model=judge_model_name)
    judge_evaluator = JudgeEvaluator()
    data_processor = DataProcessor()

    # Extract prompts from all solution strings
    prompts = []
    turns_list = []
    ground_truths = []

    for mid_turns, final_turn, solution, ground_truth in zip(
        mid_turn_str_list, final_turn_str_list, solution_str_list, ground_truth_str_list
    ):
        prompt = data_processor.extract_prompt_from_chat_format(solution)
        prompts.append(prompt)
        ground_truths.append(ground_truth)

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
    for result, mid_turns in zip(results, mid_turn_str_list):
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
# Alternative function name for compatibility
# ============================================================================


def compute_step_retrieval_format_judge_score(
    mid_turn_str,
    final_turn_str,
    solution_str,
    ground_truth_str,
    host: str,
    port: int,
    judge_model_name: str,
    use_async: bool = False,
):
    """Alternative function name for compatibility.

    Args:
        mid_turn_str: Middle turn text(s) - can be single item or batch
        final_turn_str: Final turn text - can be single item or batch
        solution_str: Complete solution string - can be single item or batch
        ground_truth_str: Ground truth answer(s) - can be single item or batch
        host: VLLM server host
        port: VLLM server port
        judge_model_name: Model name for the judge
        use_async: Whether to use async batch processing (recommended for batches)

    Returns:
        List of scores for each turn (excluding final turn)
    """
    ground_truths = ground_truth_str["target"]
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]
    return compute_score_step_retrieval_format_judge(
        mid_turn_str,
        final_turn_str,
        solution_str,
        ground_truths,
        host,
        port,
        judge_model_name,
        use_async,
    )
