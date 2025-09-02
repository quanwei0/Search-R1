#!/usr/bin/env python3
"""
Async VLLM Client for Search-Enabled Reasoning

An asynchronous client for batch processing with VLLM servers to perform 
search-enabled reasoning and answer evaluation with scoring capabilities.
"""

import argparse
import asyncio
import logging
import random
from typing import List, Tuple, Optional

from openai import AsyncOpenAI

# Import shared components from vllm_client.py
from vllm_serve.vllm_client import JudgeEvaluator, DataProcessor, DEFAULT_DATA_PATH


# ============================================================================
# Async VLLM Client Class
# ============================================================================

class AsyncVLLMClient:
    def __init__(self, host="0.0.0.0", port=8002, model="openai/gpt-oss-20b"):
        self.base_url = f"http://{host}:{port}/v1"
        self.model = model
        self.client = AsyncOpenAI(api_key="EMPTY", base_url=self.base_url)
        self.log = logging.getLogger(self.__class__.__name__)

    async def generate_text(self, prompt: str, max_tokens: int = 2048) -> Optional[str]:
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            self.log.error(f"Chat error: {e}")
            return None


# ============================================================================
# Async Batch Processing Functions  
# ============================================================================

async def run_batch(
    samples: List[Tuple[str, List[str]]],
    client: AsyncVLLMClient,
    concurrency: int = 16,
    max_tokens: int = 2048,
    max_retries: int = 2,
):
    sem = asyncio.Semaphore(concurrency)

    async def one_job(idx: int, sample):
        prompt, turns = sample
        judge_prompt = JudgeEvaluator.create_judge_prompt(prompt, turns)
        for attempt in range(max_retries + 1):
            try:
                async with sem:
                    text = await client.generate_text(judge_prompt, max_tokens=max_tokens)
                return idx, text
            except Exception:
                if attempt >= max_retries:
                    return idx, None
                await asyncio.sleep(0.5 * (1 + random.random()) * (attempt + 1))

    tasks = [asyncio.create_task(one_job(i, s)) for i, s in enumerate(samples)]
    results = [await t for t in asyncio.as_completed(tasks)]
    results.sort(key=lambda x: x[0])
    return [r[1] for r in results]


# ============================================================================
# Command Line Interface
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Async Batch evaluation with VLLM")
    parser.add_argument("--host", type=str, default="0.0.0.0", 
                       help="VLLM server host")
    parser.add_argument("--port", type=int, default=8002,
                       help="VLLM server port") 
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b",
                       help="Model name")
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH,
                       help="Path to data file")
    parser.add_argument("--num_samples", type=int, default=50,
                       help="Number of samples to process")
    parser.add_argument("--concurrency", type=int, default=16,
                       help="Number of concurrent requests")
    parser.add_argument("--max_tokens", type=int, default=2048,
                       help="Maximum tokens per generation")
    parser.add_argument("--max_retries", type=int, default=2,
                       help="Maximum number of retries")
    return parser.parse_args()


async def amain(args):
    """Main async function to run batch evaluation."""
    logging.basicConfig(level=logging.INFO)
    
    # Load sample data
    samples = DataProcessor.get_sample_data(json_file=args.data_path, num_samples=args.num_samples)
    print(f"Loaded {len(samples)} samples from {args.data_path}")
    
    # Initialize async client
    client = AsyncVLLMClient(host=args.host, port=args.port, model=args.model)
    print(f"Initialized client for {args.host}:{args.port} with model {args.model}")

    # Run batch processing
    judge_texts = await run_batch(
        samples,
        client=client,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
    )

    # Process results
    print(f"\nProcessing {len(judge_texts)} results...")
    for i, ((_prompt, turns), judge_text) in enumerate(zip(samples, judge_texts), 1):
        scores = JudgeEvaluator.extract_turn_scores_from_judge_response(judge_text or "", len(turns))
        print(f"\nSAMPLE {i}:")
        print("-" * 80)
        print(f"Turns: {len(turns)}")
        print(f"Scores: {scores}")


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point."""
    args = parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
