#!/usr/bin/env python3
"""
VLLM Client for Search-Enabled Reasoning

A client for interacting with VLLM servers to perform search-enabled reasoning
and answer evaluation with scoring capabilities.
"""

import argparse
import json
import logging
import re
from typing import List, Tuple, Optional

from openai import OpenAI


DEFAULT_DATA_PATH = "./outputs/log_val_traj/val-search-r1-ppo-qwen2.5-7b-em-gae_20250814_043740/trajectories_val_batch_0.json"


# ============================================================================
# VLLM Client Class
# ============================================================================

class VLLMClient:
    """Client for interacting with VLLM servers."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8002, model: str = "openai/gpt-oss-20b"):
        """Initialize VLLM client.
        
        Args:
            host: Host address for VLLM server
            port: Port number for VLLM server
            model: Model name to use
        """
        self.base_url = f"http://{host}:{port}/v1"
        self.model = model
        self.client = OpenAI(api_key="EMPTY", base_url=self.base_url)
        self.logger = logging.getLogger(__name__)
    
    def generate_text(self, prompt: str, max_tokens: int = 2048) -> Optional[str]:
        """Generate text using the VLLM server.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response or None if error occurred
        """
        try:
            self.logger.info(f"Connecting to VLLM server at: {self.base_url}")
            
            chat_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            
            return chat_response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error generating text: {e}")
            return None


# ============================================================================
# Judge and Evaluation Classes
# ============================================================================

class JudgeEvaluator:
    """Handles judge prompt creation and response evaluation."""
    
    @staticmethod
    def create_turn_judge_prompt(prompt: str, turns: List[str], ground_truth: str) -> str:
        """Create evaluation prompt for prompt-response assessment.
        
        Args:
            prompt: Original prompt
            turns: Pre-divided turn texts
            ground_truth: Ground truth answer for comparison
            
        Returns:
            Formatted judge prompt
        """
        
        prompt_text = f"PROMPT:\n{prompt}\n"
        turns_text = ""
        for i, turn in enumerate(turns, 1):
            turns_text += f"TURN {i}:\n{turn}\n"
        
        ground_truth_text = f"GROUND TRUTH:\n{ground_truth}\n"
        
        judge_prompt = f"""
You are an expert evaluator for multi-turn search-augmented reasoning systems. Given a user prompt, ground truth answer, and multi-turn generated response, evaluate each turn's effectiveness and compliance.

## EVALUATION TASK

Assess each turn's format compliance, content quality, and contribution toward the ground truth answer.

## SCORING CRITERIA

### FINAL TURN (Last Turn) - Score Range: [-1.0 to 1.0]

**Format Compliance:**
• Required: `<think>...</think><answer>...</answer>` (tags only, once each, in order)
• Answer in `<answer>` tag must not exceed 5 tokens

**Answer Correctness:**
• Correct and complete answer in `<answer>` tag that matches the ground truth

**Scoring Rules:**
• If format is incorrect: Final Turn Score = -1.0
• If format is correct, answer is incorrect: Final Turn Score = 0.2
• If format is correct, answer is correct: Final Turn Score = 1.0

### INTERMEDIATE TURNS - Score Range: [-1.0 to 1.0]

**Format Compliance:**
• Required: `<think>...</think><search>...</search><information>...</information>` (tags only, once each, in order)
• Correct format: +0.1
• Incorrect format: -0.2

**Information Quality:**
• Relevant information in `<information>` tag that helps toward the ground truth answer (e.g., ground truth exists in the retrieved result within `<information>` tag): +0.3
• Irrelevant or unhelpful information in `<information>` tag: +0.0

**Search Efficiency Penalty:**
• Number of searches = Total count of `<search>` tags across all turns from Turn 1 up to and including the current turn
• Search penalty = Number of searches × (-0.1)
• Encourages finding answers with fewer searches

**Intermediate Turn Score = Format Compliance + Information Quality + Search Penalty**

## OUTPUT FORMAT

Provide your evaluation using ONLY these XML tags:

<reasoning>
Systematically evaluate each turn: check format compliance, assess content quality, calculate scores with clear explanations
</reasoning>

<score>
Turn1: X.X
Turn2: X.X
Turn3: X.X
...
</score>

⚠️ REQUIREMENTS:
• Must provide exactly {len(turns)} scores (one per turn)
• Use decimal format (e.g., 0.5, -0.3, 1.0)
• Use only the specified XML tags, no additional text

## EVALUATION DATA

{prompt_text}
{turns_text}
{ground_truth_text}
**TURNS TO EVALUATE: {len(turns)}**

## Your Evaluation
"""

        return judge_prompt

    @staticmethod
    def extract_turn_scores_from_judge_response(judge_response: str, num_turns: int) -> List[float]:
        """Extract individual turn scores from judge response.
        
        Args:
            judge_response: The judge's evaluation response
            num_turns: Expected number of turns
            
        Returns:
            List of scores for each turn
        """
        scores = []
        try:
            # Extract score section from the result
            score_pattern = r'<score>(.*?)</score>'
            score_match = re.search(score_pattern, judge_response, re.DOTALL)
            
            if score_match:
                score_text = score_match.group(1).strip()
                # Parse individual turn scores
                turn_pattern = r'Turn(\d+):\s*([-+]?\d*\.?\d+)'
                turn_matches = re.findall(turn_pattern, score_text)
                
                for _, score_str in turn_matches:
                    score = float(score_str)
                    scores.append(score)
            else:
                print(f"Warning: No score section found in judge response, defaulting to zero scores")
                scores = [0.0] * num_turns
        
        except Exception as e:
            print(f"Error parsing scores: {e}, defaulting to zero scores")
            scores = [0.0] * num_turns
        
        # Ensure we have the correct number of scores for number of turns
        if len(scores) != num_turns:
            print(f"\nWarning: Expected {num_turns} scores, got {len(scores)}. Scores: {scores}. Padding.")
            print(f"Judge response: {judge_response}")
            if len(scores) < num_turns:
                print(f"Padding with zeros.")              
                scores.extend([0.0] * (num_turns - len(scores)))
            else:
                print(f"Truncating extra scores.")
                scores = scores[:num_turns]
        
        return scores

    @staticmethod
    def create_outcome_judge_prompt(prompt: str, turns: List[str], ground_truth: str) -> str:
        """Create evaluation prompt for prompt-response assessment.
        
        Args:
            prompt: Original prompt
            turns: Pre-divided turn texts
            ground_truth: Ground truth answer for comparison
            
        Returns:
            Formatted judge prompt
        """
        
        prompt_text = f"PROMPT:\n{prompt}\n"
        turns_text = ""
        for i, turn in enumerate(turns, 1):
            turns_text += f"TURN {i}:\n{turn}\n"
        
        ground_truth_text = f"GROUND TRUTH:\n{ground_truth}\n"
        
        judge_prompt = f"""
You are an expert evaluator for multi-turn search-augmented reasoning systems. Given a user prompt, ground truth answer, and multi-turn generated response, determine whether the final answer matches the ground truth.

## EVALUATION TASK

Evaluate whether the multi-turn response provides a correct final answer that matches the ground truth.

## SCORING CRITERIA

**Score 1.0 (Correct):**
• The answer within `<answer></answer>` tags matches the ground truth

**Score 0.0 (Incorrect):**
• No `<answer></answer>` tags found, OR
• The answer within `<answer></answer>` tags does not match the ground truth, OR
• The answer in `<answer>` tag exceeds 5 tokens

## OUTPUT FORMAT

Provide your evaluation using this format:

<reasoning>
[Your step-by-step reasoning about whether the answer matches the ground truth]
</reasoning>

<score>
1.0 or 0.0
</score>

⚠️ REQUIREMENTS:
• First provide reasoning, then the score
• Score must be exactly 1.0 or 0.0

## EVALUATION DATA

{prompt_text}
{turns_text}
{ground_truth_text}

## Your Evaluation
"""
        return judge_prompt

    @staticmethod
    def extract_outcome_score_from_judge_response(judge_response: str) -> float:
        """Extract outcome score from judge response.
        
        Args:
            judge_response: The judge's evaluation response
            
        Returns:
            Single outcome score as float
        """
        try:
            # Extract score from XML tag
            score_pattern = r'<score>(.*?)</score>'
            match = re.search(score_pattern, judge_response, re.DOTALL)
            
            if match:
                score_text = match.group(1).strip()
                score = float(score_text)
                # Clamp score to valid range
                return max(0.0, min(1.0, score))
            else:
                print(f"Warning: Could not find score tag in judge response")
                return 0.0
                
        except (ValueError, AttributeError) as e:
            print(f"Warning: Could not parse outcome score: {e}")
            return 0.0


# ============================================================================
# Data Processing Classes
# ============================================================================

class DataProcessor:
    """Handles data loading and processing operations."""
    
    @staticmethod
    def extract_prompt_from_chat_format(text: str) -> str:
        """Extract the user prompt from chat format.
        
        Args:
            text: Chat format text containing <|im_start|>user and <|im_end|> tags
            
        Returns:
            Extracted prompt text
        """
        # Find content between <|im_start|>user and <|im_end|>
        pattern = r'<\|im_start\|>user\s*\n(.*?)\n<\|im_end\|>'
        match = re.search(pattern, text, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        return text  # Return original text if no pattern found


    @staticmethod
    def get_sample_data(num_samples: int = 5, json_file: str = DEFAULT_DATA_PATH) -> List[Tuple[str, List[str], str]]:
        """Get multiple sample prompts and turn texts from JSON file.
        
        Args:
            num_samples: Number of samples to retrieve
            json_file: Path to JSON file containing samples
            
        Returns:
            List of (prompt, turn_texts, ground_truth) tuples
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        # Handle -1 as "all samples"
        sample_count = len(data) if num_samples == -1 else min(num_samples, len(data))
        for i in range(sample_count):
            sample = data[i]
            raw_prompt = sample["prompt"]
            
            # Extract the actual user prompt from chat format
            prompt = DataProcessor.extract_prompt_from_chat_format(raw_prompt)
            turn_texts = sample["turn_texts"]
            ground_truth_list = sample["ground_truth"]
            ground_truth = ", ".join(ground_truth_list) if ground_truth_list else None
            samples.append((prompt, turn_texts, ground_truth))
        
        return samples


# ============================================================================
# Main Function
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VLLM Client for Search-Enabled Reasoning")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b")
    parser.add_argument("--judge-mode", type=str, default="outcome", 
                        choices=["turn", "outcome"],
                        help="Judge evaluation mode: turn (evaluate each turn), outcome (evaluate final result)")
    return parser.parse_args()


def main():
    """Main function to run the client evaluation."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    

    # Initialize components with parsed arguments
    samples = DataProcessor.get_sample_data()
    client = VLLMClient(host=args.host, port=args.port, model=args.model)
    judge_evaluator = JudgeEvaluator()
    
    # Evaluate samples
    for i, (sample_prompt, sample_turns, ground_truth) in enumerate(samples):
        print(f"\nSAMPLE {i+1}:")
        print("-" * 100)
        
        if args.judge_mode == "outcome":
            judge_prompt = judge_evaluator.create_outcome_judge_prompt(sample_prompt, sample_turns, ground_truth)
        elif args.judge_mode == "turn":
            judge_prompt = judge_evaluator.create_turn_judge_prompt(sample_prompt, sample_turns, ground_truth)
        
        print(f"Evaluating sample {i+1}...")
        result = client.generate_text(judge_prompt)
        
        print("=== JUDGE RESPONSE DEBUG ===")
        print(judge_prompt)
        print(result)
        print("=== END JUDGE RESPONSE ===")
        
        if result:
            if args.judge_mode == "outcome":
                score = judge_evaluator.extract_outcome_score_from_judge_response(result)
                print(f"Sample {i+1} score: {score}")
            elif args.judge_mode == "turn":
                scores = judge_evaluator.extract_turn_scores_from_judge_response(result, len(sample_turns))
                print(f"Sample {i+1} scores: {scores}")
        else:
            print(f"Failed to evaluate sample {i+1}")
        
        print("-" * 100)


if __name__ == "__main__":
    main()