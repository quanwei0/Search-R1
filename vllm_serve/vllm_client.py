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
        self.host = host
        self.port = port
        self.model = model
        self.api_base = f"http://{self.host}:{self.port}/v1"
        self.logger = logging.getLogger(__name__)
        self.client = self._initialize_client()
    
    def _initialize_client(self) -> Optional[OpenAI]:
        """Initialize OpenAI client for VLLM server.
        
        Returns:
            Configured OpenAI client or None if initialization failed
        """
        try:
            return OpenAI(
                api_key="EMPTY",
                base_url=self.api_base,
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            return None
    
    def generate_text(self, prompt: str, max_tokens: int = 2048) -> Optional[str]:
        """Generate text using the VLLM server.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response or None if error occurred
        """
        try:
            self.logger.info(f"Connecting to VLLM server at: {self.api_base}")
            
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
    def create_judge_prompt(prompt: str, turns: List[str]) -> str:
        """Create evaluation prompt for prompt-response assessment.
        
        Args:
            prompt: Original prompt
            turns: Pre-divided turn texts
            
        Returns:
            Formatted judge prompt
        """
        
        turns_text = ""
        for i, turn in enumerate(turns, 1):
            turns_text += f"TURN {i}:\n{turn}\n\n"
        
        judge_prompt = f"""
Evaluate how effectively each turn of the response addresses the given prompt.

PROMPT:
{prompt}

{turns_text}
Follow these instructions:

1) First provide step-by-step reasoning, then assign scores turn by turn using the exact output format below (must be exact and include the tags):

<reasoning>
[Your evaluation of each turn here]
</reasoning>

<score>
Turn1: X.X
Turn2: X.X
...
</score>

2) Last turn only:
   - Format check: must include only <think>...</think> for reasoning followed by <answer>...</answer> for the final answer, in this exact sequence. No other tags are allowed. Apply a penalty if tags are missing, out of order, or extra tags are used.
   - Answer evaluation: verify that the final answer is factually correct given the original question.

3) Non-last turns:
   - Format check: must include only three tags in this exact sequence:
       1) <think>...</think> for reasoning
       2) <search>...</search> for the search query
       3) <information>...</information> for retrieved results
     No other tags are allowed. Apply a penalty if tags are missing, out of order, or extra tags are used.
   - Reasoning evaluation: assess the quality, clarity, and logic of the content in <think>.
   - Query evaluation: judge the quality and relevance of the <search> query, and compare it with the previous turn’s query (improved, declined, or same). Penalize unjustified repetition.

4) Scoring:
   - Assign a score to each turn in the range [-1, 1].
   - Use the following scale:
     * 1.0 = Excellent (correct format, strong reasoning, relevant and accurate)
     * 0.5 = Adequate (mostly correct, minor flaws)
     * 0.0 = Neutral (unclear, limited contribution)
     * -0.5 = Poor (errors, weak reasoning or queries, format issues)
     * -1.0 = Very poor (misleading, harmful, or completely wrong)
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
                    scores = [0.0] * num_turns
        
        except Exception:
            pass
            scores = [0.0] * num_turns
        
        return scores


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
    def get_sample_data(num_samples: int = 5, json_file: str = DEFAULT_DATA_PATH) -> List[Tuple[str, List[str]]]:
        """Get multiple sample prompts and turn texts from JSON file.
        
        Args:
            num_samples: Number of samples to retrieve
            json_file: Path to JSON file containing samples
            
        Returns:
            List of (prompt, turn_texts) tuples
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for i in range(min(num_samples, len(data))):
            sample = data[i]
            raw_prompt = sample["prompt"]
            
            # Extract the actual user prompt from chat format
            prompt = DataProcessor.extract_prompt_from_chat_format(raw_prompt)
            turn_texts = sample["turn_texts"]
            samples.append((prompt, turn_texts))
        
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
    for i, (sample_prompt, sample_turns) in enumerate(samples):
        print(f"\nSAMPLE {i+1}:")
        print("-" * 100)
        
        judge_prompt = judge_evaluator.create_judge_prompt(sample_prompt, sample_turns)
        
        print(f"Evaluating sample {i+1}...")
        result = client.generate_text(judge_prompt)
        
        if result:
            scores = judge_evaluator.extract_turn_scores_from_judge_response(result, len(sample_turns))
            print(f"Sample {i+1} scores: {scores}")
        else:
            print(f"Failed to evaluate sample {i+1}")
        
        print("-" * 100)


if __name__ == "__main__":
    main()