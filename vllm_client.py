#!/usr/bin/env python3
"""
VLLM Client for Search-Enabled Reasoning

A client for interacting with VLLM servers to perform search-enabled reasoning
and answer evaluation with scoring capabilities.
"""

import json
import re
import sys
from typing import List, Tuple

from openai import OpenAI


# ============================================================================
# Configuration Constants
# ============================================================================

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8002
DEFAULT_MODEL = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_DATA_PATH = "./outputs/log_val_traj/val-search-r1-ppo-qwen2.5-7b-em-gae_20250814_043740/trajectories_val_batch_0.json"


# ============================================================================
# VLLM Client Class
# ============================================================================

class VLLMClient:
    """Client for interacting with VLLM servers."""
    
    def __init__(self, port: int = DEFAULT_PORT, model: str = DEFAULT_MODEL):
        """Initialize VLLM client.
        
        Args:
            port: Port number for VLLM server
            model: Model name to use
        """
        self.port = port
        self.model = model
        self.api_base = f"http://{DEFAULT_HOST}:{port}/v1"
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=self.api_base,
        )
    
    def generate_text(self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
        """Generate text using the VLLM server.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        print(f"Connecting to VLLM server at: {self.api_base}")
        
        chat_response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        
        return chat_response.choices[0].message.content


# ============================================================================
# Prompt Creation Functions
# ============================================================================

def create_judge_prompt(prompt: str, turns: List[str]) -> str:
    """Create evaluation prompt for prompt-response assessment.
    
    Args:
        prompt: Original prompt
        turns: Pre-divided turn texts
        
    Returns:
        Formatted judge prompt
    """
    print(f"Creating judge prompt with {len(turns)} turns")
    
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
    
    print("JUDGE PROMPT:")
    print(judge_prompt)
    print("=" * 100)
    
    return judge_prompt


# ============================================================================
# Data Loading Functions
# ============================================================================

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


def get_sample_data(num_samples: int = 10, json_file: str = DEFAULT_DATA_PATH) -> List[Tuple[str, List[str]]]:
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
        prompt = extract_prompt_from_chat_format(raw_prompt)
        turn_texts = sample["turn_texts"]
        samples.append((prompt, turn_texts))
    
    return samples


# ============================================================================
# Utility Functions
# ============================================================================

def generate_text(prompt: str, port: int = DEFAULT_PORT) -> str:
    """Legacy function for backward compatibility.
    
    Args:
        prompt: Input prompt
        port: Server port
        
    Returns:
        Generated text
    """
    client = VLLMClient(port)
    return client.generate_text(prompt)


def parse_arguments() -> int:
    """Parse command line arguments.
    
    Returns:
        Port number from command line or default
    """
    return int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PORT


# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main function to run the client evaluation."""
    port = parse_arguments()
    samples = get_sample_data(5)
    client = VLLMClient(port)
    
    print(f"Using port: {port}")
    print(f"Evaluating {len(samples)} samples...")
    print("=" * 100)
    
    for i, (sample_prompt, sample_turns) in enumerate(samples):
        print(f"\nSAMPLE {i+1}:")
        print("-" * 100)
        
        judge_prompt = create_judge_prompt(sample_prompt, sample_turns)
        
        try:
            print(f"Evaluating sample {i+1}...")
            result = client.generate_text(judge_prompt)
            print(f"EVALUATION RESULT FOR SAMPLE {i+1}:")
            print(result)
        except Exception as e:
            print(f"Error evaluating sample {i+1}: {e}")
        
        print("-" * 100)


if __name__ == "__main__":
    main()