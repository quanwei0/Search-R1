import json
import glob
import os
import re
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from tqdm.auto import tqdm
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Updated scoring system prompt based on reward_data_prepare.py
SYSTEM_PROMPT = """You are a skilled expert in evaluating search trajectories for multi-step question answering (QA) tasks. \n

Given the original question, the current search state, and two candidate partial trajectories, your goal is to score each trajectory according to the [rubric] below, briefly in a concise thinking process.

#### Rubric ####

1. Search Relevance (0–10 points)  
   - 9–10 (Excellent): The query directly targets the current knowledge gap, uses precise, relevant keywords, avoids redundancy, and fully aligns with the missing information.  
   - 7–8 (Good): Addresses most of the gap with only minor redundancy or slightly suboptimal keyword choice.  
   - 5–6 (Fair): Partially relevant; contains general or tangential terms; misses key specificity.  
   - 3–4 (Poor): Loosely related to the gap; includes irrelevant or redundant elements.  
   - 0–2 (Very Poor): Completely off-topic or fails to address the information gap.  
   *Example*: If the missing info is “birthplace of person X” and the query searches for “person X career awards,” it is “Poor.”

2. Information Extraction (0–10 points) 
   - 9–10 (Excellent): Extracted info is accurate, relevant, concise; key facts identified; noise removed.  
   - 7–8 (Good): Mostly accurate and relevant; minimal noise or minor omissions.  
   - 5–6 (Fair): Some relevant facts present but also includes noise or inaccuracies.  
   - 3–4 (Poor): Mostly noise or irrelevant details; few useful facts.  
   - 0–2 (Very Poor): No useful or correct information extracted.  
   *Example*: If three relevant facts are mixed with unrelated content, score “Fair.”

3. Progress Toward Answer (0–10 points)  
   - 9–10 (Excellent): Fills a major knowledge gap or completes the final hop toward the answer.  
   - 7–8 (Good): Makes significant progress, covering at least one important missing piece.  
   - 5–6 (Fair): Partial progress; minor gaps filled but critical ones remain.  
   - 3–4 (Poor): Minimal progress; tangentially relevant but not clearly advancing the answer.  
   - 0–2 (Very Poor): No measurable progress toward the answer.  
   *Example*: Finding the bridge entity needed for the final hop is “Excellent”; retrieving tangential facts is “Poor.”

4. Search Efficiency (0–10 points)  
   - 9–10 (Excellent): Fully optimized; no wasted or redundant queries; maximum information gain per action.  
   - 7–8 (Good): Mostly efficient; only minor redundancy.  
   - 5–6 (Fair): Some inefficiency or redundancy.  
   - 3–4 (Poor): Significant redundancy or wasted actions.  
   - 0–2 (Very Poor): Highly inefficient or irrelevant actions.  
   *Example*: Repeating a similar query with no new angle is “Poor.”

5. Reasoning Quality (0–10 points)  
   - 9–10 (Excellent): Logical, well-structured, grounded in evidence, builds coherently on prior knowledge.  
   - 7–8 (Good): Mostly logical and grounded; minor reasoning gaps.  
   - 5–6 (Fair): Reasoning is partially correct but contains weak links or gaps.  
   - 3–4 (Poor): Flawed reasoning; unclear or unsupported connections.  
   - 0–2 (Very Poor): Illogical or unsupported; no meaningful connection to prior steps.  
   *Example*: If extracted info is correct but misapplied in reasoning, it may be “Poor.”
   
#### Output Instructions ####
- Provide a short *reasoning* explanation (put in <reasoning></reasoning>).
- Scores: <the overall comprehensive score of all responses in order, separate by comma in the
boxed, e.g., \\boxed{x, x} if there exists 2 candidate trajectories or \\boxed{x} if there exists 1 candidate trajectory>.
"""

PROMPT_TEMPLATE = '''
Question: {question}
Current Search State: {state}

Candidate Trajectory:
{response}

'''

class TrajectoryScorer:
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9):
        """
        Initialize trajectory scorer using vLLM for efficient inference
        
        Args:
            model_name: Hugging Face model name
            tensor_parallel_size: Number of GPUs to use for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading model with vLLM: {model_name}")
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            dtype="bfloat16",
            trust_remote_code=True
        )
        
        # Set up sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=1024,
            stop=None
        )
        
        print(f"Model loaded successfully with vLLM")
    
    def parse_scores(self, response: str) -> Dict[str, float]:
        """Parse scores from model response"""
        scores = {
            'overall_score': 0.0
        }
        
        # Look for boxed scores
        boxed_pattern = r'\\boxed\{([^}]+)\}'
        match = re.search(boxed_pattern, response)
        
        if match:
            try:
                score_str = match.group(1)
                score_parts = [float(x.strip()) for x in score_str.split(',')]
                
                if len(score_parts) >= 1:
                    scores['overall_score'] = score_parts[0] / 10.0
                    
            except (ValueError, IndexError) as e:
                print(f"Error parsing scores: {e}")
                print(f"Response: {response}")
        
        return scores

    def format_prompt(self, question: str, ground_truth: List[str], trajectory: list, state: str) -> str:
        """Format a single prompt for scoring"""
        trajectory = "".join(trajectory) if isinstance(trajectory, list) else str(trajectory)
        user_prompt =  PROMPT_TEMPLATE.format(
            question=question,
            state=state,
            response=trajectory
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Format prompt
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        return formatted_prompt

    def score_batch(self, samples: List[Dict[str, Any]], question: str, ground_truth: List[str], state: str) -> List[Dict[str, float]]:
        """Score a batch of samples using vLLM"""
        try:
            # Prepare prompts for all samples
            prompts = []
            for sample in samples:
                trajectory = sample.get('turn_texts')
                formatted_prompt = self.format_prompt(question, ground_truth, trajectory, state)
                prompts.append(formatted_prompt)
            
            # Generate responses for all prompts at once
            outputs = self.model.generate(prompts, self.sampling_params)
            
            # Parse results
            results = []
            for output in outputs:
                response_text = output.outputs[0].text
                scores = self.parse_scores(response_text)
                scores['raw_response'] = response_text
                results.append(scores)
            
            return results
            
        except Exception as e:
            print(f"Error scoring batch: {e}")
            # Return default scores for all samples
            return [{
                'overall_score': 0.0,
                'raw_response': f"Error: {str(e)}"
            } for _ in samples]


def score_trajectory_files(
    directory_path: str, 
    model_name: str = "XinnanZhang/search-irl-grpo-qwen2.5-3b-iter1-float32",
    tensor_parallel_size: int = 2,
    gpu_memory_utilization: float = 0.9,
    save_interval: int = 10,
    max_entries: Optional[int] = None
):
    """Score all samples in trajectory files"""
    pattern = os.path.join(directory_path, "trajectories_val_batch*.json")
    json_files = sorted(glob.glob(pattern))
    
    if not json_files:
        print(f"No files matching pattern found in {directory_path}")
        return
    
    # Initialize scorer
    scorer = TrajectoryScorer(
        model_name=model_name, 
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization
    )
    
    all_scored_data = []
    processed_count = 0
    
    print(f"Scoring samples from {len(json_files)} files...")
    
    for file_idx, file_path in enumerate(json_files):
        print(f"Processing file {file_idx + 1}/{len(json_files)}: {os.path.basename(file_path)}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for entry_idx, entry in enumerate(data):
                if max_entries and processed_count >= max_entries:
                    break
                    
                question_prompt = entry['prompt']
                ground_truth = entry['ground_truth']
                samples = entry['samples']
                
                # Extract question from prompt
                question_start = question_prompt.rfind("Question: ")
                if question_start != -1:
                    question = question_prompt[question_start + 10:].strip()
                    question = question.split('\n')[0]
                else:
                    question = "Question not found in prompt"
                
                print(f"  Scoring entry {entry_idx + 1}/{len(data)} with {len(samples)} samples...")
                
                # Score all samples for this entry
                scored_samples = scorer.score_batch(samples, question, ground_truth, state=question_prompt)
                
                # Add scores to samples
                for sample, scores in zip(samples, scored_samples):
                    sample.update(scores)
                
                # Create scored entry
                scored_entry = entry.copy()
                scored_entry['samples'] = samples
                all_scored_data.append(scored_entry)
                
                processed_count += 1
                
                # Save checkpoint
                if processed_count % save_interval == 0:
                    checkpoint_file = os.path.join(directory_path, f"trajectory_scores_checkpoint.json")
                    with open(checkpoint_file, 'w', encoding='utf-8') as f:
                        json.dump(all_scored_data, f, indent=2, ensure_ascii=False)
                    print(f"  Saved checkpoint with {processed_count} entries")
            
            if max_entries and processed_count >= max_entries:
                break
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    # Save final results
    output_file = os.path.join(directory_path, "trajectory_scores.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_scored_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nCompleted! Scored {processed_count} entries saved to: {output_file}")
    
    # Calculate statistics
    all_scores = {
        'overall_score': []
    }
    
    for entry in all_scored_data:
        for sample in entry['samples']:
            for key in all_scores.keys():
                if key in sample:
                    all_scores[key].append(sample[key])
    
    print("\nScoring Statistics:")
    for key, values in all_scores.items():
        if values:
            avg_score = sum(values) / len(values)
            print(f"{key.replace('_', ' ').title()}: {avg_score:.4f}")
    
    # Clean up GPU memory
    del scorer.model
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Score trajectory samples using detailed rubric")
    parser.add_argument("--directory", 
                        default="outputs/log_val_traj/nq-search-r1-quan-7b-ckpt1-sampled-512-BoN16_20250824_224019/", 
                        help="Directory containing trajectory files")
    parser.add_argument("--model", default="XinnanZhang/search-irl-grpo-qwen2.5-3b-iter1-float32", 
                        help="Model name for scoring")
    parser.add_argument("--tensor-parallel-size", type=int, default=2,
                        help="Number of GPUs to use for tensor parallelism")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6,
                        help="Fraction of GPU memory to use")
    parser.add_argument("--save-interval", type=int, default=10, 
                        help="Save checkpoint every N entries")
    parser.add_argument("--max-entries", type=int, default=None,
                        help="Maximum number of entries to process (for testing)")
    
    args = parser.parse_args()
    
    score_trajectory_files(
        args.directory, 
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        save_interval=args.save_interval,
        max_entries=args.max_entries
    )


if __name__ == "__main__":
    main()