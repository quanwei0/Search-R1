# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
- Preprocess data and split the training set into 75% for training RM and 25% for validting RM.
- All the training data is used to train SFT and RL.
- Both chosen and rejected is used to train SFT
"""

import argparse
import os
import json
import random
from typing import List, Dict, Optional

import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from verl.utils.fs import copy, makedirs


SYSTEM_PROMPT1 = """You are a skilled expert in evaluating search trajectories for multi-step question answering (QA) tasks. \n

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
- Provide a short *think* explanation (one paragraph in <think></think>).
- Then output **STRICT JSON only**, in this exact schema:
```json
{
  "scores for A": {
    "Search Relevance": <0–10>,
    "Information Extraction": <0–10>,
    "Progress Toward Answer": <0–10>,
    "Search Efficiency": <0–10>,
    "Reasoning Quality": <0–10>,
    "final score": <0–10>,
  },
"scores for B": {
    "Search Relevance": <0–10>,
    "Information Extraction": <0–10>,
    "Progress Toward Answer": <0–10>,
    "Search Efficiency": <0–10>,
    "Reasoning Quality": <0–10>,
    "final score": <0–10>,
  },
  "final_answer": "A" | "B",
}
"""

SYSTEM_PROMPT2 = """You are a skilled expert in evaluating search trajectories for multi-step question answering (QA) tasks. \n

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
boxed, e.g., \\boxed{x, x} if there exists 2 responeses>
"""

PROMPT_TEMPLATE = '''
Question: {question}
Current Search State: {state}

Candidate Trajectory A:
{response_A}

Candidate Trajectory B:
{response_B}
'''


def process_state(data, mode="original", llm_model=None, tokenizer=None):
    """
    Process the state based on different modes.
    
    Args:
        data: Dictionary containing the data with 'state', 'expert_response', 'policy_response' keys
        mode: Processing mode - "original", "last_action", or "llm_summary"
        llm_model: Language model for summarization (required if mode="llm_summary")
        tokenizer: Tokenizer for the model (required if mode="llm_summary")
    
    Returns:
        Processed state string
    """
    import re
    original_state = data.get("state", "")
    
    if mode == "original":
        return original_state
    
    elif mode == "last_action":
        # Extract state from trajectory - combine first prompt and last observation
        if data.get("horizon_index") > 0:
            last_observation = data['history'][-1]
            new_state = f"{original_state}\n\nLast observation:\n{last_observation}"
            return new_state
        else:
            return original_state
    
    elif mode == "llm_summary":
        pass
        # Use LLM to summarize the trajectory and create a new state
        # if not llm_model or not tokenizer:
        #     raise ValueError("LLM model and tokenizer required for llm_summary mode")
        
        # # Prepare prompt for summarization
        # summary_prompt = f"""Given the following search trajectory for a question-answering task, summarize the current state of knowledge.
        # Focus on: 
        # 1. What information has been gathered so far
        # 2. What key facts have been established
        # 3. What information gaps remain

        # Original Question/State:
        # {original_state}

        # Search Trajectory:
        # {response_to_parse}

        # Provide a concise summary of the current state:"""
        
        # # Format for model
        # messages = [
        #     {"role": "system", "content": "You are a helpful assistant that summarizes search trajectories concisely."},
        #     {"role": "user", "content": summary_prompt}
        # ]
        
        # # Tokenize
        # formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # inputs = tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=4096)
        
        # if hasattr(llm_model, 'device'):
        #     inputs = inputs.to(llm_model.device)
        
        # # Generate summary
        # import torch
        # with torch.no_grad():
        #     outputs = llm_model.generate(
        #         **inputs,
        #         max_new_tokens=256,
        #         temperature=0.3,  # Lower temperature for more focused summaries
        #         do_sample=True,
        #         top_p=0.95,
        #         pad_token_id=tokenizer.pad_token_id,
        #         eos_token_id=tokenizer.eos_token_id
        #     )
        
        # # Decode summary
        # summary = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # # Combine original state with summary
        # new_state = f"{original_state}\n\nCurrent knowledge state:\n{summary}"
        # return new_state
    
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'original', 'last_action', or 'llm_summary'")


def generate_llm_responses(
    prompts: List[str], 
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    batch_size: int = 8,
    max_new_tokens: int = 512,
    temperature: float = 0.7
) -> List[str]:
    """Generate LLM responses for a batch of prompts."""
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto"
    # )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    responses = []
    
    # Process in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating LLM responses"):
        batch_prompts = prompts[i:i+batch_size]
        
        # Format with system prompt
        messages_batch = []
        for prompt in batch_prompts:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT2},
                {"role": "user", "content": prompt}
            ]
            messages_batch.append(messages)
            print(messages)

        # Tokenize
        formatted_prompts = [
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in messages_batch
        ]
        
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode responses
        for j, output in enumerate(outputs):
            # Remove the input prompt from the output
            response = tokenizer.decode(
                output[inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            responses.append(response)
    
    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()
    
    return responses


def process_single_item(data_item, idx, split, state_mode="original"):
    """
    Process a single data item into the format needed for reward model training.
    
    Args:
        data_item: Single item from the dataset
        idx: Index of the item
        split: 'train' or 'test'
        state_mode: How to process the state
    
    Returns:
        Processed data dictionary
    """
    # Process the state based on the selected mode
    processed_state = process_state(data_item, mode=state_mode)
    
    # Randomly decide whether to put expert response as A or B
    swap_positions = random.random() < 0.5
    
    if swap_positions:
        response_A = data_item.get("policy_response", "")
        response_B = data_item.get("expert_response", "")
        chosen_label = "B"  # Expert is B (better)
    else:
        response_A = data_item.get("expert_response", "")
        response_B = data_item.get("policy_response", "")
        chosen_label = "A"  # Expert is A (better)
    
    # Format the evaluation prompt
    formatted_prompt = PROMPT_TEMPLATE.format(
        question=data_item.get("question", ""),
        state=processed_state,
        response_A=response_A,
        response_B=response_B
    )
    
    # Create structured data item similar to GSM8K format
    processed_item = {
        "data_source": "search_pairwise_Iter1",
        "prompt": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT2
            },
            {
                "role": "user",
                "content": formatted_prompt
            }
        ],
        "ability": "trajectory_evaluation",
        "reward_model": {
            "style": "rule",
            "ground_truth": chosen_label,
            "rejected": "A" if chosen_label == "B" else "B"
        },
        "extra_info": {
            "split": split,
            "index": idx,
            "original_question": data_item.get("question", ""),
            "ground_truth": data_item.get("ground_truth", []),
            "horizon_index": data_item.get("horizon_index", 0),
            "total_generation_steps": data_item.get("total_generation_steps", 0),
            "swap_positions": swap_positions
        }
    }
    
    return processed_item


def generate_rm_dataset(
    target_hdfs_path_dir, 
    local_dir="~/data/full_hh_rlh/rm",
    model_name: Optional[str] = None,
    generate_responses: bool = False,
    state_mode: str = "original",
    max_samples: Optional[int] = None,
    filter_empty: bool = True
):
    """
    Generate reward model training dataset from pairwise trajectory data.
    
    Args:
        target_hdfs_path_dir: HDFS directory to save to
        local_dir: Local directory to save parquet files
        model_name: Model name for LLM response generation
        generate_responses: Whether to generate LLM responses
        state_mode: How to process states ('original', 'last_action', 'llm_summary')
        max_samples: Maximum number of samples to process (for testing)
    """
    # Load data from local pairwise_data.json file
    json_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "pairwise_data.json")
    
    with open(json_path, 'r') as f:
        pairwise_data = json.load(f)
    
    # Filter out items with empty policy_response or expert_response
    original_count = len(pairwise_data)
    filtered_count = 0
    
    if filter_empty:
        pairwise_data = [
            item for item in pairwise_data 
            if (item.get("policy_response", "").strip() != "" and 
                item.get("expert_response", "").strip() != "")
        ]
        filtered_count = original_count - len(pairwise_data)
        
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} items with empty policy_response or expert_response")
            print(f"Remaining items: {len(pairwise_data)}")
    else:
        print("Skipping filtering - processing all items including those with empty responses")
    
    # Limit samples if specified (for testing)
    if max_samples:
        pairwise_data = pairwise_data[:max_samples]
    
    # Split data into train (75%) and test (25%)
    total_samples = len(pairwise_data)
    train_size = int(total_samples * 0.75)
    
    train_data = pairwise_data[:train_size]
    test_data = pairwise_data[train_size:]
    
    local_dir = os.path.expanduser(local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    for dataset, split_name in zip([train_data, test_data], ["train", "test"], strict=True):
        processed_data = []
        
        # Process each item using the structured format
        for idx, data_item in enumerate(tqdm(dataset, desc=f"Processing {split_name} data")):
            processed_item = process_single_item(data_item, idx, split_name, state_mode)
            processed_data.append(processed_item)
        
        # Convert to DataFrame for saving
        df = pd.DataFrame(processed_data)
        
        # Save to parquet file
        local_path = os.path.join(local_dir, split_name + ".parquet")
        
        df.to_parquet(path=local_path)
        print(f"Saved {len(df)} {split_name} samples to {local_path}")
        
        if target_hdfs_path_dir is not None:
            hdfs_dir = target_hdfs_path_dir + "/" + split_name + ".parquet"
            makedirs(hdfs_dir)
            copy(local_path, hdfs_dir)
            print(f"Copied to HDFS: {hdfs_dir}")
    
    # Print summary
    print("\n" + "="*50)
    print("Dataset generation complete!")
    if 'original_count' in locals():
        print(f"Original samples: {original_count}")
        if filtered_count > 0:
            print(f"Filtered samples: {filtered_count}")
    print(f"Total samples processed: {total_samples}")
    print(f"Train samples: {train_size}")
    print(f"Test samples: {total_samples - train_size}")
    print(f"State processing mode: {state_mode}")
    print(f"Output directory: {local_dir}")
    print("="*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str, default="data/pairwise_data_iter1",
                        help="Local directory to save processed data")
    parser.add_argument("--hdfs_dir", type=str, required=False, default=None,
                        help="HDFS directory to save data")
    parser.add_argument("--model_name", type=str, default=None,
                        help="Model to use for generating responses")
    parser.add_argument("--generate_responses", action="store_true",
                        help="Whether to generate LLM responses for the prompts")
    parser.add_argument("--state_mode", type=str, default="original",
                        choices=["original", "last_action", "llm_summary"],
                        help="How to process the state: 'original' (use as-is), 'last_action' (extract last observation), 'llm_summary' (use LLM to summarize)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process (for testing)")
    parser.add_argument("--no_filter", action="store_true",
                        help="Skip filtering of empty responses")

    args = parser.parse_args()

    generate_rm_dataset(
        args.hdfs_dir, 
        args.local_dir,
        model_name=args.model_name,
        generate_responses=args.generate_responses,
        state_mode=args.state_mode,
        max_samples=args.max_samples,
        filter_empty=not args.no_filter
    )
