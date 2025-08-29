import json
import glob
import os
from pathlib import Path
from collections import defaultdict, Counter
import math
import re
from verl.utils.reward_score.qa_em_new import normalize_answer

def calculate_pass_at_k(n, c, k):
    """
    Calculate pass@k using the standard formula:
    pass@k = 1 - C(n-c, k) / C(n, k)
    where n is total samples, c is correct samples, k is the subset size
    """
    if n - c < k:
        return 1.0
    if c == 0:
        return 0.0
    
    # Using the mathematical formula: 1 - (n-c choose k) / (n choose k)
    numerator = 1
    denominator = 1
    for i in range(k):
        numerator *= (n - c - i)
        denominator *= (n - i)
    
    return 1 - (numerator / denominator)

def merge_trajectory_files(directory_path, output_filename="merged_trajectories.json"):
    """
    Merge all JSON files starting with 'trajectories_val_batch' in the given directory.
    
    Args:
        directory_path (str): Path to directory containing JSON files
        output_filename (str): Name of output merged file
    """
    pattern = os.path.join(directory_path, "trajectories_val_batch*.json")
    json_files = sorted(glob.glob(pattern))
    
    if not json_files:
        print(f"No files matching pattern found in {directory_path}")
        return
    
    merged_data = []
    
    print(f"Found {len(json_files)} files to merge:")
    for file_path in json_files:
        print(f"  {os.path.basename(file_path)}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                merged_data.extend(data)
            else:
                merged_data.append(data)

    output_path = os.path.join(directory_path, output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSuccessfully merged {len(merged_data)} entries to {output_path}")


def majority_vote_samples(samples, k, strategy='simple'):
    """
    Perform majority voting on k samples using extracted_reward as answers.
    
    Args:
        samples: List of sample dictionaries
        k: Number of samples to use for voting
        strategy: 'simple' or 'weighted'
    
    Returns:
        Dict with voting results
    """
    if len(samples) < k:
        return None
    
    k_samples = samples[:k]
    
    # Extract answers from extracted_reward and normalize them
    answers = []
    for sample in k_samples:
        answer = sample.get('extracted_reward', '').strip()
        normalized_answer = normalize_answer(answer)
        answers.append(normalized_answer)

    if strategy == 'simple':
        # Simple majority voting
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common()
        winning_answer = most_common[0][0]
        winning_count = most_common[0][1]
        
        winning_sample_idx = answers.index(winning_answer)
        winning_sample = k_samples[winning_sample_idx]
        
        is_tie = len(most_common) > 1 and most_common[1][1] == winning_count
        
        return {
            'winning_answer': winning_answer,
            'winning_sample': winning_sample,
            'winning_sample_idx': winning_sample_idx,
            'vote_counts': dict(answer_counts),
            'is_tie': is_tie
        }
    
    elif strategy == 'weighted':
        # Weighted by reward scores
        answer_weights = defaultdict(float)
        answer_samples = defaultdict(list)
        
        for answer, sample in zip(answers, k_samples):
            weight = sample.get('reward', 1.0)
            answer_weights[answer] += weight
            answer_samples[answer].append((sample, weight))
        
        winning_answer = max(answer_weights, key=answer_weights.get)
        winning_weight = answer_weights[winning_answer]
        
        # Get sample with highest weight for winning answer
        winning_candidates = answer_samples[winning_answer]
        best_sample, best_weight = max(winning_candidates, key=lambda x: x[1])
        winning_sample_idx = k_samples.index(best_sample)
        
        return {
            'winning_answer': winning_answer,
            'winning_sample': best_sample,
            'winning_sample_idx': winning_sample_idx,
            'vote_weights': dict(answer_weights),
            'is_tie': False
        }

def calculate_metrics(directory_path, k_values=[1, 4, 8, 16], strategies=['simple', 'weighted']):
    """Calculate majority voting metrics for different k values."""
    pattern = os.path.join(directory_path, "trajectories_val_batch*.json")
    json_files = sorted(glob.glob(pattern))
    
    # Group samples by prompt
    prompt_samples = defaultdict(list)
    
    print(f"Loading {len(json_files)} files for majority voting...")
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            for entry in data:
                prompt = entry['prompt']
                # Store the full entry data, not just samples
                if prompt not in prompt_samples:
                    prompt_samples[prompt] = {
                        'ground_truth': entry.get('ground_truth'),
                        'samples': []
                    }
                prompt_samples[prompt]['samples'].extend(entry['samples'])
                
    
    print(f"Found {len(prompt_samples)} unique prompts")
    
    # Calculate metrics for each strategy and save detailed results
    all_results = {}
    detailed_results = []
    
    for strategy in strategies:
        print(f"\n=== {strategy.title()} Majority Voting ===")
        strategy_results = {}
        
        for k in k_values:
            correct_votes = 0
            total_prompts = 0
            tie_count = 0
            pass_at_k_scores = []
            best_of_n_scores = []
            
            for prompt, entry_data in prompt_samples.items():
                if len(entry_data['samples']) < k:
                    continue
                
                # Get ground truth from entry and normalize using normalize_answer
                ground_truth = entry_data.get('ground_truth')

                ground_truth_normalized = [normalize_answer(str(gt)) for gt in ground_truth]
            
                samples = entry_data['samples']
                
                vote_result = majority_vote_samples(samples, k, strategy)
                
                # Check correctness - see if predicted answer matches any ground truth
                predicted = vote_result['winning_answer']
                is_correct = predicted in ground_truth_normalized
                if is_correct:
                    correct_votes += 1
                
                if vote_result.get('is_tie', False):
                    tie_count += 1
                
                total_prompts += 1
                
                # Calculate pass@k for this sample
                individual_correct = [normalize_answer(s.get('extracted_reward', '')) in ground_truth_normalized for s in samples[:k]]
                num_correct = sum(individual_correct)
                pass_at_k = calculate_pass_at_k(k, num_correct, k) if num_correct > 0 else 0.0
                
                # Calculate best-of-k (highest reward sample)
                best_sample_idx = max(range(k), key=lambda i: samples[i].get('reward', float('-inf')))
                best_of_k_answer = normalize_answer(samples[best_sample_idx].get('extracted_reward', ''))
                best_of_k_correct = best_of_k_answer in ground_truth_normalized
                best_of_n_scores.append(best_of_k_correct)
                pass_at_k_scores.append(pass_at_k)

                sample_result = {
                    'prompt': prompt,
                    'strategy': strategy,
                    'k': k,
                    'ground_truth': ground_truth,
                    'ground_truth_normalized': ground_truth_normalized,
                    'extracted_rewards': [s.get('extracted_reward', '').strip() for s in samples[:k]],
                    'extracted_rewards_normalized': [normalize_answer(s.get('extracted_reward', '')) for s in samples[:k]],
                    'individual_rewards': [s.get('reward', 0) for s in samples[:k]],
                    'individual_correct': individual_correct,
                    'num_correct': num_correct,
                    'best_of_k_answer': best_of_k_answer,
                    'best_of_k_correct': best_of_k_correct,
                    'majority_vote_result': predicted,
                    'is_tie': vote_result.get('is_tie', False),
                    'pass_at_k': pass_at_k,
                    'majority_vote_correct': is_correct,
                    'vote_counts': vote_result.get('vote_counts', {}),
                    'vote_weights': vote_result.get('vote_weights', {})
                    
                }
                detailed_results.append(sample_result)
            
            if total_prompts > 0:
                accuracy = correct_votes / total_prompts
                tie_rate = tie_count / total_prompts
                avg_pass_at_k = sum(pass_at_k_scores) / total_prompts if total_prompts > 0 else 0.0
                avg_best_of_n = sum(best_of_n_scores) / total_prompts if total_prompts > 0 else 0.0

                strategy_results[f'majority_vote@{k}'] = accuracy
                strategy_results[f'majority_tie_rate@{k}'] = tie_rate
                strategy_results[f'pass@{k}'] = avg_pass_at_k
                strategy_results[f'best_of_n@{k}'] = avg_best_of_n

                print(f"majority_vote@{k}: {accuracy:.4f}, ties: {tie_rate:.3f}, pass@{k}: {avg_pass_at_k:.3f}, best_of_n@{k}: {avg_best_of_n:.3f}")
            else:
                print(f"majority_vote@{k}: No valid prompts")
        
        all_results[strategy] = strategy_results
    
    # Save summary results
    output_file = os.path.join(directory_path, "majority_voting_results.json")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save detailed results
    detailed_output_file = os.path.join(directory_path, "majority_voting_detailed_results.json")
    with open(detailed_output_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    print(f"\nMajority voting results saved to: {output_file}")
    print(f"Detailed results saved to: {detailed_output_file}")
    print(f"Detailed results contain {len(detailed_results)} entries")
    
    return all_results

if __name__ == "__main__":
    directory = "outputs/log_val_traj/nq-search-r1-quan-7b-ckpt1-sampled-512-BoN8_20250826_211626"
    
    # Merge trajectory files
    merge_trajectory_files(directory)
    
    # Calculate pass@k for common k values
    k_values = [1, 4, 8, 16]
    # Calculate majority voting metrics
    mv_results = calculate_metrics(directory, k_values)