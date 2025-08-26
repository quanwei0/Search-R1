import json
import glob
import os
from pathlib import Path
from collections import defaultdict, Counter
import math
import re

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
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    merged_data.extend(data)
                else:
                    merged_data.append(data)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    output_path = os.path.join(directory_path, output_filename)
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSuccessfully merged {len(merged_data)} entries to {output_path}")
        
    except Exception as e:
        print(f"Error writing merged file: {e}")

def calculate_pass_at_k_metrics(directory_path, k_values=[1, 4, 8, 16]):
    """
    Calculate pass@k metrics and best-of-n metrics for different k values.
    """
    pattern = os.path.join(directory_path, "trajectories_val_batch*.json")
    json_files = sorted(glob.glob(pattern))
    
    if not json_files:
        print(f"No files matching pattern found in {directory_path}")
        return
    
    # Group samples by prompt
    prompt_samples = defaultdict(list)
    
    print(f"Loading {len(json_files)} files...")
    for file_path in json_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for entry in data:
                    prompt = entry['prompt']
                    samples = entry['samples']
                    
                    # Add samples to the prompt group
                    prompt_samples[prompt].extend(samples)
                    
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue
    
    print(f"Found {len(prompt_samples)} unique prompts")
    
    # Calculate pass@k and best-of-n for each k value
    results = {}
    
    for k in k_values:
        pass_at_k_scores = []
        best_of_n_scores = []
        valid_prompts = 0
        
        for prompt, samples in prompt_samples.items():
            n = len(samples)
            
            # Only consider prompts with at least k samples
            if n < k:
                continue
                
            # Take first k samples for this prompt
            k_samples = samples[:k]
            
            # Calculate pass@k
            c = sum(1 for sample in k_samples if sample.get('answer_reward', 0) == 1)
            pass_k = calculate_pass_at_k(k, c, k)
            pass_at_k_scores.append(pass_k)
            
            # Calculate best-of-n: select sample with highest reward
            best_sample = max(k_samples, key=lambda x: x.get('reward', float('-inf')))
            best_of_n_correct = 1 if best_sample.get('answer_reward', 0) == 1 else 0
            best_of_n_scores.append(best_of_n_correct)
            
            valid_prompts += 1
        
        # Average across all prompts
        if valid_prompts > 0:
            avg_pass_at_k = sum(pass_at_k_scores) / len(pass_at_k_scores)
            avg_best_of_n = sum(best_of_n_scores) / len(best_of_n_scores)
            
            results[f'pass@{k}'] = avg_pass_at_k
            results[f'best_of_{k}'] = avg_best_of_n
            
            print(f"pass@{k}: {avg_pass_at_k:.4f} (based on {valid_prompts} prompts)")
            print(f"best_of_{k}: {avg_best_of_n:.4f} (based on {valid_prompts} prompts)")
        else:
            print(f"pass@{k}: No valid prompts with at least {k} samples")
    
    # Additional statistics
    total_samples = sum(len(samples) for samples in prompt_samples.values())
    samples_per_prompt = [len(samples) for samples in prompt_samples.values()]
    
    print(f"\nSummary Statistics:")
    print(f"Total prompts: {len(prompt_samples)}")
    print(f"Total samples: {total_samples}")
    print(f"Avg samples per prompt: {total_samples / len(prompt_samples):.2f}")
    print(f"Min samples per prompt: {min(samples_per_prompt)}")
    print(f"Max samples per prompt: {max(samples_per_prompt)}")
    
    # Count correct answers overall
    total_correct = 0
    for samples in prompt_samples.values():
        total_correct += sum(1 for sample in samples if sample.get('answer_reward', 0) == 1)
    
    print(f"Overall accuracy: {total_correct / total_samples:.4f}")
    
    # Save results to JSON
    output_file = os.path.join(directory_path, "pass_at_k_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            'metrics': results,
            'statistics': {
                'total_prompts': len(prompt_samples),
                'total_samples': total_samples,
                'avg_samples_per_prompt': total_samples / len(prompt_samples),
                'min_samples_per_prompt': min(samples_per_prompt),
                'max_samples_per_prompt': max(samples_per_prompt),
                'overall_accuracy': total_correct / total_samples
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results

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
    
    # Extract answers from extracted_reward
    answers = []
    for sample in k_samples:
        answer = sample.get('extracted_reward', '').strip().lower()
        answers.append(answer)

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
            'confidence': winning_count / k,
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
        
        total_weight = sum(answer_weights.values())
        
        return {
            'winning_answer': winning_answer,
            'winning_sample': best_sample,
            'winning_sample_idx': winning_sample_idx,
            'vote_weights': dict(answer_weights),
            'confidence': winning_weight / total_weight if total_weight > 0 else 0,
            'is_tie': False
        }

def calculate_majority_voting_metrics(directory_path, k_values=[1, 4, 8, 16], strategies=['simple', 'weighted']):
    """Calculate majority voting metrics for different k values."""
    pattern = os.path.join(directory_path, "trajectories_val_batch*.json")
    json_files = sorted(glob.glob(pattern))
    
    if not json_files:
        print(f"No files matching pattern found in {directory_path}")
        return {}
    
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
            confidence_sum = 0
            
            for prompt, entry_data in prompt_samples.items():
                if len(entry_data['samples']) < k:
                    continue
                
                # Get ground truth from entry and normalize to lowercase
                ground_truth = entry_data.get('ground_truth')
                if not ground_truth:
                    continue
                ground_truth_lower = [str(gt).strip().lower() for gt in ground_truth]
            
                samples = entry_data['samples']
                
                vote_result = majority_vote_samples(samples, k, strategy)
                
                # Check correctness - see if predicted answer matches any ground truth
                predicted = vote_result['winning_answer']
                is_correct = predicted in ground_truth_lower
                if is_correct:
                    correct_votes += 1
                
                if vote_result.get('is_tie', False):
                    tie_count += 1
                
                confidence_sum += vote_result['confidence']
                total_prompts += 1
                
                # Calculate pass@k for this sample
                individual_correct = [ans in ground_truth_lower for ans in [s.get('extracted_reward', '').strip().lower() for s in samples[:k]]]
                num_correct = sum(individual_correct)
                pass_at_k = calculate_pass_at_k(k, num_correct, k) if num_correct > 0 else 0.0
                
                # Calculate best-of-k (highest reward sample)
                best_sample_idx = max(range(k), key=lambda i: samples[i].get('reward', float('-inf')))
                best_of_k_answer = samples[best_sample_idx].get('extracted_reward', '').strip().lower()
                best_of_k_correct = best_of_k_answer in ground_truth_lower
                
                # Save detailed results
                sample_result = {
                    'prompt': prompt,
                    'strategy': strategy,
                    'k': k,
                    'ground_truth': ground_truth,
                    'ground_truth_lower': ground_truth_lower,
                    'individual_extracted_rewards': [s.get('extracted_reward', '').strip() for s in samples[:k]],
                    'individual_extracted_rewards_lower': [s.get('extracted_reward', '').strip().lower() for s in samples[:k]],
                    'individual_rewards': [s.get('reward', 0) for s in samples[:k]],
                    'individual_correct': individual_correct,
                    'num_correct': num_correct,
                    'best_of_k_answer': best_of_k_answer,
                    'best_of_k_correct': best_of_k_correct,
                    'majority_vote_result': predicted,
                    'vote_confidence': vote_result['confidence'],
                    'is_tie': vote_result.get('is_tie', False),
                    'pass_at_k': pass_at_k,
                    'majority_vote_correct': is_correct,
                    'vote_counts': vote_result.get('vote_counts', {}),
                    'vote_weights': vote_result.get('vote_weights', {})
                }
                detailed_results.append(sample_result)
            
            if total_prompts > 0:
                accuracy = correct_votes / total_prompts
                avg_confidence = confidence_sum / total_prompts
                tie_rate = tie_count / total_prompts
                
                strategy_results[f'majority_vote@{k}'] = accuracy
                strategy_results[f'majority_confidence@{k}'] = avg_confidence
                strategy_results[f'majority_tie_rate@{k}'] = tie_rate
                
                print(f"majority_vote@{k}: {accuracy:.4f} (confidence: {avg_confidence:.3f}, ties: {tie_rate:.3f})")
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
    directory = "outputs/log_val_traj/nq-search-r1-quan-7b-ckpt1-sampled-512-BoN16_20250824_224019/"
    
    # Merge trajectory files
    merge_trajectory_files(directory)
    
    # Calculate pass@k for common k values
    k_values = [1, 4, 8, 16]
    results = calculate_pass_at_k_metrics(directory, k_values)
    
    # Calculate majority voting metrics
    mv_results = calculate_majority_voting_metrics(directory, k_values)