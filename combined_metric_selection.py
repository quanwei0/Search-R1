#!/usr/bin/env python3
"""
Combined Metric-based Answer Selection
Uses individual uncertainty metrics combined with answer frequency across 16 responses 
to select the best final answer for each prompt.
"""

import json
import numpy as np
import re
from collections import Counter, defaultdict
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import os
import argparse

def extract_answer(sample):
    """Extract answer from extracted_reward field."""
    extracted_reward = sample.get('extracted_reward', None)
    if extracted_reward and extracted_reward.strip():
        return extracted_reward.strip()
    return "NO_ANSWER"

def normalize_answer(answer):
    """Normalize answer for comparison."""
    if not answer or answer == "NO_ANSWER":
        return "NO_ANSWER"
    # Basic normalization
    answer = answer.lower().strip()
    # Remove common prefixes/suffixes
    answer = re.sub(r'^(the|a|an)\s+', '', answer)
    answer = re.sub(r'\s+', ' ', answer)
    return answer

def check_answer_correctness(predicted, ground_truth_list):
    """Check if predicted answer matches any ground truth."""
    predicted_norm = normalize_answer(predicted)
    if predicted_norm == "NO_ANSWER":
        return False
    
    for gt in ground_truth_list:
        gt_norm = normalize_answer(gt)
        if predicted_norm == gt_norm or predicted_norm in gt_norm or gt_norm in predicted_norm:
            return True
    return False

def load_all_data(directory):
    """Load all trajectory data."""
    all_data = []
    pattern = os.path.join(directory, "trajectories_val_batch_*.json")
    files = sorted(glob(pattern))
    
    print(f"Loading {len(files)} files...")
    for filepath in files:
        with open(filepath, 'r') as f:
            data = json.load(f)
            all_data.extend(data)
    
    return all_data

def extract_prompt_data(data):
    """Extract structured data for each prompt with 16 responses."""
    prompts_data = []
    
    for item in data:
        prompt_info = {
            'ground_truth': item['ground_truth'],
            'data_source': item.get('data_source', 'unknown'),
            'responses': []
        }
        
        for sample in item.get('samples', []):
            # Extract answer
            answer = extract_answer(sample)
            
            # Extract metrics (final turn values)
            turn_metrics = sample.get('turn_metrics', {})
            metrics = {}
            
            for metric_name in ['entropy', 'gini_impurity', 'kl_uniform', 'log_probs']:
                if metric_name in turn_metrics and 'mean' in turn_metrics[metric_name]:
                    means = turn_metrics[metric_name]['mean']
                    if means:
                        metrics[f'{metric_name}_final'] = means[-1]
                        metrics[f'{metric_name}_mean'] = np.mean(means)
                        if len(means) >= 2:
                            metrics[f'{metric_name}_change'] = means[-1] - means[0]
            
            # Check correctness
            is_correct = check_answer_correctness(answer, item['ground_truth'])
            
            response_info = {
                'answer': answer,
                'answer_normalized': normalize_answer(answer),
                'is_correct': is_correct,
                'metrics': metrics,
                'answer_reward': sample.get('answer_reward', None),
                'reward': sample['reward'],
            }
            
            prompt_info['responses'].append(response_info)
        
        if len(prompt_info['responses']) == 16:  # Only include complete sets
            prompts_data.append(prompt_info)
    
    return prompts_data

def calculate_answer_frequency(responses):
    """Calculate frequency of normalized answers."""
    answers = [r['answer_normalized'] for r in responses if r['answer_normalized'] != "NO_ANSWER"]
    return Counter(answers)

def score_response(response, answer_frequencies, strategy='entropy_freq'):
    """Score a response based on metrics and answer frequency."""
    metrics = response['metrics']
    answer_norm = response['answer_normalized']
    
    # Base metric scores (higher is better)
    entropy_score = -metrics.get('entropy_mean', 0)  # Lower entropy is better
    gini_score = -metrics.get('gini_impurity_mean', 0)  # Lower gini is better
    kl_score = metrics.get('kl_uniform_mean', 0)  # Higher KL is better
    
    # Frequency score
    answer_freq = answer_frequencies.get(answer_norm, 0)
    freq_score = answer_freq / 16.0  # Normalize by total responses
    
    if strategy == 'kl_only':
        return kl_score
    
    elif strategy == 'frequency_only':
        return freq_score
    
    elif strategy == 'entropy_only':
        return entropy_score

    elif strategy == 'gini_only':
        return gini_score
    
    elif strategy == 'reward_only':
        # Return the reward directly
        return response.get('reward', 0)
    
    elif strategy == 'reward_weighted_freq':
        # Combine reward with frequency
        reward = response.get('reward', 0)
        return reward * freq_score

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def select_best_response(prompt_data, strategy='entropy_freq', rank_power=1):
    """Select the best response for a prompt using the given strategy.
    
    Args:
        prompt_data: Data for a single prompt with responses
        strategy: Selection strategy to use
        rank_power: Power parameter for ranking-based strategies (N-r+1)^p
    """
    responses = prompt_data['responses']
    answer_frequencies = calculate_answer_frequency(responses)
    
    if strategy == 'frequency_only':
        # Simple majority voting - select most frequent answer
        if not answer_frequencies:
            return None, -1, "NO_ANSWER"
        
        best_answer = max(answer_frequencies.keys(), key=lambda x: answer_frequencies[x])
        # Find first response with this answer
        for i, response in enumerate(responses):
            if response['answer_normalized'] == best_answer:
                return response, i, response['answer']
    
    elif strategy.endswith('_rank'):
        # Ranking-based voting for confidence metrics
        # Rank all responses by their metric score
        scored_responses = []
        for i, response in enumerate(responses):
            if response['answer_normalized'] == "NO_ANSWER":
                continue
            score = score_response(response, answer_frequencies, strategy.replace('_rank', '_only'))
            scored_responses.append((score, i, response))
        
        if not scored_responses:
            return None, -1, "NO_ANSWER"
        
        # Sort by score (higher is better for all metrics after transformation)
        scored_responses.sort(key=lambda x: x[0], reverse=True)
        
        # Assign votes based on rank: rank 1 gets N votes, rank 2 gets N-1 votes, etc.
        answer_votes = defaultdict(float)
        n_responses = len(scored_responses)
        
        for rank, (score, idx, response) in enumerate(scored_responses, 1):
            answer_norm = response['answer_normalized']
            votes = (n_responses - rank + 1) ** rank_power  # (N-r+1)^p formula
            answer_votes[answer_norm] += votes
        
        # Select answer with most votes
        if not answer_votes:
            return None, -1, "NO_ANSWER"
        
        best_answer = max(answer_votes.keys(), key=lambda x: answer_votes[x])
        
        # Find best response with this answer
        for score, idx, response in scored_responses:
            if response['answer_normalized'] == best_answer:
                return response, idx, response['answer']
    
    else:
        # For metric-based strategies, aggregate scores by answer
        answer_scores = defaultdict(list)
        answer_responses = defaultdict(list)
        answer_rewards = defaultdict(list)
        
        for i, response in enumerate(responses):
            answer_norm = response['answer_normalized']
            if answer_norm == "NO_ANSWER":
                continue
            
            score = score_response(response, answer_frequencies, strategy)

            answer_scores[answer_norm].append(score)
            answer_responses[answer_norm].append((i, response))
            answer_rewards[answer_norm].append(response['reward'])
        
        # Aggregate scores for each unique answer
        best_answer = None
        best_weighted_score = float('-inf')
        
        for answer_norm, scores in answer_scores.items():
            # Get corresponding rewards for this answer
            rewards = answer_rewards[answer_norm]
            
            weighted_score = sum(scores) / 16
            
            if weighted_score > best_weighted_score:
                best_weighted_score = weighted_score
                best_answer = answer_norm
        
        # Return the best-scoring response with the selected answer
        if best_answer:
            # Among responses with the best answer, pick the one with best individual score
            best_response = None
            best_idx = -1
            best_individual_score = float('-inf')
            
            for idx, response in answer_responses[best_answer]:
                individual_score = score_response(response, answer_frequencies, strategy)
                if individual_score > best_individual_score:
                    best_individual_score = individual_score
                    best_response = response
                    best_idx = idx
            
            return best_response, best_idx, best_response['answer']
    
    return None, -1, "NO_ANSWER"

def evaluate_strategies(prompts_data, rank_power=1):
    """Evaluate different selection strategies.
    
    Args:
        prompts_data: List of prompt data
        rank_power: Power parameter for ranking-based strategies
    """
    strategies = [
        'frequency_only',
        'entropy_only', 
        'entropy_rank',
        'kl_only',
        'kl_rank',
        'gini_only',
        'gini_rank',
        'reward_only',
        'reward_weighted_freq',
    ]
    
    results = {}
    
    for strategy in strategies:
        correct = 0
        total = 0
        no_answer_count = 0
        
        for prompt_data in prompts_data:
            best_response, best_idx, selected_answer = select_best_response(prompt_data, strategy, rank_power)
            
            if best_response is None:
                no_answer_count += 1
                continue
                
            total += 1
            if best_response['answer_reward'] == 1.0:
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        results[strategy] = {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'no_answer': no_answer_count
        }
    
    return results

def analyze_answer_patterns(prompts_data):
    """Analyze answer frequency patterns."""
    freq_stats = {
        'total_prompts': len(prompts_data),
        'consensus_levels': defaultdict(int),
        'answer_distribution': [],
        'frequency_vs_correctness': []
    }
    
    for prompt_data in prompts_data:
        responses = prompt_data['responses']
        answer_frequencies = calculate_answer_frequency(responses)
        
        if not answer_frequencies:
            continue
            
        max_freq = max(answer_frequencies.values())
        freq_stats['consensus_levels'][max_freq] += 1
        
        # Check if most frequent answer is correct
        most_frequent_answer = max(answer_frequencies.keys(), key=lambda x: answer_frequencies[x])
        is_most_frequent_correct = check_answer_correctness(most_frequent_answer, prompt_data['ground_truth'])
        
        freq_stats['frequency_vs_correctness'].append({
            'max_frequency': max_freq,
            'is_correct': is_most_frequent_correct,
            'total_valid_answers': sum(answer_frequencies.values())
        })
    
    return freq_stats

def create_visualizations(results, freq_stats, prompts_data):
    """Create visualizations of the results."""
    
    # 1. Strategy comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy comparison
    ax = axes[0, 0]
    strategies = list(results.keys())
    accuracies = [results[s]['accuracy'] for s in strategies]
    
    bars = ax.bar(strategies, accuracies, color=['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'orange'])
    ax.set_ylabel('Accuracy')
    ax.set_title('Strategy Comparison')
    ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.3f}', ha='center', va='bottom')
    
    # Consensus level distribution
    ax = axes[0, 1]
    consensus_levels = sorted(freq_stats['consensus_levels'].keys())
    consensus_counts = [freq_stats['consensus_levels'][level] for level in consensus_levels]
    
    ax.bar(consensus_levels, consensus_counts, color='lightblue')
    ax.set_xlabel('Maximum Answer Frequency')
    ax.set_ylabel('Number of Prompts')
    ax.set_title('Answer Consensus Distribution')
    
    # Frequency vs Correctness
    ax = axes[1, 0]
    freq_correct_data = freq_stats['frequency_vs_correctness']
    frequencies = [d['max_frequency'] for d in freq_correct_data]
    correctness = [d['is_correct'] for d in freq_correct_data]
    
    # Calculate accuracy by frequency level
    freq_accuracy = defaultdict(list)
    for freq, correct in zip(frequencies, correctness):
        freq_accuracy[freq].append(correct)
    
    freq_levels = sorted(freq_accuracy.keys())
    accuracies_by_freq = [np.mean(freq_accuracy[freq]) for freq in freq_levels]
    counts_by_freq = [len(freq_accuracy[freq]) for freq in freq_levels]
    
    bars = ax.bar(freq_levels, accuracies_by_freq, color='lightgreen')
    ax.set_xlabel('Answer Frequency')
    ax.set_ylabel('Accuracy of Most Frequent Answer')
    ax.set_title('Answer Frequency vs Correctness')
    ax.set_ylim(0, 1)
    
    # Add count labels
    for bar, count in zip(bars, counts_by_freq):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'n={count}', ha='center', va='bottom', fontsize=8)
    
    # Sample analysis - show metric distributions for correct vs incorrect
    ax = axes[1, 1]
    
    # Collect entropy values for correct vs incorrect responses
    correct_entropy = []
    incorrect_entropy = []
    
    for prompt_data in prompts_data:
        for response in prompt_data['responses']:
            if 'entropy_final' in response['metrics']:
                entropy_val = response['metrics']['entropy_final']
                if response['is_correct']:
                    correct_entropy.append(entropy_val)
                else:
                    incorrect_entropy.append(entropy_val)
    
    ax.hist(incorrect_entropy, bins=30, alpha=0.5, label='Incorrect', color='red', density=True)
    ax.hist(correct_entropy, bins=30, alpha=0.5, label='Correct', color='green', density=True)
    ax.set_xlabel('Final Turn Entropy')
    ax.set_ylabel('Density')
    ax.set_title('Entropy Distribution: Correct vs Incorrect')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('combined_metric_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Saved: combined_metric_analysis.png")

def print_detailed_results(results, freq_stats, prompts_data, rank_power=1):
    """Print detailed analysis results."""
    print("="*80)
    print("COMBINED METRIC + FREQUENCY ANALYSIS RESULTS")
    print("="*80)
    
    print(f"\nDataset: {len(prompts_data)} prompts with 16 responses each")
    print(f"Total responses analyzed: {len(prompts_data) * 16}")
    
    print("\n" + "-"*60)
    print("STRATEGY PERFORMANCE")
    print("-"*60)
    
    # Sort strategies by accuracy
    sorted_strategies = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    for strategy, stats in sorted_strategies:
        print(f"{strategy:20s}: {stats['accuracy']:.3f} ({stats['correct']}/{stats['total']}) "
              f"[{stats['no_answer']} no-answer]")
    
    print("\n" + "-"*60)
    print("ANSWER FREQUENCY ANALYSIS")
    print("-"*60)
    
    print("Consensus level distribution:")
    for level in sorted(freq_stats['consensus_levels'].keys()):
        count = freq_stats['consensus_levels'][level]
        print(f"  {level:2d}/16 responses agree: {count:3d} prompts ({100*count/len(prompts_data):.1f}%)")
    
    # Analyze frequency vs correctness
    freq_correct_data = freq_stats['frequency_vs_correctness']
    freq_accuracy = defaultdict(list)
    for data in freq_correct_data:
        freq_accuracy[data['max_frequency']].append(data['is_correct'])
    
    print("\nMost frequent answer accuracy by consensus level:")
    for freq in sorted(freq_accuracy.keys()):
        accuracy = np.mean(freq_accuracy[freq])
        count = len(freq_accuracy[freq])
        print(f"  {freq:2d}/16 consensus: {accuracy:.3f} accuracy ({count} cases)")
    
    print("\n" + "-"*60)
    print("KEY INSIGHTS")
    print("-"*60)
    
    best_strategy = sorted_strategies[0]
    baseline_freq = results['frequency_only']['accuracy']
    baseline_entropy = results['entropy_only']['accuracy']
    baseline_gini = results['gini_only']['accuracy']

    print(f"• Best strategy: {best_strategy[0]} ({best_strategy[1]['accuracy']:.3f} accuracy)")
    print(f"• Improvement over frequency-only: {best_strategy[1]['accuracy'] - baseline_freq:.3f}")
    print(f"• Improvement over entropy-only: {best_strategy[1]['accuracy'] - baseline_entropy:.3f}")
    print(f"• Improvement over gini-only: {best_strategy[1]['accuracy'] - baseline_gini:.3f}")

    # Consensus analysis
    high_consensus = sum(count for freq, count in freq_stats['consensus_levels'].items() if freq >= 8)
    low_consensus = sum(count for freq, count in freq_stats['consensus_levels'].items() if freq <= 4)
    
    print(f"• High consensus (≥8/16): {high_consensus} prompts ({100*high_consensus/len(prompts_data):.1f}%)")
    print(f"• Low consensus (≤4/16): {low_consensus} prompts ({100*low_consensus/len(prompts_data):.1f}%)")
    
    # Calculate accuracy for high vs low consensus cases using best strategy
    high_consensus_accuracy = 0
    low_consensus_accuracy = 0
    high_count = 0
    low_count = 0
    
    for prompt_data in prompts_data:
        answer_frequencies = calculate_answer_frequency(prompt_data['responses'])
        if not answer_frequencies:
            continue
            
        max_freq = max(answer_frequencies.values())
        best_response, _, _ = select_best_response(prompt_data, best_strategy[0], rank_power)
        
        if best_response:
            if max_freq >= 8:
                high_count += 1
                if best_response['is_correct']:
                    high_consensus_accuracy += 1
            elif max_freq <= 4:
                low_count += 1
                if best_response['is_correct']:
                    low_consensus_accuracy += 1
    
    if high_count > 0:
        print(f"• Best strategy accuracy on high consensus: {high_consensus_accuracy/high_count:.3f}")
    if low_count > 0:
        print(f"• Best strategy accuracy on low consensus: {low_consensus_accuracy/low_count:.3f}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Evaluate answer selection strategies with configurable ranking power')
    parser.add_argument('--rank-power', type=float, default=1.0,
                        help='Power parameter p for ranking formula (N-r+1)^p (default: 1.0)')
    parser.add_argument('--directory', type=str, 
                        default='/home/mhong/zhan9359/work/quan/Search-R1/outputs/log_val_traj/nq-search-r1-quan-7b-ckpt1-sampled-512-BoN16_steprewards_analysis_20250908_101826',
                        help='Directory containing trajectory files')
    args = parser.parse_args()
    
    # Load data
    print(f"Loading and processing data from {args.directory}...")
    data = load_all_data(args.directory)
    prompts_data = extract_prompt_data(data)
    
    print(f"Processed {len(prompts_data)} complete prompts (16 responses each)")
    print(f"Using rank power: {args.rank_power}")
    
    # Analyze answer patterns
    print("Analyzing answer frequency patterns...")
    freq_stats = analyze_answer_patterns(prompts_data)
    
    # Evaluate strategies
    print("Evaluating selection strategies...")
    results = evaluate_strategies(prompts_data, rank_power=args.rank_power)
    
    # Create visualizations
    create_visualizations(results, freq_stats, prompts_data)
    
    # Print detailed results
    print_detailed_results(results, freq_stats, prompts_data, rank_power=args.rank_power)

if __name__ == "__main__":
    main()