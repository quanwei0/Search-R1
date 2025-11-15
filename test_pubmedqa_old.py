#!/usr/bin/env python3
"""
Test AlphaMed model on PubMedQA pqa_labeled dataset with batch inference
"""

import json
import re
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from tqdm import tqdm

# 设置只使用前4张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def extract_answer_from_output(output_text):
    """Extract answer from model output"""
    # Look for \\boxed{} format first
    boxed_pattern = r'\\boxed\{([AB])\}'
    boxed_match = re.search(boxed_pattern, output_text)
    if boxed_match:
        return boxed_match.group(1)
    
    # Look for <answer> tags
    answer_pattern = r'<answer>([AB])</answer>'
    answer_match = re.search(answer_pattern, output_text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    
    # Look for other common answer patterns
    patterns = [
        r'[Aa]nswer[:\s]*([AB])',
        r'[Tt]he answer is[:\s]*([AB])',
        r'[Cc]orrect answer[:\s]*([AB])',
        r'[Ff]inal answer[:\s]*([AB])',
        r'[Tt]herefore[,\s]*([AB])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output_text)
        if match:
            return match.group(1).upper()
    
    # Look for yes/no patterns and convert to A/B
    yes_no_patterns = [
        r'[Aa]nswer[:\s]*(yes|no)',
        r'[Tt]he answer is[:\s]*(yes|no)',
        r'[Ff]inal answer[:\s]*(yes|no)',
    ]
    
    for pattern in yes_no_patterns:
        match = re.search(pattern, output_text, re.IGNORECASE)
        if match:
            answer = match.group(1).lower()
            return 'A' if answer == 'yes' else 'B'
    
    # Look for last occurrence of A-B
    last_letters = re.findall(r'\b([AB])\b', output_text)
    if last_letters:
        return last_letters[-1]
    
    return None

def build_prompt(example):
    """Build prompt for PubMedQA question"""
    question = example["question"]
    
    prompt = "You are a medical AI assistant.\n"
    prompt += f"Question: {question}\n"
    prompt += "Options:\n"
    prompt += "A. yes\n"
    prompt += "B. no\n"
    prompt += "Please reason step by step, and put the final answer in \\boxed{}.\n"
    
    return prompt

def main():
    # 1. Load model
    model_id = "che111/AlphaMed-8B-instruct-rl"
    
    print(f"🚀 Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Set eval mode
    model.eval()
    
    print("✅ Model loaded successfully and set to eval mode!")
    
    # 2. Create text-generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
    )
    
    # 3. Load PubMedQA dataset from jsonl file
    print("📁 Loading PubMedQA test dataset...")
    test_file = "/home/li003968/Search-R1/data/pubmedqa_test.jsonl"
    
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line.strip()))
    
    test_samples = len(test_data)
    
    print(f"📊 Testing on {test_samples} samples from PubMedQA")
    print(f"   - Total samples: {test_samples}")
    
    # 4. Prepare all prompts
    print("📝 Preparing prompts...")
    prompts = [build_prompt(ex) for ex in test_data]
    
    # 5. Batch inference
    batch_size = 8  # Adjust based on GPU memory
    max_new_tokens = 2048
    
    print(f"🚀 Starting batch inference (batch_size={batch_size}, max_new_tokens={max_new_tokens})...")
    
    all_generations = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Batch Inference"):
        batch_prompts = prompts[i : i + batch_size]
        
        # Pipeline supports list input
        with torch.no_grad():  # Ensure no gradient computation
            outputs = pipe(
                batch_prompts,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Extract generated text
        for out in outputs:
            all_generations.append(out[0]["generated_text"])
    
    # 6. Evaluate results
    print("📊 Evaluating results...")
    results = []
    correct_count = 0
    
    for i, (item, full_output) in enumerate(zip(test_data, all_generations)):
        # Extract generated part (remove prompt)
        prompt = prompts[i]
        if full_output.startswith(prompt):
            generated_text = full_output[len(prompt):]
        else:
            generated_text = full_output
        
        # Get correct answer
        correct_answer = item['answer_idx']  # 'A' or 'B'
        
        # Extract predicted answer
        predicted_answer = extract_answer_from_output(generated_text)
        
        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct_count += 1
        
        # Store result
        result = {
            'id': i,
            'pubid': item.get('pubid', f'pubmed_{i}'),
            'question': item['question'][:100] + "...",
            'correct_answer': correct_answer,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct,
            'full_output': generated_text[:300] + "..." if len(generated_text) > 300 else generated_text
        }
        results.append(result)
    
    # Calculate final accuracy
    accuracy = correct_count / test_samples
    
    # Calculate accuracy by answer type
    answer_distribution = {'A': {'correct': 0, 'total': 0}, 'B': {'correct': 0, 'total': 0}}
    
    for result in results:
        answer = result['correct_answer']
        if answer in answer_distribution:
            answer_distribution[answer]['total'] += 1
            if result['is_correct']:
                answer_distribution[answer]['correct'] += 1
    
    # Print results
    print(f"\n🎯 PubMedQA Test Results ({test_samples} samples):")
    print(f"Overall Accuracy: {accuracy:.3f} ({correct_count}/{test_samples})")
    
    print(f"\n📊 Accuracy by Answer Type:")
    for answer, stats in sorted(answer_distribution.items()):
        if stats['total'] > 0:
            answer_acc = stats['correct'] / stats['total']
            answer_text = 'yes' if answer == 'A' else 'no'
            print(f"  {answer} ({answer_text}): {answer_acc:.3f} ({stats['correct']}/{stats['total']})")
    
    # Show some examples
    print(f"\n📝 Sample Results:")
    for i in range(min(5, len(results))):
        result = results[i]
        status = "✅" if result['is_correct'] else "❌"
        print(f"  {status} Sample {i+1}: {result['question']}")
        print(f"     PubID: {result['pubid']}")
        print(f"     Correct: {result['correct_answer']}, Predicted: {result['predicted_answer']}")
        print()
    
    # Save results
    output_file = "/home/li003968/Search-R1/pubmedqa_alphamed_8b_results.json"
    output_data = {
        'model_id': model_id,
        'dataset': 'PubMedQA pqa_labeled',
        'test_samples': test_samples,
        'correct_count': correct_count,
        'accuracy': accuracy,
        'batch_size': batch_size,
        'max_new_tokens': max_new_tokens,
        'answer_distribution': {k: v['correct']/v['total'] if v['total'] > 0 else 0 for k, v in answer_distribution.items()},
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {output_file}")
    print(f"🎉 PubMedQA testing completed! Final accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    main()
