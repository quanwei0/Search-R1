#!/usr/bin/env python3
"""
Test AlphaMed-3B-instruct-rl model on first 50 samples using the specified template
"""

import json
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm
import torch

# 设置只使用前4张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def extract_answer_from_output(output_text):
    """Extract answer from model output"""
    # Look for \\boxed{} format first
    boxed_pattern = r'\\boxed\{([A-E])\}'
    boxed_match = re.search(boxed_pattern, output_text)
    if boxed_match:
        return boxed_match.group(1)
    
    # Look for other common answer patterns
    patterns = [
        r'[Aa]nswer[:\s]*([A-E])',
        r'[Tt]he answer is[:\s]*([A-E])',
        r'[Cc]orrect answer[:\s]*([A-E])',
        r'[Ff]inal answer[:\s]*([A-E])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output_text)
        if match:
            return match.group(1).upper()
    
    # Look for last occurrence of A-E
    last_letters = re.findall(r'\b([A-E])\b', output_text)
    if last_letters:
        return last_letters[-1]
    
    return None

def main():
    # Load model and tokenizer
    model_id = "che111/AlphaMed-8B-instruct-rl"  # Using 8B model as specified
    # model_id = "meta-llama/Meta-Llama-3-8B"
    print(f"🚀 Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"  # 自动分配到多张GPU
    )
    
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto"  # 确保pipeline也使用多GPU
    )
    
    print("✅ Model loaded successfully!")
    
    # 检查GPU使用情况
    print(f"🔍 GPU Usage Info:")
    print(f"   Available GPUs: {torch.cuda.device_count()}")
    print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    # 显示模型分布情况
    if hasattr(model, 'hf_device_map'):
        print(f"   Model device map: {model.hf_device_map}")
    else:
        print(f"   Model device: {next(model.parameters()).device}")
    
    # Load test data
    test_file = "/home/li003968/Search-R1/baseline_test/test.jsonl"
    print(f"📁 Loading test data from: {test_file}")
    
    test_data = []
    with open(test_file, 'r') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    
    # Only test first 100 samples
    test_data = test_data[:200]
    print(f"📊 Testing on first {len(test_data)} samples")
    
    results = []
    correct_count = 0
    
    for i, item in enumerate(tqdm(test_data, desc="Testing")):
        try:
            question = item['question']
            options = item['options']
            correct_answer = item['answer_idx']
            
            # Build options text
            options_text = ""
            for letter in sorted(options.keys()):
                options_text += f"{letter}: {options[letter]}\n"
            
            # Format question using the specified template
            prompt = f"""Question: {question}

Options:
{options_text.strip()}

Please reason step by step, and put the final answer in \\boxed{{}}"""
            
            # Generate output
            max_new_tokens = 8196  # Reduced from 8196 for faster testing
            output = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"]
            
            # Extract generated part (remove prompt)
            generated_text = output[len(prompt):]
            
            # Extract answer
            predicted_answer = extract_answer_from_output(generated_text)
            
            is_correct = predicted_answer == correct_answer
            if is_correct:
                correct_count += 1
            
            # Store result
            result = {
                'id': i,
                'question': question[:100] + "...",
                'correct_answer': correct_answer,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'meta_info': item.get('meta_info', 'unknown'),
                'full_output': generated_text[:500] + "..." if len(generated_text) > 500 else generated_text
            }
            results.append(result)
            
            # Print progress every 10 samples
            if (i + 1) % 10 == 0:
                current_accuracy = correct_count / (i + 1)
                print(f"Progress: {i+1}/100, Accuracy: {current_accuracy:.3f}")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            result = {
                'id': i,
                'question': item['question'][:100] + "...",
                'correct_answer': item['answer_idx'],
                'predicted_answer': None,
                'is_correct': False,
                'meta_info': item.get('meta_info', 'unknown'),
                'full_output': f"Error: {str(e)}"
            }
            results.append(result)
    
    # Calculate final accuracy
    accuracy = correct_count / len(test_data)
    
    # Calculate accuracy by meta_info
    meta_accuracy = {}
    for result in results:
        meta = result['meta_info']
        if meta not in meta_accuracy:
            meta_accuracy[meta] = {'correct': 0, 'total': 0}
        meta_accuracy[meta]['total'] += 1
        if result['is_correct']:
            meta_accuracy[meta]['correct'] += 1
    
    # Print results
    print(f"\n🎯 Final Results (100 samples):")
    print(f"Overall Accuracy: {accuracy:.3f} ({correct_count}/100)")
    
    print(f"\n📊 Accuracy by Category:")
    for meta, stats in meta_accuracy.items():
        meta_acc = stats['correct'] / stats['total']
        print(f"  {meta}: {meta_acc:.3f} ({stats['correct']}/{stats['total']})")
    
    # Show some examples
    print(f"\n📝 Sample Results:")
    for i in range(min(5, len(results))):
        result = results[i]
        status = "✅" if result['is_correct'] else "❌"
        print(f"  {status} Sample {i+1}: {result['question']}")
        print(f"     Correct: {result['correct_answer']}, Predicted: {result['predicted_answer']}")
    
    # Save results
    output_file = "/home/li003968/Search-R1/alphamed_8b_200samples_results.json"
    output_data = {
        'model_id': model_id,
        'test_samples': 100,
        'correct_count': correct_count,
        'accuracy': accuracy,
        'meta_accuracy': {k: v['correct']/v['total'] for k, v in meta_accuracy.items()},
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"🎉 Testing completed! Final accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    main()
