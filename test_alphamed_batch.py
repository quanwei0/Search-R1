#!/usr/bin/env python3
"""
Test AlphaMed-8B-instruct-rl model on all test samples using batch inference
"""

import json
import re
import os
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import torch

def extract_answer_from_output(output_text):
    """Extract answer from model output"""
    # Look for \\boxed{} format first
    boxed_pattern = r'\\boxed\{([A-E])\}'
    boxed_match = re.search(boxed_pattern, output_text)
    if boxed_match:
        return boxed_match.group(1)
    
    # Look for <answer> tags
    answer_pattern = r'<answer>([A-E])</answer>'
    answer_match = re.search(answer_pattern, output_text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()
    
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

def create_prompt(question, options):
    """Create prompt for a question"""
    options_text = ""
    for letter in sorted(options.keys()):
        options_text += f"{letter}: {options[letter]}\n"
    
    prompt = f"""Question: {question}

Options:
{options_text.strip()}

Please reason step by step, and put the final answer in \\boxed{{}}"""
    
    return prompt

def batch_generate(model, tokenizer, prompts, batch_size=4, max_new_tokens=2048):
    """Generate outputs for a batch of prompts"""
    all_outputs = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        # Generate - remove temperature and top_p when do_sample=False
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                num_beams=1,
                temperature=None,  # Explicitly set to None
                top_p=None  # Explicitly set to None
            )
        
        # Decode outputs
        for j, output in enumerate(outputs):
            # Get the generated part only (remove input prompt)
            input_length = inputs['input_ids'][j].shape[0]
            generated_ids = output[input_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            all_outputs.append(generated_text)
    
    return all_outputs

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test models on medical QA dataset with batch inference')
    parser.add_argument('--model_id', type=str, default='che111/AlphaMed-8B-instruct-rl',
                        help='Model ID from HuggingFace (e.g., meta-llama/Meta-Llama-3-8B-Instruct)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference (default: 32)')
    parser.add_argument('--max_new_tokens', type=int, default=2048,
                        help='Maximum number of new tokens to generate (default: 2048)')
    parser.add_argument('--test_file', type=str, default='/home/li003968/Search-R1/baseline_test/test.jsonl',
                        help='Path to test data file')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Output JSON file path (auto-generated if not specified)')
    parser.add_argument('--cuda_devices', type=str, default='0,1,2,3',
                        help='CUDA visible devices (default: 0,1,2,3)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to test (default: all)')
    
    args = parser.parse_args()
    
    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    
    # Auto-generate output filename if not specified
    if args.output_file is None:
        model_name = args.model_id.replace('/', '_').replace('-', '_')
        args.output_file = f"/home/li003968/Search-R1/results_{model_name}_{args.max_new_tokens}tokens.json"
    
    print(f"🚀 Configuration:")
    print(f"   Model: {args.model_id}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Max new tokens: {args.max_new_tokens}")
    print(f"   Test file: {args.test_file}")
    print(f"   Output file: {args.output_file}")
    print(f"   CUDA devices: {args.cuda_devices}")
    print()
    
    # Load model and tokenizer
    print(f"🚀 Loading model: {args.model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # Set padding side to left for decoder-only models
    tokenizer.padding_side = 'left'
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print("✅ Model loaded successfully!")
    
    # 检查GPU使用情况
    print(f"🔍 GPU Usage Info:")
    print(f"   Available GPUs: {torch.cuda.device_count()}")
    print(f"   CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    # 显示模型分布情况
    if hasattr(model, 'hf_device_map'):
        print(f"   Model device map: {model.hf_device_map}")
    
    # Load test data
    print(f"📁 Loading test data from: {args.test_file}")
    
    test_data = []
    with open(args.test_file, 'r') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line.strip()))
    
    # Limit number of samples if specified
    if args.num_samples is not None:
        test_data = test_data[:args.num_samples]
        print(f"📊 Testing on first {len(test_data)} samples")
    else:
        print(f"📊 Testing on all {len(test_data)} samples")
    
    # Prepare all prompts
    print("📝 Preparing prompts...")
    prompts = []
    for item in test_data:
        prompt = create_prompt(item['question'], item['options'])
        prompts.append(prompt)
    
    # Batch inference
    batch_size = args.batch_size
    max_new_tokens = args.max_new_tokens
    
    print(f"🚀 Starting batch inference (batch_size={batch_size})...")
    generated_outputs = []
    
    # Process in batches with progress bar
    for i in tqdm(range(0, len(prompts), batch_size), desc="Batch Inference"):
        batch_prompts = prompts[i:i + batch_size]
        batch_outputs = batch_generate(
            model, 
            tokenizer, 
            batch_prompts, 
            batch_size=len(batch_prompts),
            max_new_tokens=max_new_tokens
        )
        generated_outputs.extend(batch_outputs)
    
    # Process results
    print("📊 Processing results...")
    results = []
    correct_count = 0
    
    for i, (item, generated_text) in enumerate(zip(test_data, generated_outputs)):
        correct_answer = item['answer_idx']
        predicted_answer = extract_answer_from_output(generated_text)
        
        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct_count += 1
        
        result = {
            'id': i,
            'question': item['question'][:100] + "...",
            'correct_answer': correct_answer,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct,
            'meta_info': item.get('meta_info', 'unknown'),
            'full_output': generated_text[:500] + "..." if len(generated_text) > 500 else generated_text
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
    print(f"\n🎯 Final Results ({len(test_data)} samples):")
    print(f"Overall Accuracy: {accuracy:.3f} ({correct_count}/{len(test_data)})")
    
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
    output_file = args.output_file
    output_data = {
        'model_id': args.model_id,
        'test_samples': len(test_data),
        'correct_count': correct_count,
        'accuracy': accuracy,
        'batch_size': batch_size,
        'max_new_tokens': max_new_tokens,
        'meta_accuracy': {k: v['correct']/v['total'] for k, v in meta_accuracy.items()},
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {output_file}")
    print(f"🎉 Testing completed! Final accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    main()

