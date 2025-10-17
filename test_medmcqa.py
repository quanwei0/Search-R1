#!/usr/bin/env python3
"""
Test AlphaMed-8B-instruct-rl model on openlifescienceai/medmcqa test split
"""

import json
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from tqdm import tqdm
import torch

# 设置只使用前4张GPU卡
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def extract_answer_from_output(output_text):
    """Extract answer from model output"""
    # Look for \\boxed{} format first
    boxed_pattern = r'\\boxed\{([A-D])\}'
    boxed_match = re.search(boxed_pattern, output_text)
    if boxed_match:
        return boxed_match.group(1)
    
    # Look for other common answer patterns
    patterns = [
        r'[Aa]nswer[:\s]*([A-D])',
        r'[Tt]he answer is[:\s]*([A-D])',
        r'[Cc]orrect answer[:\s]*([A-D])',
        r'[Ff]inal answer[:\s]*([A-D])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output_text)
        if match:
            return match.group(1).upper()
    
    # Look for last occurrence of A-D
    last_letters = re.findall(r'\b([A-D])\b', output_text)
    if last_letters:
        return last_letters[-1]
    
    return None

def format_medmcqa_question(item):
    """Format MedMCQA question for AlphaMed model"""
    question = item['question']
    opa = item['opa']  # Option A
    opb = item['opb']  # Option B
    opc = item['opc']  # Option C
    opd = item['opd']  # Option D
    
    prompt = f"""Question: {question}

Options:
A: {opa}
B: {opb}
C: {opc}
D: {opd}

Please reason step by step, and put the final answer in \\boxed{{}}"""
    
    return prompt

def convert_answer_idx_to_letter(answer_idx):
    """Convert answer index (0,1,2,3) to letter (A,B,C,D)"""
    idx_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    return idx_to_letter.get(answer_idx, None)

def main():
    # Load model and tokenizer
    model_id = "che111/AlphaMed-8B-instruct-rl"
    
    print(f"🚀 Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer,
        torch_dtype=torch.float16
    )
    
    print("✅ Model loaded successfully!")
    
    # Load MedMCQA test dataset
    print("📁 Loading MedMCQA test dataset...")
    dataset = load_dataset("openlifescienceai/medmcqa", split="test")
    
    # Take first 20 samples for testing
    test_samples = 20
    test_data = dataset.select(range(test_samples))
    
    print(f"📊 Testing on {test_samples} samples from MedMCQA test split")
    print(f"📋 Dataset info:")
    print(f"   - Total test samples available: {len(dataset)}")
    print(f"   - Testing on first: {test_samples}")
    print(f"   - Columns: {test_data.column_names}")
    
    results = []
    correct_count = 0
    
    for i, item in enumerate(tqdm(test_data, desc="Testing MedMCQA")):
        try:
            # Format prompt
            prompt = format_medmcqa_question(item)
            
            # Get correct answer
            correct_answer_idx = item['cop']  # Correct option index (0,1,2,3)
            correct_answer_letter = convert_answer_idx_to_letter(correct_answer_idx)
            
            # Generate output
            max_new_tokens = 8196
            output = pipe(
                prompt, 
                max_new_tokens=max_new_tokens, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )[0]["generated_text"]
            
            # Extract generated part (remove prompt)
            generated_text = output[len(prompt):]
            
            # Extract answer
            predicted_answer = extract_answer_from_output(generated_text)
            
            is_correct = predicted_answer == correct_answer_letter
            if is_correct:
                correct_count += 1
            
            # Store result
            result = {
                'id': i,
                'question': item['question'][:100] + "...",
                'subject_name': item.get('subject_name', 'unknown'),
                'choice_type': item.get('choice_type', 'unknown'),
                'correct_answer': correct_answer_letter,
                'predicted_answer': predicted_answer,
                'is_correct': is_correct,
                'full_output': generated_text[:300] + "..." if len(generated_text) > 300 else generated_text
            }
            results.append(result)
            
            # Print progress every 20 samples
            if (i + 1) % 20 == 0:
                current_accuracy = correct_count / (i + 1)
                print(f"Progress: {i+1}/{test_samples}, Accuracy: {current_accuracy:.3f}")
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            result = {
                'id': i,
                'question': item['question'][:100] + "...",
                'subject_name': item.get('subject_name', 'unknown'),
                'choice_type': item.get('choice_type', 'unknown'),
                'correct_answer': convert_answer_idx_to_letter(item['cop']),
                'predicted_answer': None,
                'is_correct': False,
                'full_output': f"Error: {str(e)}"
            }
            results.append(result)
    
    # Calculate final accuracy
    accuracy = correct_count / test_samples
    
    # Calculate accuracy by subject
    subject_accuracy = {}
    choice_type_accuracy = {}
    
    for result in results:
        # By subject
        subject = result['subject_name']
        if subject not in subject_accuracy:
            subject_accuracy[subject] = {'correct': 0, 'total': 0}
        subject_accuracy[subject]['total'] += 1
        if result['is_correct']:
            subject_accuracy[subject]['correct'] += 1
            
        # By choice type
        choice_type = result['choice_type']
        if choice_type not in choice_type_accuracy:
            choice_type_accuracy[choice_type] = {'correct': 0, 'total': 0}
        choice_type_accuracy[choice_type]['total'] += 1
        if result['is_correct']:
            choice_type_accuracy[choice_type]['correct'] += 1
    
    # Print results
    print(f"\n🎯 MedMCQA Test Results ({test_samples} samples):")
    print(f"Overall Accuracy: {accuracy:.3f} ({correct_count}/{test_samples})")
    
    print(f"\n📚 Accuracy by Subject:")
    for subject, stats in sorted(subject_accuracy.items()):
        if stats['total'] > 0:
            subject_acc = stats['correct'] / stats['total']
            print(f"  {subject}: {subject_acc:.3f} ({stats['correct']}/{stats['total']})")
    
    print(f"\n🔢 Accuracy by Choice Type:")
    for choice_type, stats in sorted(choice_type_accuracy.items()):
        if stats['total'] > 0:
            choice_acc = stats['correct'] / stats['total']
            print(f"  {choice_type}: {choice_acc:.3f} ({stats['correct']}/{stats['total']})")
    
    # Show some examples
    print(f"\n📝 Sample Results:")
    for i in range(min(5, len(results))):
        result = results[i]
        status = "✅" if result['is_correct'] else "❌"
        print(f"  {status} Sample {i+1}: {result['question']}")
        print(f"     Subject: {result['subject_name']}")
        print(f"     Correct: {result['correct_answer']}, Predicted: {result['predicted_answer']}")
        print()
    
    # Save results
    output_file = "/home/li003968/Search-R1/medmcqa_alphamed_8b_results.json"
    output_data = {
        'model_id': model_id,
        'dataset': 'openlifescienceai/medmcqa',
        'split': 'test',
        'test_samples': test_samples,
        'correct_count': correct_count,
        'accuracy': accuracy,
        'subject_accuracy': {k: v['correct']/v['total'] for k, v in subject_accuracy.items() if v['total'] > 0},
        'choice_type_accuracy': {k: v['correct']/v['total'] for k, v in choice_type_accuracy.items() if v['total'] > 0},
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Results saved to: {output_file}")
    print(f"🎉 MedMCQA testing completed! Final accuracy: {accuracy:.3f}")

if __name__ == "__main__":
    main()
