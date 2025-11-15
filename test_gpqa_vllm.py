#!/usr/bin/env python3
"""
Test AlphaMed model on GPQA-M (Medical) test split with batch inference using vLLM (0.6.3)
"""

import json
import re
import os
import tempfile

# 必须在导入vllm之前设置环境变量
# 使用数据盘的tmp目录
tmp_dir = "/mnt/data1/li003968/tmp"
os.makedirs(tmp_dir, exist_ok=True)

# 设置多个环境变量确保生效
os.environ["TMPDIR"] = tmp_dir
os.environ["TEMP"] = tmp_dir
os.environ["TMP"] = tmp_dir
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# 修改tempfile的默认目录
tempfile.tempdir = tmp_dir

from datasets import load_dataset
from vllm import LLM, SamplingParams  # vLLM 0.6.3

# -----------------------------
# 工具函数
# -----------------------------
def extract_answer_from_output(output_text, valid_options=None):
    """Extract answer from model output
    
    Args:
        output_text: The generated text from model
        valid_options: List of valid option letters (e.g., ['A', 'B', 'C', 'D'])
                      If None, defaults to A-D
    """
    if valid_options is None:
        valid_options = ['A', 'B', 'C', 'D']
    
    # Create regex pattern based on valid options
    options_pattern = '|'.join(valid_options)
    
    # Look for \\boxed{} format first
    boxed_pattern = rf'\\boxed\{{({options_pattern})\}}'
    boxed_match = re.search(boxed_pattern, output_text)
    if boxed_match:
        return boxed_match.group(1)

    # Look for <answer> tags
    answer_pattern = rf'<answer>({options_pattern})</answer>'
    answer_match = re.search(answer_pattern, output_text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).upper()

    # Look for other common answer patterns
    patterns = [
        rf'[Aa]nswer[:\s]*({options_pattern})',
        rf'[Tt]he answer is[:\s]*({options_pattern})',
        rf'[Cc]orrect answer[:\s]*({options_pattern})',
        rf'[Ff]inal answer[:\s]*({options_pattern})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output_text)
        if match:
            return match.group(1).upper()

    # Look for last occurrence of valid options
    last_letters = re.findall(rf'\b({options_pattern})\b', output_text)
    if last_letters:
        return last_letters[-1]

    return None


def build_prompt(example):
    """Build prompt for GPQA question"""
    question = example["Question"]
    
    # GPQA has options: Correct Answer, Incorrect Answer 1, Incorrect Answer 2, Incorrect Answer 3
    correct_answer = example["Correct Answer"]
    incorrect_answers = [
        example["Incorrect Answer 1"],
        example["Incorrect Answer 2"],
        example["Incorrect Answer 3"]
    ]
    
    # Put correct answer first as option A
    options = {
        'A': correct_answer,
        'B': incorrect_answers[0],
        'C': incorrect_answers[1],
        'D': incorrect_answers[2]
    }
    
    prompt = "You are a medical AI assistant.\n"
    prompt += f"Question: {question}\n"
    prompt += "Options:\n"
    for letter in ['A', 'B', 'C', 'D']:
        prompt += f"{letter}. {options[letter]}\n"
    prompt += "Please reason step by step, and put the final answer in \\boxed{}.\n"
    
    return prompt, 'A'  # Correct answer is always A since we put it first


def main():
    # model_id = "che111/AlphaMed-8B-instruct-rl"
    # model_id = "meta-llama/Llama-3.1-8B-Instruct"
    model_id = "johnsnowlabs/JSL-MedLlama-3-8B-v1.0"
    print(f"🚀 Loading model with vLLM 0.6.3: {model_id}")
    
    # 使用4张GPU
    llm = LLM(
        model=model_id,
        dtype="float16",
        tensor_parallel_size=4,
    )

    # vLLM 的采样参数
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=2048,
    )

    print("📁 Loading GPQA dataset...")
    # Load full GPQA dataset
    try:
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_main")
        full_data = dataset["train"]
    except:
        print("⚠️  Trying alternative dataset name...")
        dataset = load_dataset("google/gpqa", "gpqa_main")
        full_data = dataset["train"]
    
    print(f"📊 Total GPQA samples: {len(full_data)}")
    
    # Filter for medical-related subdomains (Molecular Biology + Genetics)
    print("🔍 Filtering for medical subdomains (Molecular Biology + Genetics)...")
    test_data = [item for item in full_data if item.get("Subdomain") in ["Molecular Biology", "Genetics"]]
    
    test_samples = len(test_data)
    print(f"✅ Filtered to {test_samples} medical samples")
    
    # Show subdomain distribution
    from collections import Counter
    subdomain_dist = Counter([item["Subdomain"] for item in test_data])
    print(f"\n📈 Subdomain distribution:")
    for subdomain, count in subdomain_dist.items():
        print(f"  - {subdomain}: {count} samples")

    # 准备 prompts 和正确答案
    print("\n📝 Preparing prompts...")
    prompts = []
    correct_answers = []
    for ex in test_data:
        prompt, correct_answer = build_prompt(ex)
        prompts.append(prompt)
        correct_answers.append(correct_answer)

    # ---------- vLLM 一次性生成 ----------
    print("🚀 Generating with vLLM ...")
    vllm_outputs = llm.generate(prompts, sampling_params)

    # ---------- 评估 ----------
    print("📊 Evaluating results ...")

    results = []
    correct_count = 0
    
    # GPQA可能有subdomain信息
    subdomain_accuracy = {}

    for i, (item, out) in enumerate(zip(test_data, vllm_outputs)):
        # vLLM 0.6.3 的每个结果
        generated_text = out.outputs[0].text

        # 正确答案 (always 'A' since we put correct answer first)
        correct_answer = correct_answers[i]

        # 模型预测
        predicted_answer = extract_answer_from_output(generated_text, ['A', 'B', 'C', 'D'])
        is_correct = (predicted_answer == correct_answer)
        if is_correct:
            correct_count += 1

        # 按 subdomain 统计
        subdomain = item.get("Subdomain", "unknown")
        if subdomain not in subdomain_accuracy:
            subdomain_accuracy[subdomain] = {"correct": 0, "total": 0}
        subdomain_accuracy[subdomain]["total"] += 1
        if is_correct:
            subdomain_accuracy[subdomain]["correct"] += 1

        results.append({
            "id": i,
            "question": item["Question"][:100] + "...",
            "subdomain": subdomain,
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "full_output": generated_text[:300] + "..." if len(generated_text) > 300 else generated_text,
        })

    accuracy = correct_count / test_samples

    print(f"\n🎯 GPQA Test Results ({test_samples} samples):")
    print(f"Overall Accuracy: {accuracy:.3f} ({correct_count}/{test_samples})")

    if len(subdomain_accuracy) > 1:  # Only show if there are multiple subdomains
        print("\n📚 Accuracy by Subdomain:")
        for subdomain, stats in sorted(subdomain_accuracy.items()):
            if stats["total"] > 0:
                acc = stats["correct"] / stats["total"]
                print(f"  {subdomain}: {acc:.3f} ({stats['correct']}/{stats['total']})")

    # Show some examples
    print(f"\n📝 Sample Results (first 5):")
    for i in range(min(5, len(results))):
        result = results[i]
        status = "✅" if result['is_correct'] else "❌"
        print(f"  {status} Sample {i+1}: {result['question']}")
        print(f"     Subdomain: {result['subdomain']}")
        print(f"     Correct: {result['correct_answer']}, Predicted: {result['predicted_answer']}")
        print()

    # 保存结果
    output_file = "./gpqa_medical_vllm_0.6.3_results.json"
    output_data = {
        "model_id": model_id,
        "dataset": "GPQA Medical (Molecular Biology + Genetics)",
        "test_samples": test_samples,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "max_tokens": sampling_params.max_tokens,
        "subdomain_accuracy": {
            k: v["correct"] / v["total"] if v["total"] > 0 else 0 
            for k, v in subdomain_accuracy.items()
        },
        "results": results,
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {output_file}")
    print(f"🎉 Done. Final accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()

