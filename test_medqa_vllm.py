#!/usr/bin/env python3
"""
Test AlphaMed model on MedQA test split with batch inference using vLLM (0.6.3)
"""

import json
import re
import os
import tempfile

# 只使用6、7两张卡
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

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

from vllm import LLM, SamplingParams  # vLLM 0.6.3

# -----------------------------
# 工具函数
# -----------------------------
def extract_answer_from_output(output_text, valid_options=None):
    """Extract answer from model output"""
    if valid_options is None:
        valid_options = ['A', 'B', 'C', 'D', 'E']
    
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


def build_prompt_cot(example):
    """Build prompt for MedQA question with Chain-of-Thought"""
    question = example["question"]
    options = example["options"]

    prompt = "You are a medical AI assistant.\n"
    prompt += f"Question: {question}\n"
    prompt += "Options:\n"
    for letter in sorted(options.keys()):
        prompt += f"{letter}. {options[letter]}\n"
    prompt += "Please reason step by step, and put the final answer in \\boxed{}.\n"
    
    return prompt


def build_prompt_direct(example):
    """Build prompt for MedQA question - Direct inference without CoT"""
    question = example["question"]
    options = example["options"]

    prompt = "You are a medical AI assistant.\n"
    prompt += f"Question: {question}\n"
    prompt += "Options:\n"
    for letter in sorted(options.keys()):
        prompt += f"{letter}. {options[letter]}\n"
    prompt += "Answer: \\boxed{"
    
    return prompt


# 默认使用直接推理版本
def build_prompt(example):
    """Build prompt - choose between CoT or Direct"""
    # 切换这里来选择不同的prompt策略
    # return build_prompt_cot(example)  # Chain-of-Thought
    return build_prompt_direct(example)  # Direct inference


def main():
    # model_id = "che111/AlphaMed-8B-instruct-rl"
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"🚀 Loading model with vLLM 0.6.3: {model_id}")
    print(f"🎮 Using GPUs: 6, 7 (tensor_parallel_size=2)")
    
    # 使用2张GPU (6, 7)
    llm = LLM(
        model=model_id,
        dtype="float16",
        tensor_parallel_size=2,
    )

    # vLLM 的采样参数
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=2048,
    )

    print("📁 Loading MedQA test dataset...")
    test_file = "/home/li003968/Search-R1/baseline_test/test.jsonl"
    
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line.strip()))
    
    test_samples = len(test_data)
    print(f"📊 Testing on {test_samples} samples from MedQA")

    # 准备 prompts
    print("📝 Preparing prompts...")
    prompts = [build_prompt(ex) for ex in test_data]

    # ---------- vLLM 一次性生成 ----------
    print("🚀 Generating with vLLM ...")
    vllm_outputs = llm.generate(prompts, sampling_params)

    # ---------- 评估 ----------
    print("📊 Evaluating results ...")

    results = []
    correct_count = 0
    meta_accuracy = {}

    for i, (item, out) in enumerate(zip(test_data, vllm_outputs)):
        # vLLM 0.6.3 的每个结果
        generated_text = out.outputs[0].text

        # 正确答案
        correct_answer = item['answer_idx']  # e.g., 'A', 'B', 'C', etc.

        # 获取有效选项
        valid_options = list(item['options'].keys())

        # 模型预测
        predicted_answer = extract_answer_from_output(generated_text, valid_options)
        is_correct = (predicted_answer == correct_answer)
        if is_correct:
            correct_count += 1

        # 按 meta_info 统计
        meta = item.get("meta_info", "unknown")
        if meta not in meta_accuracy:
            meta_accuracy[meta] = {"correct": 0, "total": 0}
        meta_accuracy[meta]["total"] += 1
        if is_correct:
            meta_accuracy[meta]["correct"] += 1

        results.append({
            "id": i,
            "question": item["question"][:100] + "...",
            "meta_info": meta,
            "num_options": len(valid_options),
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "full_output": generated_text[:300] + "..." if len(generated_text) > 300 else generated_text,
        })

    accuracy = correct_count / test_samples

    print(f"\n🎯 MedQA Test Results ({test_samples} samples):")
    print(f"Overall Accuracy: {accuracy:.3f} ({correct_count}/{test_samples})")

    print("\n📚 Accuracy by Category:")
    for meta, stats in sorted(meta_accuracy.items()):
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            print(f"  {meta}: {acc:.3f} ({stats['correct']}/{stats['total']})")

    # Show some examples
    print(f"\n📝 Sample Results (first 5):")
    for i in range(min(5, len(results))):
        result = results[i]
        status = "✅" if result['is_correct'] else "❌"
        print(f"  {status} Sample {i+1}: {result['question']}")
        print(f"     Category: {result['meta_info']}")
        print(f"     Correct: {result['correct_answer']}, Predicted: {result['predicted_answer']}")
        print()

    # 保存结果
    output_file = "./medqa_vllm_0.6.3_results.json"
    output_data = {
        "model_id": model_id,
        "dataset": "MedQA",
        "test_samples": test_samples,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "max_tokens": sampling_params.max_tokens,
        "meta_accuracy": {
            k: v["correct"] / v["total"] if v["total"] > 0 else 0 
            for k, v in meta_accuracy.items()
        },
        "results": results,
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {output_file}")
    print(f"🎉 Done. Final accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()

