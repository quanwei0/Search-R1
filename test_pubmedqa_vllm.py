#!/usr/bin/env python3
"""
Test AlphaMed model on PubMedQA test split with batch inference using vLLM (0.6.3)
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


def build_prompt_cot(example):
    """Build prompt for PubMedQA question with Chain-of-Thought"""
    question = example["question"]

    prompt = "You are a medical AI assistant.\n"
    prompt += f"Question: {question}\n"
    prompt += "Options:\n"
    prompt += "A. yes\n"
    prompt += "B. no\n"
    prompt += "Please reason step by step, and put the final answer in \\boxed{}.\n"
    
    return prompt


def build_prompt_direct(example):
    """Build prompt for PubMedQA question - Direct inference without CoT"""
    question = example["question"]

    prompt = "You are a medical AI assistant.\n"
    prompt += f"Question: {question}\n"
    prompt += "Options:\n"
    prompt += "A. yes\n"
    prompt += "B. no\n"
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

    print("📁 Loading PubMedQA test dataset...")
    test_file = "/home/li003968/Search-R1/data/pubmedqa_test.jsonl"
    
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_data.append(json.loads(line.strip()))
    
    test_samples = len(test_data)
    print(f"📊 Testing on {test_samples} samples from PubMedQA")

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
    answer_distribution = {'A': {'correct': 0, 'total': 0}, 'B': {'correct': 0, 'total': 0}}

    for i, (item, out) in enumerate(zip(test_data, vllm_outputs)):
        # vLLM 0.6.3 的每个结果
        generated_text = out.outputs[0].text

        # 正确答案
        correct_answer = item['answer_idx']  # 'A' or 'B'

        # 模型预测
        predicted_answer = extract_answer_from_output(generated_text)
        is_correct = (predicted_answer == correct_answer)
        if is_correct:
            correct_count += 1

        # 按答案类型统计
        if correct_answer in answer_distribution:
            answer_distribution[correct_answer]['total'] += 1
            if is_correct:
                answer_distribution[correct_answer]['correct'] += 1

        results.append({
            "id": i,
            "pubid": item.get("pubid", f"pubmed_{i}"),
            "question": item["question"][:100] + "...",
            "correct_answer": correct_answer,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct,
            "full_output": generated_text[:300] + "..." if len(generated_text) > 300 else generated_text,
        })

    accuracy = correct_count / test_samples

    print(f"\n🎯 PubMedQA Test Results ({test_samples} samples):")
    print(f"Overall Accuracy: {accuracy:.3f} ({correct_count}/{test_samples})")

    print("\n📊 Accuracy by Answer Type:")
    for answer, stats in sorted(answer_distribution.items()):
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            answer_text = 'yes' if answer == 'A' else 'no'
            print(f"  {answer} ({answer_text}): {acc:.3f} ({stats['correct']}/{stats['total']})")

    # Show some examples
    print(f"\n📝 Sample Results (first 5):")
    for i in range(min(5, len(results))):
        result = results[i]
        status = "✅" if result['is_correct'] else "❌"
        print(f"  {status} Sample {i+1}: {result['question']}")
        print(f"     PubID: {result['pubid']}")
        print(f"     Correct: {result['correct_answer']}, Predicted: {result['predicted_answer']}")
        print()

    # 保存结果
    output_file = "./pubmedqa_vllm_0.6.3_results.json"
    output_data = {
        "model_id": model_id,
        "dataset": "PubMedQA pqa_labeled",
        "test_samples": test_samples,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "max_tokens": sampling_params.max_tokens,
        "answer_distribution": {
            k: v["correct"] / v["total"] if v["total"] > 0 else 0 
            for k, v in answer_distribution.items()
        },
        "results": results,
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {output_file}")
    print(f"🎉 Done. Final accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()

