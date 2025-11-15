#!/usr/bin/env python3
"""
Download and process GPQA dataset to match MedQA format
GPQA: Graduate-Level Google-Proof Q&A Benchmark
"""

import json
import pandas as pd
import argparse
import os
import random
from datasets import load_dataset

def download_gpqa(output_dir='/home/li003968/Search-R1/data', subset='gpqa_main'):
    """
    Download GPQA dataset from HuggingFace datasets
    
    Args:
        output_dir: Directory to save the processed data
        subset: GPQA subset to use (gpqa_main, gpqa_diamond, gpqa_extended)
    """
    print(f"🚀 开始下载GPQA数据集 (subset: {subset})...")
    
    # 加载GPQA数据集
    try:
        dataset = load_dataset("Idavidrein/gpqa", subset)
    except:
        print("⚠️  尝试备用数据源...")
        dataset = load_dataset("google/gpqa", subset)
    
    # GPQA只有train split，我们需要手动分割成train和test
    full_data = dataset["train"]
    
    print(f"✅ 下载完成! 共 {len(full_data)} 个样本")
    
    # 转换为MedQA格式
    processed_data = []
    
    for idx, item in enumerate(full_data):
        # GPQA字段: Question, Correct Answer, Incorrect Answer 1/2/3, etc.
        question = item['Question']
        correct_answer_text = item['Correct Answer']
        incorrect_answers = [
            item['Incorrect Answer 1'],
            item['Incorrect Answer 2'],
            item['Incorrect Answer 3']
        ]
        
        # 创建所有选项的列表并打乱顺序
        all_answers = [correct_answer_text] + incorrect_answers
        random.seed(42 + idx)  # 使用固定种子确保可复现
        random.shuffle(all_answers)
        
        # 找出正确答案的位置
        correct_idx = all_answers.index(correct_answer_text)
        idx_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        correct_answer = idx_to_letter[correct_idx]
        
        # 构建options字典
        options = {
            'A': all_answers[0],
            'B': all_answers[1],
            'C': all_answers[2],
            'D': all_answers[3]
        }
        
        # 获取subdomain信息
        subdomain = item.get('Subdomain', 'unknown')
        
        # 构建MedQA格式的数据项
        processed_item = {
            'question': question.strip(),
            'answer': '',  # GPQA没有详细解释
            'options': options,
            'meta_info': subdomain,
            'answer_idx': correct_answer,
            'subdomain': subdomain
        }
        
        processed_data.append(processed_item)
    
    # 随机打乱数据
    random.seed(42)
    random.shuffle(processed_data)
    
    # 分割成train和test (80/20分割)
    split_idx = int(len(processed_data) * 0.8)
    train_data = processed_data[:split_idx]
    test_data = processed_data[split_idx:]
    
    # 保存为jsonl格式
    train_file = os.path.join(output_dir, f'gpqa_{subset}_train.jsonl')
    test_file = os.path.join(output_dir, f'gpqa_{subset}_test.jsonl')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存train
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存test
    with open(test_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"💾 数据已保存:")
    print(f"   Train: {train_file} ({len(train_data)} 样本)")
    print(f"   Test:  {test_file} ({len(test_data)} 样本)")
    print(f"📊 总样本数: {len(processed_data)}")
    
    # 统计subdomain分布
    train_subdomains = pd.Series([item['subdomain'] for item in train_data]).value_counts()
    test_subdomains = pd.Series([item['subdomain'] for item in test_data]).value_counts()
    
    print(f"\n📈 Train按subdomain分布:")
    for subdomain, count in train_subdomains.items():
        print(f"  - {subdomain}: {count} 样本")
    
    print(f"\n📈 Test按subdomain分布:")
    for subdomain, count in test_subdomains.items():
        print(f"  - {subdomain}: {count} 样本")
    
    return train_file, test_file

def make_prefix(question_text, options):
    """
    Generate adaptive prompt that encourages strategic search for unfamiliar concepts
    """
    # 构建选项文本
    options_text = ""
    for letter in sorted(options.keys()):
        options_text += f"{letter}: {options[letter]}\n"
    
    prefix = f"""Answer the given multiple choice question. \
Think step-by-step inside <think> and </think> tags. \
When you encounter:
- Unfamiliar terminology or concepts
- Complex mechanisms you're uncertain about
- Specific facts or guidelines you need to verify
- Any information where you lack confidence

You can search for clarification using <search> query </search>, and results will appear between <information> and </information>. \
Use search strategically to fill knowledge gaps and improve answer accuracy. \
After sufficient reasoning and any necessary searches, provide your final answer inside <answer> and </answer> with ONLY the letter (e.g., <answer>A</answer>).

Question: {question_text}

Options:
{options_text.strip()}
"""
    
    return prefix

def process_jsonl_to_alphamed_format(input_file, output_file, data_source='gpqa', split='train'):
    """
    Convert gpqa jsonl to alphamed_search format
    """
    processed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line.strip())
            
            question_text = data['question'].strip()
            correct_answer = data['answer_idx'].strip()
            options = data['options']
            meta_info = data.get('meta_info', 'unknown')
            
            # 生成prompt
            prompt_content = make_prefix(question_text, options)
            
            # 构建solution
            solution = {
                "target": [correct_answer],
                "options": options,
            }
            
            # 构建数据项，完全按照alphamed_search格式
            processed_item = {
                "id": f"gpqa_{split}_{idx}",
                "question": question_text,
                "golden_answers": [correct_answer],
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": prompt_content,
                }],
                "ability": "scientific-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'subset_name': f'GPQA-{meta_info}',
                    'original_question': question_text,
                    'meta_info': meta_info,
                    'subdomain': data.get('subdomain', 'unknown'),
                    'original_answer': data.get('answer', ''),
                },
                "metadata": None
            }
            
            processed_data.append(processed_item)
    
    # 转换为DataFrame并保存为parquet
    df = pd.DataFrame(processed_data)
    df.to_parquet(output_file)
    
    print(f"✅ 处理完成!")
    print(f"📊 总样本数: {len(processed_data)}")
    print(f"💾 保存到: {output_file}")
    
    # 显示样本统计
    meta_info_counts = df['extra_info'].apply(lambda x: x['meta_info']).value_counts()
    print(f"\n📈 按meta_info分布:")
    for meta, count in meta_info_counts.items():
        print(f"  - {meta}: {count} 样本")
    
    # 显示第一个样本作为示例
    if len(processed_data) > 0:
        print(f"\n📝 第一个样本示例:")
        sample = processed_data[0]
        print(f"ID: {sample['id']}")
        print(f"Question: {sample['question'][:100]}...")
        print(f"Golden Answer: {sample['golden_answers']}")
        print(f"Options: {list(sample['reward_model']['ground_truth']['options'].keys())}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Download GPQA and convert to alphamed_search format')
    parser.add_argument('--download_only', action='store_true',
                       help='Only download data without processing')
    parser.add_argument('--process_only', action='store_true',
                       help='Only process existing jsonl file')
    parser.add_argument('--data_dir', 
                       default='/home/li003968/Search-R1/data',
                       help='Data directory')
    parser.add_argument('--output_dir',
                       default='/home/li003968/Search-R1/data/alphamed_search',
                       help='Output directory for processed parquet files')
    parser.add_argument('--subset',
                       default='gpqa_main',
                       choices=['gpqa_main', 'gpqa_diamond', 'gpqa_extended'],
                       help='GPQA subset to use')
    
    args = parser.parse_args()
    
    train_jsonl = os.path.join(args.data_dir, f'gpqa_{args.subset}_train.jsonl')
    test_jsonl = os.path.join(args.data_dir, f'gpqa_{args.subset}_test.jsonl')
    train_parquet = os.path.join(args.output_dir, f'gpqa_{args.subset}_train.parquet')
    test_parquet = os.path.join(args.output_dir, f'gpqa_{args.subset}_test.parquet')
    
    if not args.process_only:
        # 下载数据
        train_jsonl, test_jsonl = download_gpqa(args.data_dir, args.subset)
    
    if not args.download_only:
        # 处理train数据
        print(f"\n🚀 开始转换GPQA train数据到alphamed格式...")
        print(f"📁 输入文件: {train_jsonl}")
        print(f"📁 输出文件: {train_parquet}")
        
        if not os.path.exists(train_jsonl):
            print(f"❌ 错误: 输入文件不存在: {train_jsonl}")
            return
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        df_train = process_jsonl_to_alphamed_format(
            train_jsonl, 
            train_parquet, 
            f'gpqa_{args.subset}',
            'train'
        )
        
        # 处理test数据
        print(f"\n🚀 开始转换GPQA test数据到alphamed格式...")
        print(f"📁 输入文件: {test_jsonl}")
        print(f"📁 输出文件: {test_parquet}")
        
        if not os.path.exists(test_jsonl):
            print(f"❌ 错误: 输入文件不存在: {test_jsonl}")
            return
        
        df_test = process_jsonl_to_alphamed_format(
            test_jsonl, 
            test_parquet, 
            f'gpqa_{args.subset}',
            'test'
        )
        
        print(f"\n🎉 转换完成!")
        print(f"✅ GPQA train dataset已保存到: {train_parquet}")
        print(f"✅ GPQA test dataset已保存到: {test_parquet}")

if __name__ == '__main__':
    main()

