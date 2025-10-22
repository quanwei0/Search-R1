#!/usr/bin/env python3
"""
Download and process MedMCQA test split to match MedQA format
"""

import json
import pandas as pd
import argparse
import os
from datasets import load_dataset

def download_medmcqa_test(output_dir='/home/li003968/Search-R1/data', sample_size=1000):
    """
    Download MedMCQA validation split from HuggingFace datasets and randomly sample
    """
    print("🚀 开始下载MedMCQA validation split...")
    
    # 加载MedMCQA数据集的validation split作为test
    dataset = load_dataset("openlifescienceai/medmcqa", split="validation")
    
    print(f"✅ 下载完成! 原始共 {len(dataset)} 个样本")
    
    # 随机采样
    import random
    random.seed(42)  # 设置随机种子以保证可复现
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    sampled_indices = sorted(indices[:sample_size])
    dataset = dataset.select(sampled_indices)
    
    print(f"🎲 随机选取 {len(dataset)} 个样本")
    
    # 转换为MedQA格式
    processed_data = []
    
    for idx, item in enumerate(dataset):
        # MedMCQA字段: id, question, opa, opb, opc, opd, cop (correct option: 0,1,2,3), exp, subject_name, topic_name
        
        # 构建options字典
        options = {
            'A': item['opa'].strip(),
            'B': item['opb'].strip(),
            'C': item['opc'].strip(),
            'D': item['opd'].strip()
        }
        
        # 将cop (0,1,2,3) 转换为 (A,B,C,D)
        cop_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        correct_answer = cop_to_letter[item['cop']]
        
        # 构建meta_info (使用subject_name)
        meta_info = item.get('subject_name', 'unknown')
        if pd.isna(meta_info) or meta_info == '':
            meta_info = 'unknown'
        
        # 构建MedQA格式的数据项
        processed_item = {
            'question': item['question'].strip(),
            'answer': '',  # 不使用explanation
            'options': options,
            'meta_info': meta_info,
            'answer_idx': correct_answer
        }
        
        processed_data.append(processed_item)
    
    # 保存为jsonl格式
    output_file = os.path.join(output_dir, 'medmcqa_test.jsonl')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"💾 数据已保存到: {output_file}")
    print(f"📊 总样本数: {len(processed_data)}")
    
    # 统计meta_info分布
    meta_counts = pd.Series([item['meta_info'] for item in processed_data]).value_counts()
    print(f"\n📈 按学科分布:")
    for subject, count in meta_counts.items():
        print(f"  - {subject}: {count} 样本")
    
    return output_file

def make_prefix(question_text, options):
    """
    Generate adaptive prompt that encourages strategic search for unfamiliar concepts
    """
    # 构建选项文本
    options_text = ""
    for letter in sorted(options.keys()):
        options_text += f"{letter}: {options[letter]}\n"
    
    prefix = f"""Answer the given medical multiple choice question. \
Think step-by-step inside <think> and </think> tags. \
When you encounter:
- Unfamiliar medical terminology or drug names
- Complex disease mechanisms you're uncertain about
- Specific treatment protocols or guidelines you need to verify
- Any information where you lack confidence

You can search for clarification using <search> query </search>, and results will appear between <information> and </information>. \
Use search strategically to fill knowledge gaps and improve answer accuracy. \
After sufficient reasoning and any necessary searches, provide your final answer inside <answer> and </answer> with ONLY the letter (e.g., <answer>A</answer>).

Question: {question_text}

Options:
{options_text.strip()}
"""
    
    return prefix

def process_jsonl_to_alphamed_format(input_file, output_file, data_source='medmcqa_test'):
    """
    Convert medmcqa_test.jsonl to alphamed_search format (same as MedQA)
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
                "id": f"medmcqa_test_{idx}",
                "question": question_text,
                "golden_answers": [correct_answer],
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": prompt_content,
                }],
                "ability": "medical-reasoning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": solution
                },
                "extra_info": {
                    'split': 'test',
                    'index': idx,
                    'subset_name': f'MedMCQA-Test-{meta_info}',
                    'original_question': question_text,
                    'meta_info': meta_info,
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
    print(f"\n📝 第一个样本示例:")
    sample = processed_data[0]
    print(f"ID: {sample['id']}")
    print(f"Question: {sample['question'][:100]}...")
    print(f"Golden Answer: {sample['golden_answers']}")
    print(f"Options: {list(sample['reward_model']['ground_truth']['options'].keys())}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Download MedMCQA test split and convert to alphamed_search format')
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
    parser.add_argument('--sample_size',
                       type=int,
                       default=1000,
                       help='Number of samples to randomly select from test split')
    
    args = parser.parse_args()
    
    jsonl_file = os.path.join(args.data_dir, 'medmcqa_test.jsonl')
    parquet_file = os.path.join(args.output_dir, 'medmcqa_test.parquet')
    
    if not args.process_only:
        # 下载数据
        jsonl_file = download_medmcqa_test(args.data_dir, args.sample_size)
    
    if not args.download_only:
        # 处理数据
        print(f"\n🚀 开始转换MedMCQA test数据到alphamed格式...")
        print(f"📁 输入文件: {jsonl_file}")
        print(f"📁 输出文件: {parquet_file}")
        print(f"📝 使用自适应搜索提示词 (adaptive prompt)")
        
        # 检查输入文件是否存在
        if not os.path.exists(jsonl_file):
            print(f"❌ 错误: 输入文件不存在: {jsonl_file}")
            return
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 处理文件
        df = process_jsonl_to_alphamed_format(
            jsonl_file, 
            parquet_file, 
            'medmcqa_test'
        )
        
        print(f"\n🎉 转换完成!")
        print(f"✅ MedMCQA test dataset已保存到: {parquet_file}")

if __name__ == '__main__':
    main()

