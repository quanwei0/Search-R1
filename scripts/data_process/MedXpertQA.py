#!/usr/bin/env python3
"""
Download and process MedXpertQA to match MedQA format
"""

import json
import pandas as pd
import argparse
import os
from datasets import load_dataset

def download_medxpertqa(output_dir='/home/li003968/Search-R1/data', subset='Text', max_samples=None):
    """
    Download MedXpertQA dataset from HuggingFace datasets
    
    Args:
        output_dir: Directory to save the processed data
        subset: 'Text' (2.46k) or 'MM' (2.01k)
        max_samples: Maximum number of samples to process (None for all)
    """
    print(f"🚀 开始下载MedXpertQA数据集 (subset: {subset})...")
    
    # 加载MedXpertQA数据集
    try:
        # 使用test split (2450 samples for Text subset)
        dataset = load_dataset("TsinghuaC3I/MedXpertQA", subset, split="test")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None
    
    print(f"✅ 下载完成! 共 {len(dataset)} 个样本")
    
    # 限制样本数
    if max_samples and max_samples < len(dataset):
        dataset = dataset.select(range(max_samples))
        print(f"🎲 选取前 {max_samples} 个样本")
    
    # 转换为MedQA格式
    processed_data = []
    
    for idx, item in enumerate(dataset):
        # MedXpertQA字段: question, options (dict), label
        
        question_text = item['question'].strip()
        options = item['options']
        correct_answer = item['label'].strip()  # 'A', 'B', 'C', etc.
        
        # 获取category信息（如果有）
        category = item.get('category', item.get('medical_task', 'medxpert'))
        
        # 构建MedQA格式的数据项
        processed_item = {
            'question': question_text,
            'answer': '',  # MedXpertQA没有详细解释
            'options': options,
            'meta_info': category if category else 'medxpert',
            'answer_idx': correct_answer
        }
        
        processed_data.append(processed_item)
    
    # 保存为jsonl格式
    output_file = os.path.join(output_dir, f'medxpertqa_{subset.lower()}.jsonl')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"💾 数据已保存到: {output_file}")
    print(f"📊 总样本数: {len(processed_data)}")
    
    # 统计meta_info分布
    if processed_data:
        meta_counts = pd.Series([item['meta_info'] for item in processed_data]).value_counts()
        print(f"\n📈 按类别分布:")
        for meta, count in meta_counts.items():
            print(f"  - {meta}: {count} 样本")
    
    # 统计选项数量分布
    option_counts = pd.Series([len(item['options']) for item in processed_data]).value_counts()
    print(f"\n📊 按选项数量分布:")
    for num_options, count in sorted(option_counts.items()):
        print(f"  - {num_options}个选项: {count} 样本")
    
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

def process_jsonl_to_alphamed_format(input_file, output_file, data_source='medxpertqa'):
    """
    Convert medxpertqa jsonl to alphamed_search format (same as MedQA)
    """
    processed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line.strip())
            
            question_text = data['question'].strip()
            correct_answer = data['answer_idx'].strip()
            options = data['options']
            meta_info = data.get('meta_info', 'medxpertqa')
            
            # 生成prompt
            prompt_content = make_prefix(question_text, options)
            
            # 构建solution
            solution = {
                "target": [correct_answer],
                "options": options,
            }
            
            # 构建数据项，完全按照alphamed_search格式
            processed_item = {
                "id": f"medxpertqa_{idx}",
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
                    'subset_name': f'MedXpertQA-{meta_info}',
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
    if len(processed_data) > 0:
        print(f"\n📝 第一个样本示例:")
        sample = processed_data[0]
        print(f"ID: {sample['id']}")
        print(f"Question: {sample['question'][:100]}...")
        print(f"Golden Answer: {sample['golden_answers']}")
        print(f"Options: {list(sample['reward_model']['ground_truth']['options'].keys())}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Download MedXpertQA and convert to alphamed_search format')
    parser.add_argument('--download_only', action='store_true',
                       help='Only download data without processing')
    parser.add_argument('--process_only', action='store_true',
                       help='Only process existing jsonl file')
    parser.add_argument('--data_dir', 
                       default='/home/li003968/Search-R1/data',
                       help='Data directory')
    parser.add_argument('--output_dir',
                       default='/home/li003968/Search-R1/data/MedXpertQA',
                       help='Output directory for processed parquet files')
    parser.add_argument('--subset',
                       default='Text',
                       choices=['Text', 'MM'],
                       help='MedXpertQA subset: Text (2.46k) or MM (2.01k)')
    parser.add_argument('--max_samples',
                       type=int,
                       default=None,
                       help='Maximum number of samples to process')
    
    args = parser.parse_args()
    
    jsonl_file = os.path.join(args.data_dir, f'medxpertqa_{args.subset.lower()}.jsonl')
    parquet_file = os.path.join(args.output_dir, 'test.parquet')
    
    if not args.process_only:
        # 下载数据
        result = download_medxpertqa(args.data_dir, args.subset, args.max_samples)
        if result is None:
            print("❌ 下载失败，退出")
            return
        jsonl_file = result
    
    if not args.download_only:
        # 处理数据
        print(f"\n🚀 开始转换MedXpertQA数据到alphamed格式...")
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
            f'medxpertqa_{args.subset.lower()}'
        )
        
        # 创建一个train文件（复制test的前几个样本作为dummy train）
        train_parquet = os.path.join(args.output_dir, 'train.parquet')
        train_df = df.head(min(100, len(df)))
        train_df.to_parquet(train_parquet)
        
        print(f"\n🎉 转换完成!")
        print(f"✅ Test dataset已保存到: {parquet_file}")
        print(f"✅ Train dataset (dummy)已保存到: {train_parquet}")

if __name__ == '__main__':
    main()

