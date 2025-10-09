#!/usr/bin/env python3
"""
Convert test.jsonl to alphamed_search dataset format
Based on the existing alphamed_search.py processing logic
"""

import json
import pandas as pd
import argparse
import os

def make_prefix(question_text, options, template_type='base'):
    """
    Generate prompt prefix based on alphamed_search.py logic
    """
    if template_type == 'base':
        # 构建选项文本
        options_text = ""
        for letter in sorted(options.keys()):
            options_text += f"{letter}: {options[letter]}\n"
        
        prefix = f"""Answer the given medical multiple choice question. \
You must conduct reasoning inside <think> and </think> first every time you get new information. \
For this medical question, you MUST search for information about EACH option before making your final decision. \
You can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You MUST search for each option (A, B, C, D, E if present) one by one to gather relevant medical information. \
After searching for ALL options, analyze the search results and provide your final answer inside <answer> and </answer> with ONLY the letter (e.g., <answer>A</answer>). \
Your answer MUST be one of the provided options.

Question: {question_text}

Options:
{options_text.strip()}
"""
    else:
        raise NotImplementedError
    
    return prefix

def process_jsonl_to_alphamed_format(input_file, output_file, data_source='test_medical'):
    """
    Convert test.jsonl to alphamed_search format
    """
    processed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            data = json.loads(line.strip())
            
            question_text = data['question'].strip()
            correct_answer = data['answer_idx'].strip()  # 使用answer_idx作为正确答案
            options = data['options']
            meta_info = data.get('meta_info', 'unknown')
            
            # 生成prompt
            prompt_content = make_prefix(question_text, options, template_type='base')
            
            # 构建solution
            solution = {
                "target": [correct_answer],  # 保持与原格式一致，使用列表
                "options": options,  # 添加选项信息用于验证
            }
            
            # 构建数据项，完全按照alphamed_search格式
            processed_item = {
                "id": f"test_{idx}",
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
                    'subset_name': f'MedQA-Test-{meta_info}',
                    'original_question': question_text,
                    'meta_info': meta_info,
                    'original_answer': data.get('answer', ''),  # 保留原始答案文本
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
    parser = argparse.ArgumentParser(description='Convert test.jsonl to alphamed_search format')
    parser.add_argument('--input_file', 
                       default='/home/li003968/Search-R1/data/test.jsonl',
                       help='Input JSONL file path')
    parser.add_argument('--output_file', 
                       default='/home/li003968/Search-R1/data/test_alphamed_format.parquet',
                       help='Output parquet file path')
    parser.add_argument('--data_source', 
                       default='test_medical',
                       help='Data source identifier')
    
    args = parser.parse_args()
    
    print("🚀 开始转换test.jsonl到alphamed格式...")
    print(f"📁 输入文件: {args.input_file}")
    print(f"📁 输出文件: {args.output_file}")
    print(f"🏷️  数据源: {args.data_source}")
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input_file):
        print(f"❌ 错误: 输入文件不存在: {args.input_file}")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 处理文件
    df = process_jsonl_to_alphamed_format(
        args.input_file, 
        args.output_file, 
        args.data_source
    )
    
    print(f"\n🎉 转换完成! 现在你可以:")
    print(f"1. 检查输出文件: {args.output_file}")
    print(f"2. 上传到Hugging Face作为test dataset")
    print(f"3. 使用相同的reward函数进行评估")

if __name__ == '__main__':
    main()
