#!/usr/bin/env python3
"""
Convert baseline_test/test.jsonl to parquet format with adaptive prompt
"""

import json
import pandas as pd
import os

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

def process_jsonl_to_parquet(input_file, output_file, data_source='baseline_test'):
    """
    Convert test.jsonl to parquet format with adaptive prompt
    """
    processed_data = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip():  # 跳过空行
                continue
                
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
            
            # 构建数据项
            processed_item = {
                "id": f"baseline_test_{idx}",
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
                    'subset_name': f'BaselineTest-{meta_info}',
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

if __name__ == '__main__':
    # 设置路径
    input_file = '/home/li003968/Search-R1/baseline_test/test.jsonl'
    output_file = '/home/li003968/Search-R1/baseline_test/test.parquet'
    
    print("🚀 开始转换baseline_test/test.jsonl到parquet格式...")
    print(f"📁 输入文件: {input_file}")
    print(f"📁 输出文件: {output_file}")
    print(f"📝 使用自适应搜索提示词 (adaptive prompt)")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 错误: 输入文件不存在: {input_file}")
        exit(1)
    
    # 处理文件
    df = process_jsonl_to_parquet(input_file, output_file)
    
    print(f"\n🎉 转换完成!")
    print(f"✅ Test dataset已保存到: {output_file}")




