#!/usr/bin/env python3
"""
Convert MMLU-Pro health data to baseline test format (simple jsonl)
"""

import json
import os

def convert_mmlu_to_baseline_format(
    input_file='/home/li003968/Search-R1/data/mmlu_pro_health_test.jsonl',
    output_file='/home/li003968/Search-R1/baseline_test/mmlu_pro_health_test.jsonl'
):
    """
    Convert MMLU-Pro health jsonl to baseline test format
    Format matches: question, answer, options, meta_info, answer_idx
    """
    print("🚀 开始转换MMLU-Pro Health数据到baseline格式...")
    print(f"📁 输入文件: {input_file}")
    print(f"📁 输出文件: {output_file}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 错误: 输入文件不存在: {input_file}")
        return
    
    # 创建输出目录
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 读取并转换数据
    converted_count = 0
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            data = json.loads(line.strip())
            
            # 数据已经是正确格式，直接写入
            # 格式: question, answer, options, meta_info, answer_idx
            baseline_item = {
                'question': data['question'],
                'answer': data.get('answer', ''),
                'options': data['options'],
                'meta_info': data.get('meta_info', 'mmlu_pro_health'),
                'answer_idx': data['answer_idx']
            }
            
            f_out.write(json.dumps(baseline_item, ensure_ascii=False) + '\n')
            converted_count += 1
    
    print(f"✅ 转换完成!")
    print(f"📊 总样本数: {converted_count}")
    print(f"💾 保存到: {output_file}")
    
    # 显示第一个样本
    with open(output_file, 'r', encoding='utf-8') as f:
        first_line = f.readline()
        sample = json.loads(first_line)
        print(f"\n📝 第一个样本示例:")
        print(f"Question: {sample['question'][:100]}...")
        print(f"Answer idx: {sample['answer_idx']}")
        print(f"Options: {list(sample['options'].keys())}")
        print(f"Meta info: {sample['meta_info']}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert MMLU-Pro health data to baseline test format')
    parser.add_argument('--input_file', 
                       default='/home/li003968/Search-R1/data/mmlu_pro_health_test.jsonl',
                       help='Input JSONL file path')
    parser.add_argument('--output_file', 
                       default='/home/li003968/Search-R1/baseline_test/mmlu_pro_health_test.jsonl',
                       help='Output JSONL file path (baseline format)')
    
    args = parser.parse_args()
    
    convert_mmlu_to_baseline_format(args.input_file, args.output_file)

if __name__ == '__main__':
    main()


