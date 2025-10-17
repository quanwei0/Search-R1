# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the AlphaMed19K dataset to parquet format for medical multiple choice questions
"""

import re
import os
import pandas as pd
import argparse

try:
    from verl.utils.hdfs_io import copy, makedirs
except ImportError:
    # 如果 verl 不可用，使用标准库替代
    import shutil
    def copy(src, dst):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    def makedirs(path):
        os.makedirs(path, exist_ok=True)


def extract_options_from_question(question_text):
    """
    从问题文本中提取选项
    """
    options = {}
    
    # 分割文本，寻找选项
    parts = re.split(r'\s+([A-E]):\s*', question_text)
    
    # 第一部分是问题主干，后面是选项
    if len(parts) > 1:
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                letter = parts[i]
                content = parts[i + 1].strip()
                # 清理内容，去除多余的空白字符
                cleaned_content = re.sub(r'\s+', ' ', content)
                options[letter] = cleaned_content
    
    return options


def extract_question_stem(question_text):
    """
    从完整问题文本中提取问题主干（去除选项部分）
    """
    # 找到第一个选项的位置
    first_option_match = re.search(r'\s+([A-E]):\s*', question_text)
    if first_option_match:
        return question_text[:first_option_match.start()].strip()
    return question_text.strip()


def make_prefix(dp, template_type):
    question_text = dp['question']
    
    # 提取问题主干和选项
    question_stem = extract_question_stem(question_text)
    options = extract_options_from_question(question_text)
    
    if template_type == 'base':
        """医学选择题模板，要求对每个选项进行搜索（旧版）"""
        
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

Question: {question_stem}

Options:
{options_text.strip()}
"""
    
    elif template_type == 'adaptive':
        """自适应搜索模板，建议在遇到不熟悉概念时搜索"""
        
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

Question: {question_stem}

Options:
{options_text.strip()}
"""
    
    else:
        raise NotImplementedError
    
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', type=str, default='alphamed19k')
    parser.add_argument('--input_file', default='./data/AlphaMed19K/train19k.parquet')
    parser.add_argument('--local_dir', default='./data/alphamed_search')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--template_type', type=str, default='adaptive', choices=['base', 'adaptive'],
                        help='Prompt template: base (strict search all) or adaptive (strategic search)')

    args = parser.parse_args()

    data_source = args.data_source

    # 读取 parquet 文件
    df = pd.read_parquet(args.input_file)
    
    # 创建输出目录
    os.makedirs(args.local_dir, exist_ok=True)

    def process_fn(row, idx, split='train'):
        question_text = row['question'].strip()
        correct_answer = row['answer'].strip()
        subset_name = row['subset_name']
        
        # 提取选项用于验证
        options = extract_options_from_question(question_text)
        question_stem = extract_question_stem(question_text)
        
        # 生成 prompt
        question = make_prefix(row, template_type=args.template_type)
        
        solution = {
            "target": [correct_answer],  # 保持与原格式一致，使用列表
            "options": options,  # 添加选项信息用于验证
        }

        data = {
            "id": f"{split}_{idx}",  # 添加 id 字段
            "question": question_stem,  # 添加纯问题字段
            "golden_answers": [correct_answer],  # 添加 golden_answers 字段
            "data_source": data_source,
            "prompt": [{
                "role": "user",
                "content": question,
            }],
            "ability": "medical-reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": solution
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'subset_name': subset_name,
                'original_question': question_text,
            },
            "metadata": None  # 添加 metadata 字段
        }
        return data

    # 使用所有数据作为训练集，不划分test
    total_samples = len(df)
    train_df = df
    
    print(f"Total samples: {total_samples}")
    print(f"Train samples: {len(train_df)}")
    print(f"Using adaptive prompt template")
    
    # 处理训练集数据
    train_processed_data = []
    for idx, row in train_df.iterrows():
        train_processed_data.append(process_fn(row, idx, split='train'))

    # 转换为 DataFrame 并保存
    train_processed_df = pd.DataFrame(train_processed_data)
    
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    # 保存训练集
    train_processed_df.to_parquet(os.path.join(local_dir, 'train.parquet'))
    
    print(f"✅ Saved train set to {os.path.join(local_dir, 'train.parquet')}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
