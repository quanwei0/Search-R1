#!/usr/bin/env python3
"""
Upload test dataset to Hugging Face Hub
"""

from huggingface_hub import HfApi
import pandas as pd
import os

def main():
    # 初始化API
    api = HfApi()
    
    repo_id = "RedMist137/alphamed_search"
    test_file_path = "data/test_alphamed_format.parquet"
    
    print("🚀 上传test数据集到Hugging Face...")
    print(f"📁 文件: {test_file_path}")
    print(f"🎯 仓库: {repo_id}")
    
    # 检查文件是否存在
    if not os.path.exists(test_file_path):
        print(f"❌ 错误: 文件不存在: {test_file_path}")
        return
    
    # 读取数据查看统计信息
    df = pd.read_parquet(test_file_path)
    print(f"📊 数据统计: {len(df)} 个样本")
    
    try:
        # 上传test文件，重命名为test.parquet以保持一致性
        api.upload_file(
            path_or_fileobj=test_file_path,
            path_in_repo="test_medical.parquet",  # 使用不同名称避免覆盖原test.parquet
            repo_id=repo_id,
            repo_type="dataset",
            commit_message="Add test_medical dataset (converted from test.jsonl)"
        )
        
        print("✅ 上传成功!")
        print(f"🔗 查看数据集: https://huggingface.co/datasets/{repo_id}")
        print(f"📄 新文件: test_medical.parquet")
        
        print("\n💡 使用方法:")
        print("```python")
        print("from datasets import load_dataset")
        print(f'dataset = load_dataset("{repo_id}", data_files="test_medical.parquet")')
        print("```")
        
    except Exception as e:
        print(f"❌ 上传失败: {e}")
        print("\n💡 请确保:")
        print("1. 已登录 Hugging Face: huggingface-cli login")
        print("2. 有权限访问该仓库")

if __name__ == "__main__":
    main()
