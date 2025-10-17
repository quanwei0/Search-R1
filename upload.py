from huggingface_hub import HfApi
import os
import pandas as pd

# 初始化API
api = HfApi()

# 仓库名称
repo_id = "RedMist137/alphamed_search_adaptive_prompt"

# 创建仓库（如果不存在）
try:
    api.create_repo(repo_id, repo_type="dataset", private=False)
    print(f"✅ 创建仓库: {repo_id}")
except Exception as e:
    print(f"ℹ️  仓库已存在或创建失败: {e}")

# 数据文件路径
train_file = "data/alphamed_search/train.parquet"
test_file = "data/alphamed_search/test.parquet"

# 显示数据统计
print("\n📊 数据集统计:")
if os.path.exists(train_file):
    train_df = pd.read_parquet(train_file)
    print(f"  Train: {len(train_df)} 样本")
else:
    print(f"  ❌ Train文件不存在: {train_file}")

if os.path.exists(test_file):
    test_df = pd.read_parquet(test_file)
    print(f"  Test: {len(test_df)} 样本")
else:
    print(f"  ❌ Test文件不存在: {test_file}")

print(f"\n🚀 开始上传到 {repo_id}...")

# 上传训练集
if os.path.exists(train_file):
    print("\n📤 上传 train.parquet...")
    api.upload_file(
        path_or_fileobj=train_file,
        path_in_repo="train.parquet",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload training dataset with adaptive prompt (19,778 samples)"
    )
    print("  ✅ train.parquet 上传完成")

# 上传测试集
if os.path.exists(test_file):
    print("\n📤 上传 test.parquet...")
    api.upload_file(
        path_or_fileobj=test_file,
        path_in_repo="test.parquet",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload test dataset with adaptive prompt (500 samples)"
    )
    print("  ✅ test.parquet 上传完成")

print(f"\n🎉 上传完成！")
print(f"🔗 查看数据集: https://huggingface.co/datasets/{repo_id}")
print(f"\n💡 使用方法:")
print(f"```python")
print(f"from datasets import load_dataset")
print(f"dataset = load_dataset('{repo_id}')")
print(f"```")
