from huggingface_hub import HfApi
import os

# 初始化API
api = HfApi()

# 创建仓库（如果不存在）
try:
    api.create_repo("RedMist137/alphamed_search", repo_type="dataset")
except:
    pass  # 仓库已存在

# 直接上传原始文件，保持目录结构
api.upload_file(
    path_or_fileobj="data/alphamed_search_split/train.parquet",
    path_in_repo="train.parquet",
    repo_id="RedMist137/alphamed_search",
    repo_type="dataset"
)

api.upload_file(
    path_or_fileobj="data/alphamed_search_split/test.parquet", 
    path_in_repo="test.parquet",
    repo_id="RedMist137/alphamed_search",
    repo_type="dataset"
)

print("上传完成！文件名保持为 train.parquet 和 test.parquet")
