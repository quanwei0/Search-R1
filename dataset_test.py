import pyarrow.parquet as pq
import os

# Read the dataset from the folder
dataset_path = "./data/hotpotqa_search"

# Read train parquet file
train_table = pq.read_table(os.path.join(dataset_path, "train.parquet"))
test_table = pq.read_table(os.path.join(dataset_path, "test.parquet"))

# Show dataset size
print(f"Dataset loaded from: {dataset_path}")
print(f"Train dataset size: {train_table.num_rows}")
print(f"Test dataset size: {test_table.num_rows}")
print(f"Total dataset size: {train_table.num_rows + test_table.num_rows}")
print(f"\nTrain columns: {train_table.column_names}")

# # Show one sample data
# print("\n" + "="*50)
# print("Sample data from train set:")
# print("="*50)
# for col_name in train_table.column_names:
#     print(f"{col_name}: {train_table[col_name][0].as_py()}")
