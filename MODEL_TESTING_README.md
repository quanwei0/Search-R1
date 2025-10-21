# Model Testing Guide

## 📝 测试脚本使用说明

### 1. Python脚本: `test_alphamed_batch.py`

批量推理脚本,支持命令行参数配置。

#### 参数说明:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model_id` | str | `che111/AlphaMed-8B-instruct-rl` | HuggingFace模型ID |
| `--batch_size` | int | 32 | 批量大小 |
| `--max_new_tokens` | int | 2048 | 最大生成token数 |
| `--test_file` | str | `baseline_test/test.jsonl` | 测试数据路径 |
| `--output_file` | str | 自动生成 | 输出JSON文件路径 |
| `--cuda_devices` | str | `0,1,2,3` | 使用的GPU设备 |
| `--num_samples` | int | None | 测试样本数(None=全部) |

#### 使用示例:

```bash
# 测试LLaMA3-8B模型,2048 tokens
python test_alphamed_batch.py \
    --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
    --batch_size 32 \
    --max_new_tokens 2048

# 测试Qwen2.5-7B模型,4096 tokens
python test_alphamed_batch.py \
    --model_id "Qwen/Qwen2.5-7B-Instruct" \
    --batch_size 16 \
    --max_new_tokens 4096

# 只测试前100个样本
python test_alphamed_batch.py \
    --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
    --num_samples 100

# 自定义输出文件
python test_alphamed_batch.py \
    --model_id "meta-llama/Meta-Llama-3-8B-Instruct" \
    --output_file "./my_results.json"
```

---

### 2. Bash脚本: `test_models_simple.sh`

自动化测试多个模型在不同配置下的性能。

#### 包含的测试:

- **LLaMA-3-8B-Instruct**: 2048 tokens, 4096 tokens
- **Qwen2.5-7B-Instruct**: 2048 tokens, 4096 tokens

#### 使用方法:

```bash
# 添加执行权限
chmod +x test_models_simple.sh

# 运行测试
./test_models_simple.sh
```

#### 输出:

- 每个模型的结果保存在 `results_<model>_<tokens>tokens.json`
- 自动生成对比总结表格
- 显示各类别(step1, step2&3)的准确率

---

## 📊 输出格式

### JSON结果文件结构:

```json
{
  "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
  "test_samples": 1273,
  "correct_count": 850,
  "accuracy": 0.668,
  "batch_size": 32,
  "max_new_tokens": 2048,
  "meta_accuracy": {
    "step1": 0.675,
    "step2&3": 0.660
  },
  "results": [
    {
      "id": 0,
      "question": "...",
      "correct_answer": "C",
      "predicted_answer": "C",
      "is_correct": true,
      "meta_info": "step1",
      "full_output": "..."
    }
  ]
}
```

---

## 🎯 性能建议

### Batch Size 设置:

| Max Tokens | 推荐Batch Size | GPU内存需求 |
|------------|----------------|-------------|
| 2048 | 32 | ~40GB |
| 4096 | 16 | ~40GB |
| 8192 | 8 | ~40GB |

### 速度估算:

- **Batch=32, Tokens=2048**: ~5-8分钟 (1273样本)
- **Batch=16, Tokens=4096**: ~10-15分钟 (1273样本)

---

## 🔧 自定义测试

### 修改Bash脚本测试的模型:

编辑 `test_models_simple.sh`:

```bash
# 添加/修改要测试的模型
MODELS=(
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "your-model/model-name"  # 添加你的模型
)

# 修改token设置
MAX_TOKENS=(2048 4096 8192)  # 添加8192测试
```

---

## 📈 查看结果

所有结果文件保存在当前目录:

```bash
# 查看所有结果文件
ls results_*.json

# 快速查看某个结果的准确率
python -c "import json; data=json.load(open('results_meta_llama_Meta_Llama_3_8B_Instruct_2048tokens.json')); print(f'Accuracy: {data[\"accuracy\"]:.3f}')"
```

---

## 💡 Tips

1. **GPU内存不足?** 降低 `--batch_size`
2. **想要更快?** 增加 `--batch_size` (如果GPU内存够)
3. **测试少量样本?** 使用 `--num_samples 100`
4. **调试模型输出?** 检查JSON中的 `full_output` 字段


