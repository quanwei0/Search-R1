#!/bin/bash
# Simple script to test different models on medical QA dataset

echo "🚀 Starting model comparison tests"
echo "=================================="
echo ""

# Define test file (can be changed via command line argument)
TEST_FILE="/home/li003968/Search-R1/baseline_test/mmlu_pro_health_test.jsonl"

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Error: Test file not found: $TEST_FILE"
    exit 1
fi

echo "📁 Test file: $TEST_FILE"
echo "📊 Total samples: $(wc -l < $TEST_FILE)"
echo ""

# Define models to test
MODELS=(
    "che111/AlphaMed-8B-instruct-rl"
    "meta-llama/Meta-Llama-3-8B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
)

# Define max_new_tokens settings
MAX_TOKENS=(2048)

# Test each combination
for model in "${MODELS[@]}"; do
    for tokens in "${MAX_TOKENS[@]}"; do
        echo ""
        echo "📊 Testing: $model with max_tokens=$tokens"
        echo "-------------------------------------------"
        
        # Adjust batch size based on max_tokens
        if [ $tokens -eq 2048 ]; then
            BATCH_SIZE=128
        else
            BATCH_SIZE=32
        fi
        
        # Run test
        python test_alphamed_batch.py \
            --model_id "$model" \
            --batch_size $BATCH_SIZE \
            --max_new_tokens $tokens \
            --cuda_devices "0,1,2,3" \
            --test_file "$TEST_FILE"
        
        if [ $? -eq 0 ]; then
            echo "✅ Test completed successfully!"
        else
            echo "❌ Test failed!"
        fi
        
        echo ""
        sleep 5  # Short pause between tests
    done
done

echo ""
echo "=================================="
echo "🎉 All tests completed!"
echo ""

# Generate summary
echo "📊 Generating summary..."
python3 - <<'EOF'
import json
import glob

print("\n" + "="*80)
print("📊 MODEL COMPARISON SUMMARY")
print("="*80 + "\n")

# Find all result files
result_files = glob.glob("./results_*.json")

if not result_files:
    print("❌ No result files found!")
    exit(1)

# Collect and display results
results = []
for result_file in sorted(result_files):
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        model_name = data['model_id'].split('/')[-1]
        results.append({
            'model': model_name,
            'max_tokens': data['max_new_tokens'],
            'accuracy': data['accuracy'],
            'correct': data['correct_count'],
            'total': data['test_samples'],
            'meta_accuracy': data.get('meta_accuracy', {})
        })
    except Exception as e:
        print(f"⚠️  Error reading {result_file}: {e}")

# Sort by model and max_tokens
results.sort(key=lambda x: (x['model'], x['max_tokens']))

# Print summary table
print(f"{'Model':<35} {'Max Tokens':<12} {'Accuracy':<12} {'Correct/Total':<15}")
print("-" * 80)

for r in results:
    model = r['model']
    max_tokens = r['max_tokens']
    accuracy = f"{r['accuracy']:.3f}"
    correct_total = f"{r['correct']}/{r['total']}"
    print(f"{model:<35} {max_tokens:<12} {accuracy:<12} {correct_total:<15}")

# Print detailed breakdown
print("\n" + "="*80)
print("📈 DETAILED ACCURACY BY CATEGORY")
print("="*80 + "\n")

for r in results:
    print(f"📌 {r['model']} (max_tokens={r['max_tokens']})")
    print(f"   Overall: {r['accuracy']:.3f}")
    if r['meta_accuracy']:
        for category, acc in r['meta_accuracy'].items():
            print(f"   - {category}: {acc:.3f}")
    print()

print("="*80)
print("✅ Summary complete!")
print("="*80 + "\n")
EOF

echo ""
echo "📁 All result files are saved in the current directory"
echo ""

