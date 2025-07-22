#!/bin/bash

# LMUnit Evaluation Script
# Usage: ./run_all_evaluations.sh <model_path> <tensor_parallel_size> [output_dir]
# Example: ./run_all_evaluations.sh /path/to/model 4 ./results

set -e  # Exit on any error

# Check if correct number of arguments provided
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: $0 <model_path> <tensor_parallel_size> [output_dir]"
    echo "Example: $0 /path/to/model 4 ./results"
    exit 1
fi

MODEL_PATH="$1"
TENSOR_PARALLEL_SIZE="$2"
OUTPUT_DIR="$3"

# Validate tensor parallel size is a number
if ! [[ "$TENSOR_PARALLEL_SIZE" =~ ^[0-9]+$ ]]; then
    echo "Error: tensor_parallel_size must be a positive integer"
    exit 1
fi

echo "=========================================="
echo "LMUnit Evaluation Suite"
echo "Model: $MODEL_PATH"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
if [ -n "$OUTPUT_DIR" ]; then
    echo "Output Directory: $OUTPUT_DIR"
fi
echo "=========================================="

# Change to eval directory
cd "$(dirname "$0")/../eval"

# Get list of available tasks
echo "Available tasks: infobench, flask, biggenbench, lfqa, rewardbenchv1"

# Run all evaluation tasks
TASKS=("infobench" "flask" "biggenbench" "lfqa" "rewardbenchv1")
for task in "${TASKS[@]}"; do
    echo ""
    echo "==========================================
Running evaluation for task: $task
=========================================="
    
    if [ -n "$OUTPUT_DIR" ]; then
        echo "Skipping evaluation for $task"
        python eval.py \
            --task "$task" \
            --model-path "$MODEL_PATH" \
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
            --output-dir "$OUTPUT_DIR"
    else
        python eval.py \
            --task "$task" \
            --model-path "$MODEL_PATH" \
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    fi
    
    echo "Completed: $task"
done

# Run reward bench evaluation
echo ""
echo "=========================================="
echo "Running RewardBench2 evaluation"
echo "=========================================="

if [ -n "$OUTPUT_DIR" ]; then
    echo "Skipping evaluation for rewardbench2"
    python reward_bench2.py \
        --model-path "$MODEL_PATH" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --output-dir "$OUTPUT_DIR" \
        --mode_unit_test "custom_per_subset"
else
    python reward_bench2.py \
        --model-path "$MODEL_PATH" \
        --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
        --mode_unit_test "custom_per_subset"
fi

echo "Completed: RewardBench2"

echo ""
echo "=========================================="
echo "All evaluations completed successfully!"
echo "Model: $MODEL_PATH"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
if [ -n "$OUTPUT_DIR" ]; then
    echo "Results saved to: $OUTPUT_DIR"
fi
echo "=========================================="

# Generate results summary table
if [ -n "$OUTPUT_DIR" ]; then
    echo ""
    echo "=========================================="
    echo "EVALUATION RESULTS SUMMARY"
    echo "=========================================="
    
    # Extract model name from path
    # Extract just the model name from the path for cleaner display
    MODEL_NAME=$MODEL_PATH
    
    # Initialize arrays for tasks and their performance values
    declare -a TASK_NAMES=()
    declare -a PERFORMANCE_VALUES=()
    
    # Collect all task results
    ALL_TASKS=("${TASKS[@]}" "rewardbenchv2")
    
    for task in "${ALL_TASKS[@]}"; do
        METRICS_FILE="$OUTPUT_DIR/$task/metrics/${task}_metrics.json"
        if [ -f "$METRICS_FILE" ]; then
            # Extract performance value using python
            PERFORMANCE=$(python3 -c "
import json
try:
    with open('$METRICS_FILE', 'r') as f:
        data = json.load(f)
    print(f\"{data.get('performance', 'N/A'):.4f}\" if isinstance(data.get('performance'), (int, float)) else 'N/A')
except:
    print('N/A')
")
            TASK_NAMES+=("$task")
            PERFORMANCE_VALUES+=("$PERFORMANCE")
        fi
    done
    
    # Generate markdown table
    echo ""
    echo "## Results Table (Markdown Format)"
    echo ""
    
    # Header row
    printf "| %-30s |" "Model"
    for task in "${TASK_NAMES[@]}"; do
        printf " %-12s |" "$task"
    done
    echo ""
    
    # Separator row
    printf "|%s|" "$(printf '%0.s-' {1..32})"
    for task in "${TASK_NAMES[@]}"; do
        printf "%s|" "$(printf '%0.s-' {1..14})"
    done
    echo ""
    
    # Data row
    printf "| %-30s |" "$MODEL_NAME"
    for performance in "${PERFORMANCE_VALUES[@]}"; do
        printf " %-12s |" "$performance"
    done
    echo ""
fi