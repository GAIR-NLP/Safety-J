#!/bin/bash

# Set language: "en" for English, "zh" for Chinese
LANG="en"

# Define model paths as variables
# You should set these paths according to your environment
EVALUATOR_MODEL_PATH="/path/to/evaluator/model"
QWEN_MODEL_PATH="/path/to/qwen/model"
# Note: QWEN_MODEL_PATH can be replaced with the path to any other chat-based large language model

# Set file paths based on language
if [ "$LANG" = "en" ]; then
    INPUT_FILE="../data/critique_level/en.json"
    REFERENCE_FILE="../data/critique_level/en.json"
    EVALUATOR_OUTPUT="../results/output_en.json"
    AIU_OUTPUT="../results/aius_en.json"
    PRECISION_FILE="../results/precision_analysis_en.txt"
    RECALL_FILE="../results/recall_analysis_en.txt"
else
    # Chinese file paths
    INPUT_FILE="../data/critique_level/zh.json"
    REFERENCE_FILE="../data/critique_level/zh.json"
    EVALUATOR_OUTPUT="../results/output_zh.json"
    AIU_OUTPUT="../results/aius_zh.json"
    PRECISION_FILE="../results/precision_analysis_zh.txt"
    RECALL_FILE="../results/recall_analysis_zh.txt"
fi

# Run evaluator script
echo "Running Evaluator Script..."
python critique_level_evaluator.py \
    --evaluator_model_path $EVALUATOR_MODEL_PATH \
    --lang $LANG \
    --input_file $INPUT_FILE \
    --output_file $EVALUATOR_OUTPUT

# Check if evaluator script ran successfully
if [ $? -ne 0 ]; then
    echo "Evaluator script failed. Exiting."
    exit 1
fi

# Run Qwen script
echo "Running Qwen Script..."
python critique_level_llm.py \
    --qwen_model_path $QWEN_MODEL_PATH \
    --lang $LANG \
    --input_file $EVALUATOR_OUTPUT \
    --reference_file $REFERENCE_FILE \
    --aiu_output_file $AIU_OUTPUT \
    --precision_file $PRECISION_FILE \
    --recall_file $RECALL_FILE

# Check if Qwen script ran successfully
if [ $? -ne 0 ]; then
    echo "Qwen script failed. Exiting."
    exit 1
fi

echo "Analysis completed successfully."