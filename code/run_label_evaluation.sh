#!/bin/bash

# Set the language: "en" for English, "zh" for Chinese
LANG="en"

# Set the paths
# MODEL_PATH="/path/to/your/model"
MODEL_PATH="/nas/shared/GAIR/yxzhen/internlm2_7b_chat_bilingual_version2_noaug_addition_filter_lr3e5_3epoch_dpo_with_version3_and_version4_noaug_addition_filter_lr5e8_1epoch_lr4e7_1epoch_lr4e8_1epoch"
INPUT_PATH="../data/label_level/BeaverTails.json"
OUTPUT_PATH="../results/label_level_output_${LANG}.json"

# Run the evaluator script
python ./label_level_evaluation.py \
    --model_path $MODEL_PATH \
    --input_path $INPUT_PATH \
    --output_path $OUTPUT_PATH \
    --lang $LANG

# Check if the script ran successfully
if [ $? -ne 0 ]; then
    echo "Evaluator script failed. Exiting."
    exit 1
fi

echo "Evaluation completed successfully. Results saved to $OUTPUT_PATH"