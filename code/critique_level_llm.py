import os
import json
import re
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Set environment variables
os.environ['CUDA_VISIBLE_DEVICES'] = "4,5,6,7"
os.environ['RAY_memory_monitor_refresh_ms'] = '0'

# Constants
SYSTEM_PROMPT = "You are a helpful assistant.\n"
IM_START, IM_END = "<|im_start|>", "<|im_end|>"

def load_data(file_path):
    print(f"Loading data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    print(f"Loaded {len(data)} items")
    return data

def save_data(data, file_path):
    print(f"Saving data to {file_path}")
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} items")

def create_aiu_prompt(cleaned_critique, lang='en'):
    file_path = '../prompts/prompt_for_extract_aius_en.txt' if lang == 'en' else '../prompts/prompt_for_extract_aius_zh.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return f"{SYSTEM_PROMPT}{IM_START}user\n{content.replace('{input}', cleaned_critique)}{IM_END}\n{IM_START}assistant\n"

def create_recall_prompt(cleaned_critique, claim, lang='en'):
    file_path = '../prompts/recall_en.txt' if lang == 'en' else '../prompts/recall_zh.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    content = content.replace("{text}", cleaned_critique)
    content = content.replace("{claim}", claim)
    return f"{SYSTEM_PROMPT}{IM_START}user\n{content}{IM_END}\n{IM_START}assistant\n"

def create_precision_prompt(prompt, response, claim, lang='en'):
    file_path = '../prompts/precision_en.txt' if lang == 'en' else '../prompts/precision_zh.txt'
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    content = content.replace("{text}", f'A：{prompt}\nB：{response}')
    content = content.replace("{claim}", claim)
    return f"{SYSTEM_PROMPT}{IM_START}user\n{content}{IM_END}\n{IM_START}assistant\n"

def extract_aius(llm, data, sampling_params, lang):
    print("Extracting AIUs...")
    contents = [create_aiu_prompt(item['cleaned_critique'], lang) for item in data]
    outputs = llm.generate(contents, sampling_params)
    
    for item, output in tqdm(zip(data, outputs), total=len(data)):
        aius_text = output.outputs[0].text.strip()
        aius = [re.sub(r'^\d+\.\s*', '', aiu) for aiu in aius_text.split('\n')]
        item['aius'] = aius

    print(f"Extracted AIUs for {len(data)} items")
    return data

def calculate_metrics(llm, data, reference_data, sampling_params, filename1, filename2, lang):
    
    recall_contents = []
    precision_contents = []
    
    for item1, item in tqdm(zip(data, reference_data), total=len(data), desc="Preparing prompts"):
        for claim in item['aius']:
            recall_contents.append(create_recall_prompt(item1['cleaned_critique'], claim, lang))
        for claim in item1['aius']:
            precision_contents.append(create_precision_prompt(item1['prompt'], item1['response'], claim, lang))

    print(f"Generating outputs for {len(recall_contents)} recall prompts")
    recall_outputs = llm.generate(recall_contents, sampling_params)
    print(f"Generating outputs for {len(precision_contents)} precision prompts")
    precision_outputs = llm.generate(precision_contents, sampling_params)

    print(f"Number of recall outputs: {len(recall_outputs)}")
    print(f"Number of precision outputs: {len(precision_outputs)}")
    print("Calculating metrics...")

    recalls = []
    precisions = []
    count_right_recall = 0
    count_right_precision = 0

    total_recall_aius = sum(len(item['aius']) for item in reference_data)
    total_precision_aius = sum(len(item['aius']) for item in data)

    print(f"Total recall AIUs: {total_recall_aius}")
    print(f"Total precision AIUs: {total_precision_aius}")

    recall_index = 0
    precision_index = 0

    for idx, (item1, item) in enumerate(tqdm(zip(data, reference_data), total=len(data), desc="Processing outputs")):
        count_right_recall_item = 0
        count_right_precision_item = 0

        for i in range(len(item['aius'])):
            if recall_index < len(recall_outputs):
                output = recall_outputs[recall_index].outputs[0].text
                recall_index += 1
                match = re.search(r'<答案开始>(.*?)<答案结束>', output, re.DOTALL) if lang == 'zh' else re.search(r'<Answer Start>(.*?)<Answer End>', output, re.DOTALL)
                if match and ("正确" in match.group(1) if lang == 'zh' else "Correct" in match.group(1)):
                    count_right_recall_item += 1
                    count_right_recall += 1
                
                with open(filename2, 'a', encoding='utf-8') as file_error:
                    file_error.write(f'===================================={idx}=====================================\n')
                    file_error.write(f'reference aiu:\n{item["aius"][i]}\n')
                    file_error.write(f'hypothesis critique:\n{item1["cleaned_critique"]}\n')
                    file_error.write(f'analysis:\n{output}\n')
            else:
                print(f"Warning: Recall index out of range at item {idx}, AIU {i}")

        for i in range(len(item1['aius'])):
            if precision_index < len(precision_outputs):
                output = precision_outputs[precision_index].outputs[0].text
                precision_index += 1
                match = re.search(r'<答案开始>(.*?)<答案结束>', output, re.DOTALL) if lang == 'zh' else re.search(r'<Answer Start>(.*?)<Answer End>', output, re.DOTALL)
                if match and ("正确" in match.group(1) if lang == 'zh' else "Correct" in match.group(1)):
                    count_right_precision_item += 1
                    count_right_precision += 1
                
                with open(filename1, 'a', encoding='utf-8') as file_error:
                    file_error.write(f'===================================={idx}=====================================\n')
                    file_error.write(f'query and response:\n{item1["prompt"]}\nB：{item1["response"]}\n')
                    file_error.write(f'aiu:\n{item1["aius"][i]}\n')
                    file_error.write(f'analysis\n{output}\n')
            else:
                print(f"Warning: Precision index out of range at item {idx}, AIU {i}")

        recalls.append(count_right_recall_item / len(item['aius']) if len(item['aius']) > 0 else 0)
        precisions.append(count_right_precision_item / len(item1['aius']) if len(item1['aius']) > 0 else 0)

    recall = sum(recalls) / len(recalls) if recalls else 0
    precision = sum(precisions) / len(precisions) if precisions else 0
    recall_all = count_right_recall / total_recall_aius if total_recall_aius > 0 else 0
    precision_all = count_right_precision / total_precision_aius if total_precision_aius > 0 else 0

    f1s = [2 * p * r / (p + r) if p + r > 0 else 0 for p, r in zip(precisions, recalls)]
    f1_score = sum(f1s) / len(f1s) if f1s else 0
    f1_score_all = 2 * recall_all * precision_all / (recall_all + precision_all) if recall_all + precision_all > 0 else 0

    metrics = {
        'recall(macro)': recall,
        'recall(micro)': recall_all,
        'precision(macro)': precision,
        'precision(micro)': precision_all,
        'f1_score(macro)': f1_score,
        'f1_score(micro)': f1_score_all
    }

    print("Metrics calculation completed")
    return metrics

def write_metrics(metrics, filename):
    print(f"Writing metrics to {filename}")
    with open(filename, 'a', encoding='utf-8') as file:
        for key, value in metrics.items():
            file.write(f'{key}: {value}\n')
    print("Metrics written successfully")

def parse_arguments():
    parser = argparse.ArgumentParser(description="AIU Extraction and Metrics Calculation with Qwen Model")
    parser.add_argument("--qwen_model_path", required=True, help="Path to the Qwen model")
    parser.add_argument("--lang", required=True, choices=['zh', 'en'], help="Language to process")
    parser.add_argument("--input_file", required=True, help="Path to the input JSON file (output from evaluator)")
    parser.add_argument("--reference_file", required=True, help="Path to the reference JSON file")
    parser.add_argument("--aiu_output_file", required=True, help="Path to save the AIU output JSON file")
    parser.add_argument("--precision_file", required=True, help="Path to save precision analysis")
    parser.add_argument("--recall_file", required=True, help="Path to save recall analysis")
    return parser.parse_args()

def main():
    args = parse_arguments()
    print(f"Qwen model path: {args.qwen_model_path}")
    print(f"Language: {args.lang}")
    print(f"Input file: {args.input_file}")

    sampling_params = SamplingParams(max_tokens=4096, temperature=0, stop=['<|endoftext|>', '<|im_end|>'])
    
    print("Initializing Qwen LLM")
    qwen_llm = LLM(model=args.qwen_model_path, trust_remote_code=True,tensor_parallel_size=4)
    print("Qwen LLM initialized")

    data = load_data(args.input_file)
    
    # AIU extraction
    aiu_data = extract_aius(qwen_llm, data, sampling_params, args.lang)
    save_data(aiu_data, args.aiu_output_file)

    # Metrics calculation
    reference_data = load_data(args.reference_file)
    metrics = calculate_metrics(qwen_llm, aiu_data, reference_data, sampling_params, args.precision_file, args.recall_file, args.lang)
    
    print("Results:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    write_metrics(metrics, args.precision_file)
    write_metrics(metrics, args.recall_file)

    print("AIU extraction and metrics calculation completed.")

if __name__ == "__main__":
    main()