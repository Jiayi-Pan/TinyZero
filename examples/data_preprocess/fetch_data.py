import os
import json
from datasets import Dataset
from typing import List, Dict
from tqdm import tqdm
import argparse


def load_workfront_data(json_file_path: str) -> List[Dict]:
    samples = []
    
    with open(json_file_path, 'r') as f:
        data_list = json.load(f)
    
    for data in data_list:
        expected_response = data['expected_response']
        
        api_call_steps = format_workfront_api_call(expected_response)
        
        samples.append({
            "question": data['prompt'],
            "answer": api_call_steps,
            "template_type": "workfront_api",
            "original_response": expected_response
        })
    
    return samples


def format_workfront_api_call(expected_response: Dict) -> str:
    return json.dumps(expected_response, indent=2)


def make_prefix(question, template_type='base'):
    """Create the prompt prefix for API tasks"""
    if template_type == 'base':
        prefix = f"""A conversation between User and Assistant. The Assistant provides step-by-step API calls to complete user requests.

User: {question}
Assistant: I'll help you with that API task. Let me break it down step by step.

<thinking>
I need to determine the correct sequence of API calls to complete this request.
</thinking>

<answer>"""
    elif template_type == 'qwen-instruct':
        prefix = f"""<|im_start|>system
You are an API assistant that provides step-by-step API calls to complete user requests. Always provide clear, sequential API calls with proper parameters.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
I'll help you with that API task. Let me break it down step by step.

<thinking>
I need to determine the correct sequence of API calls to complete this request.
</thinking>

<answer>"""
    return prefix


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/workfront_api_tasks')
    parser.add_argument('--json_file', required=True, help='Path to the JSON file with Workfront data')
    parser.add_argument('--train_split', type=float, default=0.8, help='Proportion of data for training')
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'workfront_api_tasks'

    # Load the Workfront dataset
    print("Loading Workfront API dataset...")
    raw_samples = load_workfront_data(args.json_file)
    
    # Split into train and test
    total_samples = len(raw_samples)
    train_size = int(total_samples * args.train_split)
    
    train_samples = raw_samples[:train_size]
    test_samples = raw_samples[train_size:]

    def process_samples(samples, split):
        processed_data = []
        for idx, sample in enumerate(samples):
            question = make_prefix(sample['question'], template_type=args.template_type)
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "api_planning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": {
                        "correct_answer": sample['answer'],
                        "question": sample['question'],
                        "template_type": sample['template_type'],
                        "original_response": sample.get('original_response', {})
                    }
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                }
            }
            processed_data.append(data)
        return processed_data
    
    train_dataset = Dataset.from_list(process_samples(train_samples, 'train'))
    test_dataset = Dataset.from_list(process_samples(test_samples, 'test'))

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Saving datasets to {local_dir}")
    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
    
    print(f"Generated {len(train_dataset)} training samples and {len(test_dataset)} test samples")
    print("Sample data:")
    print("Question:", train_samples[0]['question'])
    print("Answer:", train_samples[0]['answer'])