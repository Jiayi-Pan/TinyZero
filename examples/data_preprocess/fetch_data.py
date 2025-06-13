import os
import json
from datasets import Dataset
from typing import List, Dict
from tqdm import tqdm
import argparse


def load_context_from_file(
    context_file_path: str = "verl/utils/dataset/context.txt",
) -> str:
    """Load context content from a text file"""
    try:
        with open(context_file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(
            f"Warning: Context file {context_file_path} not found. Using fallback context."
        )
        return "Workfront API Context - Context file not found"


def load_workfront_data(json_file_path: str) -> List[Dict]:
    samples = []

    with open(json_file_path, "r") as f:
        data_list = json.load(f)

    for data in data_list:
        expected_response = data["expected_response"]

        api_call_steps = format_workfront_api_call(expected_response)

        samples.append(
            {
                "question": data["prompt"],
                "answer": api_call_steps,
                "template_type": "wf_api",
                "original_response": expected_response,
            }
        )

    return samples


def format_workfront_api_call(expected_response: Dict) -> str:
    return json.dumps(expected_response, indent=2)


def make_prefix(
    question, template_type="base", context_file_path="verl/utils/dataset/context.txt"
):
    """Create the prompt prefix for API tasks"""
    workfront_context = load_context_from_file(context_file_path)

    if template_type == "base":
        prefix = f"""You are a helpful AI assistant designed to convert natural language queries into structured JSON commands for querying the Workfront project management system. You use Workfront's custom object names and metadata to do the same using the context given below.

Your role is to interpret a user's natural language request, determine the correct object (objCode like TASK, PROJ, or USER), extract relevant fields (the attributes to display), and construct appropriate filters (conditions the data must satisfy). 


You will take the user's natural language prompt and finally give a structured JSON response after understanding context with the following structure and ALWAYS include just the final JSON with the correct json structure in <final_json> tags. The tags should always be called <final_json> and always inside tags use ```json``` to indicate the json structure. 
USE STRUCTURE EXACTLY LIKE BELOW 

Structure:
<final_json>
```json
{{
  "objCode": "TASK | PROJ | USER", // Choose based on what the user is asking about
  "fields": [],        // Include ALL relevant fields mentioned in the query      
  "filters": {{}} // Include ALL conditions mentioned in the query
}}
```
</final_json>
The JSON must be wrapped in triple backticks to indicate code formatting.

Heres are some examples:

Example 1:
User Prompt: What are all the tasks with high priority due next week?

Assistant:
<final_json>
```json
{{
  "objCode": "TASK",
  "fields": ["ID", "name", "priority", "plannedCompletionDate"],
  "filters": {{
        "priority": 3,
        "actualCompletionDate_Mod": "isnull",
        "plannedCompletionDate": "$$TODAYb+1w",
        "plannedCompletionDate_Mod": "between",
        "plannedCompletionDate_Range": "$$TODAYe+1w"
  }}
}}
```
</final_json>

Example 2:
User Prompt: Show me all projects that are currently on hold

Assistant:
<final_json>
```json
{{
  "objCode": "PROJ",
  "fields": ["ID", "name", "status", "plannedCompletionDate"],
  "filters": {{
        "status": "OHD"
  }}
}}
```
</final_json>

Example 3:
User Prompt: Find users with email addresses containing '@company.com'

Assistant:
<final_json>
```json
{{
  "objCode": "USER",
  "fields": ["ID", "name", "emailAddr", "username"],
  "filters": {{
        "emailAddr_Mod": "cicontains",
        "emailAddr": "@company.com"
  }}
}}
```
</final_json>

{workfront_context}

User: {question}
Assistant: I'll help you with defining the correct JSON object with the correct obj_code, fields, and filters.

<thinking>
I need to understand the user's request and determine:
1. Which object type (TASK, PROJ, or USER) they are asking about
2. What specific fields they need to see
3. What conditions (filters) they want to apply
4. How to structure the response to match their exact needs
"""
    elif template_type == "qwen-instruct":
        prefix = f"""<|im_start|>system
You are a helpful AI assistant designed to convert natural language queries into structured JSON commands for querying the Workfront project management system. You use Workfront's custom object names and metadata to do the same using the context given below.

Your role is to interpret a user's natural language request, determine the correct object (objCode like TASK, PROJ, or USER), extract relevant fields (the attributes to display), and construct appropriate filters (conditions the data must satisfy). 

IMPORTANT: You must analyze the user's query carefully and return a response that specifically matches their request. Do not return generic responses.

You will take the user's natural language prompt and give a structured JSON response with the following structure:

Structure:
```json
{{{{
  'objCode': 'TASK | PROJ | USER',  // Choose based on what the user is asking about
  'fields': [],  // Include ALL relevant fields mentioned in the query
  'filters': {{{{}}}}  // Include ALL conditions mentioned in the query
}}}}
```

The JSON must be wrapped in triple backticks to indicate code formatting.

Here are some examples:

Example 1:
User Prompt: What are all the tasks with high priority due next week?

Answer:
```json
{{{{
  "objCode": "TASK",
  "fields": ["ID", "name", "priority", "plannedCompletionDate"],
  "filters": {{{{
        "priority": 3,
        "actualCompletionDate_Mod": "isnull",
        "plannedCompletionDate": "$$TODAYb+1w",
        "plannedCompletionDate_Mod": "between",
        "plannedCompletionDate_Range": "$$TODAYe+1w"
  }}}}
}}}}
```

Example 2:
User Prompt: Show me all projects that are currently on hold

Answer:
```json
{{{{
  "objCode": "PROJ",
  "fields": ["ID", "name", "status", "plannedCompletionDate"],
  "filters": {{{{
        "status": "OHD"
  }}}}
}}}}
```

Example 3:
User Prompt: Find users with email addresses containing '@company.com'

Answer:
```json
{{{{
  "objCode": "USER",
  "fields": ["ID", "name", "emailAddr", "username"],
  "filters": {{{{
        "emailAddr_Mod": "cicontains",
        "emailAddr": "@company.com"
  }}}}
}}}}
```

{workfront_context}

<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
I'll help you with defining the correct JSON object with the correct obj_code, fields, and filters.

<thinking>
I need to understand the user's request and determine:
1. Which object type (TASK, PROJ, or USER) they are asking about
2. What specific fields they need to see
3. What conditions (filters) they want to apply
4. How to structure the response to match their exact needs
</thinking>

"""
    return prefix


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/workfront_api_tasks")
    parser.add_argument(
        "--json_file", required=True, help="Path to the JSON file with Workfront data"
    )
    parser.add_argument(
        "--train_split", type=float, default=0.8, help="Proportion of data for training"
    )
    parser.add_argument("--template_type", type=str, default="base")

    args = parser.parse_args()

    data_source = "wf_api"

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
            question = make_prefix(sample["question"], template_type=args.template_type)

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "api_planning",
                "reward_model": {"style": "rule", "ground_truth": sample["answer"]},
            }
            processed_data.append(data)
        return processed_data

    train_dataset = Dataset.from_list(process_samples(train_samples, "train"))
    test_dataset = Dataset.from_list(process_samples(test_samples, "test"))

    local_dir = os.path.expanduser(args.local_dir)
    os.makedirs(local_dir, exist_ok=True)

    print(f"Saving datasets to {local_dir}")
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    print(
        f"Generated {len(train_dataset)} training samples and {len(test_dataset)} test samples"
    )
    print("Sample data:")
    print("Question:", train_samples[0]["question"])
    print("Answer:", train_samples[0]["answer"])
