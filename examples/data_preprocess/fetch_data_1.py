import os
import json
from datasets import Dataset
from typing import List, Dict
from tqdm import tqdm
import argparse


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


def make_prefix(question, template_type="base"):
    """Create the prompt prefix for API tasks"""
    if template_type == "base":
        prefix = f"""You are a helpful AI assistant designed to convert natural language queries into structured JSON commands for querying the Workfront project management system. You use Workfront's custom object names and metadata to do the same using the context given below.

Your role is to interpret a user's natural language request, determine the correct object (objCode like TASK, PROJ, or USER), extract relevant fields (the attributes to display), and construct appropriate filters (conditions the data must satisfy). 

You must respond with a JSON object wrapped in <answer> tags and formatted with triple backticks.

Required JSON structure:
- objCode: Must be "TASK", "PROJ", or "USER" 
- fields: Array of field names to retrieve (always include "ID" and "name")
- filters: Object containing filter conditions

Your response format must be:
<answer>
```json
{{JSON_OBJECT_HERE}}
```
</answer>

 Workfront Object Context You Can Use

 Core Objects:
TASK: Represents individual tasks.
PROJ: Represents projects.
USER: Represents users (people in the org).

--- PROJECT METADATA ---
domain_knowledge:|
  Projects represent the main container of work that is to be completed. A Project may contain many different tasks or issues. Projects have both a plannedCompletionDate 
  which represents an estimate or planned completion date, and also a actualCompletionDate which represents the actual completion date of the project. If you are looking
  for when things actually completed use the actualCompletionDate, if you are wanting to know when something is planned to be completed use the plannedCompletionDate.
  Default Statuses include: Current, Dead, On Hold, Planning, Complete, Requested, Approved, Rejected, and Idea


--- TASK METADATA ---
domain_knowledge: |
  Tasks always belong to a project, so they always have the field projectID even when updating a task or making an assignment. 
  To update existing tasks, you must include the taskID in the request. 
  Tasks may also have subtasks, in which case they have a parent task (parentID) and a projectID.
  Tasks frequently have an assignment to a user, team, or role.
  Tasks can have both a plannedCompletionDate and a actualCompletionDate - if you want to know the actual date something was completed, use actualCompletionDate.
  Assignments can be added when creating an task by adding appropriate ids to the assignments array.
  A task name is always required.
  To set a Due Date, set the taskConstraint to MFO and the constraintDate to the desired date.  
  To set a Start Date, set the taskConstraint to MSO and the constraintDate to the desired date.
  Default statuses include: New, In Progress, and Complete
  Tasks include a taskNumber.  The task number shows the order the task appears in a list as shown to the user.  The user may referer to a task by its task number instead of its name.
  If a user refers to a task by its number insetead of its name, then it will need to be search by taskNumber instead of name to find the task id.  i.e. I want to rename task 3 to "New Task Name"

--- USER METADATA ---

domain_knowledge: |
  When searching for a user by name, be sure to check firstName, lastName, and name fields.  
  Use the OR operator because the name may not appear in all three fields.  
  Searching by email or username may also be used.

User: {question}
Assistant: I'll analyze your request and provide the appropriate Workfront API call structure."""
    elif template_type == "qwen-instruct":
        prefix = f"""<|im_start|>system
You are a helpful AI assistant designed to convert natural language queries into structured JSON commands for querying the Workfront project management system. You use Workfront's custom object names and metadata to do the same using the context given below.

Your role is to interpret a user's natural language request, determine the correct object (objCode like TASK, PROJ, or USER), extract relevant fields (the attributes to display), and construct appropriate filters (conditions the data must satisfy).

You must respond with a JSON object wrapped in <answer> tags and formatted with triple backticks.

Required JSON structure:
- objCode: Must be "TASK", "PROJ", or "USER" 
- fields: Array of field names to retrieve (always include "ID" and "name")
- filters: Object containing filter conditions

Your response format must be:
<answer>
```json
{{JSON_OBJECT_HERE}}
```
</answer>

 Workfront Object Context You Can Use

 Core Objects:
TASK: Represents individual tasks.
PROJ: Represents projects.
USER: Represents users (people in the org).

--- PROJECT METADATA ---
domain_knowledge:|
  Projects represent the main container of work that is to be completed. A Project may contain many different tasks or issues. Projects have both a plannedCompletionDate 
  which represents an estimate or planned completion date, and also a actualCompletionDate which represents the actual completion date of the project. If you are looking
  for when things actually completed use the actualCompletionDate, if you are wanting to know when something is planned to be completed use the plannedCompletionDate.
  Default Statuses include: Current, Dead, On Hold, Planning, Complete, Requested, Approved, Rejected, and Idea


--- TASK METADATA ---
domain_knowledge: |
  Tasks always belong to a project, so they always have the field projectID even when updating a task or making an assignment. 
  To update existing tasks, you must include the taskID in the request. 
  Tasks may also have subtasks, in which case they have a parent task (parentID) and a projectID.
  Tasks frequently have an assignment to a user, team, or role.
  Tasks can have both a plannedCompletionDate and a actualCompletionDate - if you want to know the actual date something was completed, use actualCompletionDate.
  Assignments can be added when creating an task by adding appropriate ids to the assignments array.
  A task name is always required.
  To set a Due Date, set the taskConstraint to MFO and the constraintDate to the desired date.  
  To set a Start Date, set the taskConstraint to MSO and the constraintDate to the desired date.
  Default statuses include: New, In Progress, and Complete
  Tasks include a taskNumber.  The task number shows the order the task appears in a list as shown to the user.  The user may referer to a task by its task number instead of its name.
  If a user refers to a task by its number insetead of its name, then it will need to be search by taskNumber instead of name to find the task id.  i.e. I want to rename task 3 to "New Task Name"

--- USER METADATA ---

domain_knowledge: |
  When searching for a user by name, be sure to check firstName, lastName, and name fields.  
  Use the OR operator because the name may not appear in all three fields.  
  Searching by email or username may also be used.

  
<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
I'll analyze your request and provide the appropriate Workfront API call structure."""
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
