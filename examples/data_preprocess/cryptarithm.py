import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

def extract_solution(solution_str):
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[-1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[-1]

    answer_pattern = r'<answer>\s*(\d+)\s*\+\s*(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)\s*</answer>'
    matches = list(re.finditer(answer_pattern, solution_str))

    if matches:
        match = matches[-1]
        return (match.group(1), match.group(2), match.group(3), match.group(4))
    else:
        return None


def make_prefix(equation):
    question = f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The user highly desires that the assistant does not give up, produces high quality thoughts, and produces many thoughts. User: Cryptarithms are puzzles in which the digits of a mathematical equation—often a simple addition problem—are replaced by letters or symbols. Each letter uniquely represents a single digit, and the challenge is to determine the correct digit for each letter so that the arithmetic operation is valid. Please solve this Cryptarithm. {equation[0]} + {equation[1]} + {equation[2]} = {equation[3]}. Show your work in <think> </think> tags. And return the decrypted equation's numbers, not symbols, in <answer> </answer> tags, for example <answer> 1948 + 3756 + 9574 = 15278 </answer>. Assistant: Let me solve this accurately step by step. <think> """

    return question


if __name__ == '__main__':
    default_num_samples = 2000

    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/workspace/TinyZeroGRPO/data/cryptarithm')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--train_size', default=int(0.8 * default_num_samples + 1))
    parser.add_argument('--test_size', default=int(0.2 * default_num_samples + 1))

    args = parser.parse_args()

    full_train_dataset = datasets.load_dataset("json", data_files="/workspace/TinyZeroGRPO/data/cryptarithm/train.jsonl", split='train', field=None)
    full_test_dataset = datasets.load_dataset("json", data_files="/workspace/TinyZeroGRPO/data/cryptarithm/test.jsonl", split='train', field=None)
    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size

    assert len(full_train_dataset) >= TRAIN_SIZE
    assert len(full_test_dataset) >= TEST_SIZE

    train_dataset = full_train_dataset.select(range(TRAIN_SIZE))
    test_dataset = full_test_dataset.select(range(TEST_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            question = make_prefix(example['encrypted_equation'])

            data = {
                "data_source": "cryptarithm",
                "prompt": [{
                    "role": "user",
                    "content": question,
                }],
                "ability": "math",
                "reward_model": {
                    "encrypted_equation_str": example['encrypted_equation_str'], # For model string
                    "encrypted_equation": example['encrypted_equation'], # For reward model string array
                    "num_solutions": example['num_solutions']
                },
                "extra_info": {
                    'split': split,  # Train or test
                    'index': idx,
                }
            }
            return data
        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 
