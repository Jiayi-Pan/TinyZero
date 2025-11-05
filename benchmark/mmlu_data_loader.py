import os
import pandas as pd

def load_mmlu_data(data_dir, num_shots=5):
    """
    Loads the MMLU dataset and prepares it for a few-shot benchmark.
    """
    data = {}
    for split in ["dev", "test", "val"]:
        split_dir = os.path.join(data_dir, split)
        if not os.path.exists(split_dir):
            continue
        for filename in os.listdir(split_dir):
            if filename.endswith(".csv"):
                subject = filename.replace(f"_{split}.csv", "")
                if subject not in data:
                    data[subject] = {}
                df = pd.read_csv(os.path.join(split_dir, filename), header=None)
                data[subject][split] = df

    prompts = []
    answers = []
    subjects = []

    for subject in data:
        if "test" not in data[subject]:
            continue

        test_df = data[subject]["test"]
        dev_df = data[subject].get("dev")

        if dev_df is not None and num_shots > 0:
            dev_samples = dev_df.sample(min(num_shots, len(dev_df)), random_state=42)
        else:
            dev_samples = None

        for i, row in test_df.iterrows():
            prompt = ""
            if dev_samples is not None:
                for j, dev_row in dev_samples.iterrows():
                    prompt += f"Question: {dev_row[0]}\n"
                    prompt += f"A. {dev_row[1]}\n"
                    prompt += f"B. {dev_row[2]}\n"
                    prompt += f"C. {dev_row[3]}\n"
                    prompt += f"D. {dev_row[4]}\n"
                    prompt += f"Answer: {dev_row[5]}\n\n"

            prompt += f"Question: {row[0]}\n"
            prompt += f"A. {row[1]}\n"
            prompt += f"B. {row[2]}\n"
            prompt += f"C. {row[3]}\n"
            prompt += f"D. {row[4]}\n"
            prompt += "Answer:"

            prompts.append(prompt)
            answers.append(row[5])
            subjects.append(subject)

    return prompts, answers, subjects
