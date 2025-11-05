import hydra
from omegaconf import DictConfig
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import re

from benchmark.mmlu_data_loader import load_mmlu_data

@hydra.main(config_path="config", config_name="mmlu_benchmark")
def main(cfg: DictConfig):
    """
    Runs the MMLU benchmark.
    """
    print("Loading MMLU data...")
    prompts, answers, subjects = load_mmlu_data(cfg.data_dir, cfg.num_shots)

    print(f"Loading model: {cfg.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, torch_dtype=torch.bfloat16, device_map="auto")

    correct = 0
    total = 0
    results = []

    for i in tqdm(range(len(prompts))):
        inputs = tokenizer(prompts[i], return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract the first letter from the prediction
        match = re.search(r'[A-D]', response.split("Answer:")[-1])
        prediction = match.group(0) if match else ''

        if prediction == answers[i]:
            correct += 1
        total += 1

        results.append({
            "subject": subjects[i],
            "prompt": prompts[i],
            "prediction": prediction,
            "answer": answers[i],
            "correct": prediction == answers[i]
        })

    accuracy = correct / total
    print(f"MMLU Accuracy: {accuracy}")

    # Save results to a CSV file
    df = pd.DataFrame(results)
    df.to_csv("mmlu_results.csv", index=False)

if __name__ == "__main__":
    main()
