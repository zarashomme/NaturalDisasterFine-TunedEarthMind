import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

# --- Argument parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description='EarthMind MCQ Benchmark')
    parser.add_argument('--qa_json', required=True, help='Path to qa.json (e.g., GeoBench/Single/qa.json)')
    parser.add_argument('--image_root', default = 'GeoBench', help='Root path to geobench folder (e.g., GeoBench')
    parser.add_argument('--model_path', default='sy1998/EarthMind-4B', help='EarthMind model path or repo')
    parser.add_argument('--results_dir', default='Results-earthmind', help='Directory to save results')
    parser.add_argument('--device', default='cuda:0', help='Device for inference')
    return parser.parse_args()

cfg = parse_args()

# --- Load Model and Tokenizer ---
print(f"Loading model from: {cfg.model_path} ...")
model = AutoModelForCausalLM.from_pretrained(
    cfg.model_path,
    torch_dtype=torch.bfloat16,
    device_map=cfg.device,
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
print("Model and tokenizer loaded successfully.")

# --- Load QA Data ---
with open(cfg.qa_json, 'r') as f:
    qa_data = json.load(f)

results_dict = {}

for i, question in enumerate(tqdm(qa_data, desc="Benchmarking")):
    image_path = os.path.join(cfg.image_root, question.get("image_path", ""))
    ground_truth = question.get("ground_truth", "")
    ground_truth_option = question.get("ground_truth_option", "")
    options_list = question.get("options_list", [])
    task = question.get("task", "")
    question_id = question.get("question_id", "")
    cls_description = question.get("cls_description", "")
    options_str = question.get("options", "")
    prompts = question.get("prompts", [])

    for prompt in prompts:
        # Build MCQ prompt (as in singleBench.py)
        choices = f"Options: {options_str}"
        full_prompt = (
            f"<image> For the given the Multiple Choice Question Answer below, analyze the question and answer strictly from one of the options below. "
            f"Strictly answer the choice only. No additional text. Provide only the letter (A., B., C., D. or E.) corresponding to the correct answer for the multiple-choice question given. "
            f"{cls_description}\n{prompt}\n{choices}"
        )
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {image_path} | {e}")
            continue
        try:
            # EarthMind's API: model.predict_forward(image=..., text=..., tokenizer=...)
            res = model.predict_forward(
                image=image,
                text=full_prompt,
                tokenizer=tokenizer,
            )
            response = res.get("prediction", "")
            # Extract only the first valid letter (A, B, C, D, or E)
            valid_choices = {"A", "B", "C", "D", "E"}
            predicted_answer = response[0] if response and response[0] in valid_choices else None
        except Exception as e:
            print(f"Error in prediction: {e}")
            predicted_answer = None

        key = f"{i}_{prompt[:16]}"  # Unique key per question+prompt
        if key not in results_dict:
            results_dict[key] = {
                "predicted_answers": [],
                "ground_truth": ground_truth,
                "questions": [prompt],
                "name_images": [image_path],
                "ground_truth_option": ground_truth_option,
                "options_list": options_list,
                "task": task,
                "question_id": question_id,
                "cls_description": cls_description,
                "options": options_str
            }
        results_dict[key]["predicted_answers"].append(predicted_answer)

# --- Save Results ---
os.makedirs(cfg.results_dir, exist_ok=True)
result_file = os.path.join(cfg.results_dir, f"evaluation_results_{os.path.basename(os.path.dirname(cfg.qa_json))}.json")
with open(result_file, "w") as f:
    json.dump(list(results_dict.values()), f, indent=4, default=str)
print(f"Results saved to {result_file}") 
