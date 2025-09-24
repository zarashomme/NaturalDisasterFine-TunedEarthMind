import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

# --- Argument parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description='EarthMind Captioning Benchmark')
    parser.add_argument('--qa_json', required=True, help='Path to qa.json (e.g., GeoBench/Captioning/qa.json)')
    parser.add_argument('--image_root', required=True, help='Root path to images (e.g., GeoBench/Captioning/images)')
    parser.add_argument('--model_path', default='sy1998/EarthMind-4B', help='EarthMind model path or repo')
    parser.add_argument('--results_dir', default='Results-earthmind-captioning', help='Directory to save results')
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
    task = question.get("task", "")
    question_id = question.get("question_id", "")
    prompts = question.get("prompts", [])

    for prompt in prompts:
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image: {image_path} | {e}")
            continue
        try:
            res = model.predict_forward(
                image=image,
                text="<image> " + prompt,
                tokenizer=tokenizer,
            )
            response = res.get("prediction", "")
        except Exception as e:
            print(f"Error in prediction: {e}")
            response = None

        key = f"{i}_{prompt[:16]}"  # Unique key per question+prompt
        if key not in results_dict:
            results_dict[key] = {
                "predicted_captions": [],
                "ground_truth": ground_truth,
                "prompts": [prompt],
                "name_images": [image_path],
                "task": task,
                "question_id": question_id
            }
        results_dict[key]["predicted_captions"].append(response)

# --- Save Results ---
os.makedirs(cfg.results_dir, exist_ok=True)
result_file = os.path.join(cfg.results_dir, f"evaluation_results_{os.path.basename(os.path.dirname(cfg.qa_json))}.json")
with open(result_file, "w") as f:
    json.dump(list(results_dict.values()), f, indent=4, default=str)
print(f"Results saved to {result_file}") 
