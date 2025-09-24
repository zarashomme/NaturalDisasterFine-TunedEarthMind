import os
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import cv2

try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")

def parse_args():
    parser = argparse.ArgumentParser(description='Run segmentation for each prompt in qa.json')
    parser.add_argument('--qa_json', required=True, help='Path to qa.json')
    parser.add_argument('--image_root', required=True, help='Root path to images')
    parser.add_argument('--results_dir', default='results', help='Directory to save results')
    parser.add_argument('--model_path', default='sy1998/EarthMind-4B', help='Model path or repo')
    parser.add_argument('--device', default='cuda:0', help='Device for inference')
    return parser.parse_args()

cfg = parse_args()

# --- Prepare output directories ---
mask_dir = os.path.join(cfg.results_dir, 'masks')
overlay_dir = os.path.join(cfg.results_dir, 'overlays')
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(overlay_dir, exist_ok=True)
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

results = []

for i, entry in enumerate(tqdm(qa_data, desc='Running segmentation')):
    # Use image_path and ground_truth from JSON
    rel_image_path = entry.get('image_path')
    image_path = os.path.join(cfg.image_root, rel_image_path) if rel_image_path else None
    prompts = entry.get('prompts', [])
    question_id = entry.get('question_id', i)
    gt_mask_path = entry.get('ground_truth', None)  # Use 'ground_truth' as mask path
    image_name = entry.get('image_name') or (os.path.basename(rel_image_path) if rel_image_path else f'image_{i}.png')

    if not image_path or not os.path.exists(image_path):
        print(f"Image path missing or does not exist: {image_path}")
        continue

    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        continue

    for j, prompt in enumerate(prompts):
        if j>0:
            break
        # Run the actual model
        try:
            res = model.predict_forward(
                image=image,
                text="<image> Please " + prompt,
                tokenizer=tokenizer,
            )
            pred = res.get("prediction", "")
            # Extract the mask as in multiplerunSeg.py
            if '[SEG]' in pred and 'prediction_masks' in res:
                pred_masks = res['prediction_masks'][0]
                pred_mask = pred_masks[0]  # Should be a numpy array
                # Convert to binary mask (white=foreground, black=background)
                bin_mask = ((pred_mask > 0.5) * 255).astype(np.uint8)
                # Save binary mask
                mask_filename = f"{os.path.splitext(image_name)[0]}_q{question_id}_p{j}.png"
                mask_save_path = os.path.join(mask_dir, mask_filename)
                Image.fromarray(bin_mask).save(mask_save_path)

                # Visualization overlay
                overlay_save_path = os.path.join(overlay_dir, mask_filename)
                if Visualizer is not None and image_path:
                    img_cv = cv2.imread(image_path)
                    if img_cv is not None:
                        visualizer = Visualizer()
                        visualizer.set_image(img_cv)
                        visualizer.draw_binary_masks((bin_mask > 0).astype(np.bool_), colors='w', alphas=0.7)
                        visual_result = visualizer.get_image()
                        cv2.imwrite(overlay_save_path, visual_result)
                    else:
                        # If cv2.imread fails, just save the binary mask
                        Image.fromarray(bin_mask).save(overlay_save_path)
                else:
                    # If no visualizer, just save the binary mask as overlay
                    Image.fromarray(bin_mask).save(overlay_save_path)

                results.append({
                    'image_name': image_name,
                    'question_id': question_id,
                    'prompt': prompt,
                    'pred_mask_path': mask_save_path,
                    'overlay_path': overlay_save_path,
                    'gt_mask_path': gt_mask_path
                })
            else:
                print(f"No valid segmentation mask for {image_name} prompt {j}")
        except Exception as e:
            print(f"Error running model for {image_name} prompt {j}: {e}")
            continue

# --- Save results ---
results_file = os.path.join(cfg.results_dir, 'segmentation_results.json')
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Segmentation results saved to {results_file}") 
