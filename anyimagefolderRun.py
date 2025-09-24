import os
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import cv2
import argparse


try:
    from mmengine.visualization import Visualizer
except ImportError:
    Visualizer = None
    print("Warning: mmengine is not installed, visualization is disabled.")


def parse_args():
    parser = argparse.ArgumentParser(description='Video Reasoning Segmentation')
    parser.add_argument('--image_folder', default="preimages", help='Path to image file')
    parser.add_argument('--model_path', default="sy1998/EarthMind-4B")
    parser.add_argument('--results', default="results", help='The dir to save results.')
    parser.add_argument('--text', type=str, default="<image>Please segment the left chimney.")
    # parser.add_argument('--select', type=int, default=-1)
    args = parser.parse_args()
    return args


cfg = parse_args()
# --- Configuration ---
model_path = cfg.model_path # Use your local path or "sy1998/EarthMind-4B"
device = "cuda:0" # Or "cpu" if no GPU


# --- 1. Load Model and Tokenizer ---
print(f"Loading model from: {model_path}...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    # CHANGE THIS LINE: Set the dtype to BFloat16
    torch_dtype=torch.bfloat16,
    device_map=device,
    trust_remote_code=True,
    # token="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" # Uncomment if a token is required
)


# Load the tokenizer separately
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


print("Model and tokenizer loaded successfully.")






image_files = []
image_paths = []
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff",".webp",".jp2"}
for filename in sorted(list(os.listdir(cfg.image_folder))):
    if os.path.splitext(filename)[1].lower() in image_extensions:
        image_files.append(filename)
        image_paths.append(os.path.join(cfg.image_folder, filename))




image_list = []
for img_path in image_paths:
    img = Image.open(img_path).convert('RGB')
    image_list.append(img)


result = []
for i in range(len(image_list)):
    print("##########", image_list[i])
    res = model.predict_forward(
            image=image_list[i],
            text=cfg.text,
            tokenizer=tokenizer,
    )
    result.append(res)
print(f"The input is:\n{cfg.text}")

def visualize(pred_mask, image_path, result_path):
    visualizer = Visualizer()
    img = cv2.imread(image_path)
    visualizer.set_image(img)
    visualizer.draw_binary_masks(pred_mask, colors='r', alphas=0.4)
    visual_result = visualizer.get_image()

    output_path = os.path.join(result_path, os.path.basename(image_path))
    cv2.imwrite(output_path, visual_result)

for i in range(len(result)):
    print("\nPrediction for :" + image_files[i])
    pred = result[i].get("prediction", "No prediction returned or incorrect key.")
    print(pred)
    
    if '[SEG]' in pred and Visualizer is not None:
        res = result[i]
        pred_masks = res['prediction_masks'][0]
        pred_mask = pred_masks[0]
        os.makedirs(cfg.results, exist_ok=True)
        visualize(pred_mask, image_paths[i],cfg.results)

