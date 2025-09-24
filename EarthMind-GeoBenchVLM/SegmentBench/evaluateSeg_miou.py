import os
import json
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate segmentation results (mIoU only)')
    parser.add_argument('--results_json', required=True, help='Path to segmentation_results.json')
    return parser.parse_args()

def load_mask(mask_path):
    mask = np.array(Image.open(mask_path).convert('L'))
    # Convert to binary: 1 for white/foreground, 0 for black/background
    return (mask > 127).astype(np.uint8)

def compute_iou(pred_mask, gt_mask):
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0  # Both empty = perfect, else 0
    return intersection / union

def main():
    args = parse_args()
    with open(args.results_json, 'r') as f:
        results = json.load(f)

    iou_scores = []
    valid_count = 0
    for entry in tqdm(results, desc='Evaluating'):
        pred_mask_path = entry.get('pred_mask_path')
        gt_mask_path = entry.get('gt_mask_path')
        if gt_mask_path and not gt_mask_path.startswith("GeoBench/"):
            gt_mask_path = os.path.join("GeoBench", gt_mask_path)
        if not pred_mask_path or not gt_mask_path or not os.path.exists(gt_mask_path) or not os.path.exists(pred_mask_path):
            continue  # Skip if missing files
        try:
            pred_mask = load_mask(pred_mask_path)
            gt_mask = load_mask(gt_mask_path)
            iou = compute_iou(pred_mask, gt_mask)
            iou_scores.append(iou)
            valid_count += 1
        except Exception as e:
            continue

    print(f"\nEvaluated {valid_count} mask pairs.")
    if iou_scores:
        miou = np.mean(iou_scores)
    else:
        miou = 0.0
    print(f"Mean IoU (mIoU): {miou:.4f}")

    out_json = os.path.splitext(args.results_json)[0] + '_eval.json'
    with open(out_json, 'w') as f:
        json.dump({'iou_scores': iou_scores, 'mIoU': miou}, f, indent=2)
    print(f"Per-sample IoUs and mIoU saved to {out_json}")

if __name__ == "__main__":
    main() 