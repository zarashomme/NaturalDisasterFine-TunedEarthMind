import os
import json
from PIL import Image
import numpy as np
import argparse
from tqdm import tqdm

print("Current working directory:", os.getcwd())

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate segmentation results')
    parser.add_argument('--results_json', required=True, help='Path to segmentation_results.json')
    parser.add_argument('--iou_thresholds', nargs='+', type=float, default=[0.5, 0.25], help='IoU thresholds for precision reporting')
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
        print(f"Raw pred_mask_path: {repr(pred_mask_path)}")
        print(f"Raw gt_mask_path: {repr(gt_mask_path)}")
        print(f"Checking existence: pred_mask_path={pred_mask_path} ({os.path.exists(pred_mask_path)}), gt_mask_path={gt_mask_path} ({os.path.exists(gt_mask_path)})")
        if not pred_mask_path or not gt_mask_path or not os.path.exists(gt_mask_path) or not os.path.exists(pred_mask_path):
            print(f"Skipping: pred_mask_path={pred_mask_path}, gt_mask_path={gt_mask_path}")
            continue
        try:
            pred_mask = load_mask(pred_mask_path)
            gt_mask = load_mask(gt_mask_path)
            print(f"Evaluating: pred_mask_path={pred_mask_path}, gt_mask_path={gt_mask_path}")
            print(f"  pred_mask shape: {pred_mask.shape}, unique: {np.unique(pred_mask)}")
            print(f"  gt_mask shape: {gt_mask.shape}, unique: {np.unique(gt_mask)}")
            if pred_mask.shape != gt_mask.shape:
                print("  WARNING: Mask shapes do not match!")
            iou = compute_iou(pred_mask, gt_mask)
            print(f"  IoU: {iou}")
            iou_scores.append(iou)
            valid_count += 1
        except Exception as e:
            print(f"Error evaluating {pred_mask_path} vs {gt_mask_path}: {e}")
            continue

    print(f"\nEvaluated {valid_count} mask pairs.")
    for thresh in args.iou_thresholds:
        precision = np.mean([iou >= thresh for iou in iou_scores]) if iou_scores else 0.0
        print(f"Precision @ IoU>{thresh:.2f}: {precision:.4f}")

    # Optionally, save per-sample IoUs
    out_json = os.path.splitext(args.results_json)[0] + '_eval.json'
    with open(out_json, 'w') as f:
        json.dump({'iou_scores': iou_scores, 'thresholds': args.iou_thresholds}, f, indent=2)
    print(f"Per-sample IoUs saved to {out_json}")

if __name__ == "__main__":
    main() 
