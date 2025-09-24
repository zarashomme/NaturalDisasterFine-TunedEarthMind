import os
import json
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description='Evaluate accuracy by task type from EarthMind results.')
parser.add_argument('--results_file', required=True, help='Path to the results JSON file (e.g., Results-earthmind/evaluation_results_Single.json)')
args = parser.parse_args()

with open(args.results_file, 'r') as f:
    results = json.load(f)

task_stats = defaultdict(lambda: {'total': 0, 'correct': 0})

for entry in results:
    task = entry.get('task', 'Unknown')
    gt = entry.get('ground_truth_option')
    preds = entry.get('predicted_answers', [])
    # Only consider the first prediction for accuracy
    pred = preds[0] if preds else None
    task_stats[task]['total'] += 1
    if pred is not None and gt is not None and pred == gt:
        task_stats[task]['correct'] += 1

print(f"{'Task':30} {'#Questions':>10} {'#Correct':>10} {'Accuracy':>10}")
print('-' * 65)
for task, stats in task_stats.items():
    total = stats['total']
    correct = stats['correct']
    accuracy = correct / total if total > 0 else 0.0
    print(f"{task:30} {total:10} {correct:10} {accuracy:10.3f}") 
