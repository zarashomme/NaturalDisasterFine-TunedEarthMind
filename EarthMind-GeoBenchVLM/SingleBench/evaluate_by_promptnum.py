import os
import json
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser(description='Evaluate accuracy by prompt number and task type from EarthMind results.')
parser.add_argument('--results_file', required=True, help='Path to the results JSON file (e.g., Results-earthmind/evaluation_results_Single.json)')
args = parser.parse_args()

with open(args.results_file, 'r') as f:
    results = json.load(f)

# Track prompt order for each (image, qid) pair
prompt_order = defaultdict(list)
# Store stats: {prompt_idx: {task: {'total': int, 'correct': int}}}
prompt_task_stats = defaultdict(lambda: defaultdict(lambda: {'total': 0, 'correct': 0}))

for entry in results:
    image = entry.get('name_images', [None])[0]
    qid = entry.get('question_id', None)
    task = entry.get('task', 'Unknown')
    gt = entry.get('ground_truth_option')
    preds = entry.get('predicted_answers', [])
    pred = preds[0] if preds else None
    question = entry.get('questions', [''])[0]

    # Use a tuple of (image, qid) to group prompts
    group_key = (image, qid)
    prompt_list = prompt_order[group_key]
    if question not in prompt_list:
        prompt_list.append(question)
    prompt_idx = prompt_list.index(question)

    prompt_task_stats[prompt_idx][task]['total'] += 1
    if pred is not None and gt is not None and pred == gt:
        prompt_task_stats[prompt_idx][task]['correct'] += 1

# Print a table for each prompt index
for prompt_idx in sorted(prompt_task_stats.keys()):
    print(f"\nPrompt #{prompt_idx+1}")
    print(f"{'Task':30} {'#Questions':>10} {'#Correct':>10} {'Accuracy':>10}")
    print('-' * 65)
    for task, stats in prompt_task_stats[prompt_idx].items():
        total = stats['total']
        correct = stats['correct']
        accuracy = correct / total if total > 0 else 0.0
        print(f"{task:30} {total:10} {correct:10} {accuracy:10.3f}") 