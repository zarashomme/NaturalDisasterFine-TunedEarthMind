import json
import os
import argparse
from bert_score import score
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='BERTScore Evaluation for Captioning Results')
    parser.add_argument('--results_file', required=True, help='Path to evaluation results JSON file')
    parser.add_argument('--output_dir', default='bertscore_results', help='Directory to save BERTScore results')
    parser.add_argument('--lang', default='en', help='Language for BERTScore evaluation')
    return parser.parse_args()

def extract_captions_and_references(results_file):
    """Extract predicted captions and ground truths from results file."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    predicted_captions = []
    ground_truths = []
    
    for result in results:
        # Get the first predicted caption (or handle multiple predictions)
        predicted_captions_list = result.get('predicted_captions', [])
        if predicted_captions_list:
            # Take the first prediction if multiple exist
            predicted_caption = predicted_captions_list[0]
            if predicted_caption is not None:  # Skip None predictions
                predicted_captions.append(predicted_caption)
                ground_truths.append(result.get('ground_truth', ''))
    
    return predicted_captions, ground_truths

def write_caption_files(predicted_captions, ground_truths, output_dir):
    """Write captions to separate files for BERTScore evaluation."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Write predicted captions
    with open(os.path.join(output_dir, 'pred.txt'), 'w', encoding='utf-8') as f:
        for caption in predicted_captions:
            f.write(caption + '\n')
    
    # Write ground truth captions
    with open(os.path.join(output_dir, 'refs.txt'), 'w', encoding='utf-8') as f:
        for ref in ground_truths:
            f.write(ref + '\n')
    
    print(f"Written {len(predicted_captions)} captions to {output_dir}/")
    print(f"Predictions: {output_dir}/pred.txt")
    print(f"References: {output_dir}/refs.txt")

def run_bertscore_evaluation(predicted_captions, ground_truths, lang='en'):
    """Run BERTScore evaluation."""
    print(f"Running BERTScore evaluation on {len(predicted_captions)} captions...")
    
    try:
        (P, R, F), hashname = score(predicted_captions, ground_truths, lang=lang, return_hash=True)
        
        # Calculate mean scores
        p_mean = P.mean().item()
        r_mean = R.mean().item()
        f_mean = F.mean().item()
        
        # Calculate standard deviations
        p_std = P.std().item()
        r_std = R.std().item()
        f_std = F.std().item()
        
        print(f"\nBERTScore Results:")
        print(f"Hash: {hashname}")
        print(f"Precision: {p_mean:.6f} ± {p_std:.6f}")
        print(f"Recall: {r_mean:.6f} ± {r_std:.6f}")
        print(f"F1: {f_mean:.6f} ± {f_std:.6f}")
        
        return {
            'precision': p_mean,
            'recall': r_mean,
            'f1': f_mean,
            'precision_std': p_std,
            'recall_std': r_std,
            'f1_std': f_std,
            'hash': hashname,
            'num_samples': len(predicted_captions),
            'P': P,
            'R': R,
            'F': F
        }
        
    except Exception as e:
        print(f"Error running BERTScore evaluation: {e}")
        return None

def save_bertscore_results(results, output_dir):
    """Save BERTScore results to JSON file."""
    # Create a JSON-serializable version of results
    json_results = {
        'precision': results['precision'],
        'recall': results['recall'],
        'f1': results['f1'],
        'precision_std': results['precision_std'],
        'recall_std': results['recall_std'],
        'f1_std': results['f1_std'],
        'hash': results['hash'],
        'num_samples': results['num_samples']
    }
    
    results_file = os.path.join(output_dir, 'bertscore_results.json')
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=4)
    print(f"BERTScore results saved to: {results_file}")

def main():
    args = parse_args()
    
    # Check if results file exists
    if not os.path.exists(args.results_file):
        print(f"Error: Results file {args.results_file} not found!")
        return
    
    print(f"Loading results from: {args.results_file}")
    
    # Extract captions and references
    predicted_captions, ground_truths = extract_captions_and_references(args.results_file)
    
    if not predicted_captions:
        print("No valid predictions found in results file!")
        return
    
    print(f"Found {len(predicted_captions)} valid predictions")
    
    # Write caption files
    write_caption_files(predicted_captions, ground_truths, args.output_dir)
    
    # Run BERTScore evaluation
    bertscore_results = run_bertscore_evaluation(predicted_captions, ground_truths, args.lang)
    
    if bertscore_results:
        # Save results
        save_bertscore_results(bertscore_results, args.output_dir)
        
        # Also save individual scores for detailed analysis
        scores_file = os.path.join(args.output_dir, 'individual_scores.json')
        individual_scores = {
            'precision_scores': bertscore_results['P'].tolist(),
            'recall_scores': bertscore_results['R'].tolist(),
            'f1_scores': bertscore_results['F'].tolist(),
            'predictions': predicted_captions,
            'references': ground_truths
        }
        with open(scores_file, 'w') as f:
            json.dump(individual_scores, f, indent=4)
        print(f"Individual scores saved to: {scores_file}")

if __name__ == "__main__":
    main() 
