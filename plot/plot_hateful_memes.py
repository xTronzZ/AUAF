import os,sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

def compute_calibration_metrics(acc_values, conf_values):
    # Simple correlation between accuracy and confidence
    if len(acc_values) < 2: return 0, 0
    
    # Expected: High Confidence -> High Accuracy
    # So we expect Positive Correlation
    # But if x-axis is "Uncertainty" (Low Conf), we expect Negative Correlation.
    # Here inputs are (Accuracy, Confidence).
    from scipy.stats import spearmanr, linregress
    
    spearman_corr, _ = spearmanr(conf_values, acc_values)
    
    reg = linregress(conf_values, acc_values)
    r_squared = reg.rvalue ** 2
    
    return spearman_corr, r_squared

title_dic = {
    'curve_similarity': 'Max Probability',
    'curve_consistency': 'Consistency',
    'curve_adversarial': 'Adversarial',
    'curve_joint': 'Joint (Avg)'
}

def load_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def plot_hateful_rejection(results, ax=None):
    # Curve: Removed % vs AUROC (or Accuracy)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    # Baseline
    baseline = results.get('baseline', {}).get('auroc', 0.5)
    ax.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.3f})')
    
    for key, data_points in results.items():
        if not key.startswith('curve_'): continue
        
        lbl = title_dic.get(key, key)
        
        x = [p['fraction_removed'] * 100 for p in data_points]
        y = [p['metric_val'] for p in data_points]
        
        # Calculate Area Under Rejection Curve (AURC-ish) for legend sorting?
        # Or just last value?
        # Let's just plot.
        ax.plot(x, y, label=lbl, linewidth=2)
        
    ax.set_xlabel('Rejection Rate (%)')
    ax.set_ylabel('AUROC')
    ax.set_title('Rejection Curve (Hateful Memes)')
    ax.set_xlim(0, 95) # Don't go to 100 as metrics become unstable
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend()
    return ax

def plot_hateful_calibration(results_path, ax=None):
    # For calibration, we need raw logits/probs usually to bin them.
    # But current JSON output of eval_hateful_memes1.py only stores rejection curve points.
    # To plot calibration, we need a different JSON content: {bin_avg_conf: ..., bin_acc: ...}
    # Currently eval_hateful_memes1.py DOES NOT SAVE THIS.
    # It only saves rejection curves.
    # So I will instruct the user or rewrite eval script to save it?
    # User said "Design a curve", implying I can make the plotting script do it 
    # IF the data is available.
    # Assuming we modify eval_hateful_memes1 to save calibration stats.
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('json_path', help="Path to hateful_memes_results_joint.json")
    parser.add_argument('--output', '-o', default='plots/hateful_memes_rejection.png', 
                        help="Output path for the plot (default: plots/hateful_memes_rejection.png)")
    args = parser.parse_args()
    
    data = load_data(args.json_path)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    plot_hateful_rejection(data, ax)
    
    plt.tight_layout()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {args.output}")
