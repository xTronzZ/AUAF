import os,sys,glob
import json
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline
try:
    from metrics import compute_calibration_metrics
except ImportError:
    # Fallback or assume script is run from correct directory
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from metrics import compute_calibration_metrics

data = {}
results_table = {}

# Support wildcards in arguments (e.g. for Windows)
files = []
smooth = False
for arg in sys.argv[1:]:
    if arg == '--smooth':
        smooth = True
        continue
    expanded = glob.glob(arg)
    if expanded:
        files.extend(expanded)
    else:
        files.append(arg)

if not files:
    print("No files provided. Usage: python plot_calibration.py results/metrics_calibration_*.json")
    sys.exit(1)

for fname in files:
    with open(fname) as f:
        stem = os.path.splitext(os.path.basename(fname))[0]
        parts = stem.split('_')
        if len(parts) >= 4 and parts[0] == 'metrics' and parts[1] == 'calibration':
            k = '_'.join(parts[3:]) # Skip dataset name
        elif len(parts) >= 3 and parts[0] == 'metrics' and parts[1] == 'calibration':
            k = '_'.join(parts[2:])
        else:
            k = stem

        print(f"Processing: {k}")
        data[k] = json.load(f)
        results_table[k] = {}

# Increase figure height slightly to accommodate potential legend placement
# figsize=(width, height)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

# Iterate and plot
# Filter out single_top1consistency specifically
methods = sorted(data.keys())
methods = [m for m in methods if m != "single_top1consistency"]

for method in methods:
  for model in data[method].keys():
    # We iterate over the two subplots (Image Retrieval, Text Retrieval usually)
    for ax, k in zip(axes.ravel(), data[method][model].keys()):
        # Extract data
        # Data format seems to be list of tuples/lists: [lower, upper, recall] ??? 
        # Original code: y = np.array([c[2] for c in data[method][model][k]])[::-1]
        raw_data = data[method][model][k]
        if not raw_data:
            continue
            
        y = np.array([c[2] for c in raw_data])[::-1]
        x = np.arange(y.shape[0]) + 1
        
        # Plot with markers
        x_plot = x
        y_plot = y
        marker = 'o'
        
        if smooth and len(x_plot) > 3:
            try:
                # Smoothing using B-spline
                x_new = np.linspace(x_plot.min(), x_plot.max(), 300)
                spl = make_interp_spline(x_plot, y_plot, k=3)
                y_smooth = spl(x_new)
                x_plot = x_new
                y_plot = y_smooth
                marker = None
            except Exception as e:
                print(f"Smoothing failed for {method}-{model}-{k}: {e}")
        
        ax.plot(x_plot, y_plot, label=method, marker=marker, linewidth=2, markersize=6)
        
        ax.set_title(k, fontsize=14)
        ax.set_xticks(x)
        ax.set_ylabel('Recall@1', fontsize=12)
        ax.set_xlabel('Uncertainty Level (Bin)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

        # Compute metrics
        spearman_corr, r_squared = compute_calibration_metrics(y, x)
        # print(method, model, k, spearman_corr, r_squared)
        
        if method not in results_table:
            results_table[method] = {}
        # Store metrics: spearman, r2, combination
        results_table[method][k] = (float(spearman_corr), float(r_squared), float(-spearman_corr * r_squared))

# Print Metrics Table
print("\n" + "="*80)
print(f"{'Method':<30} | {'Text R@1 (S/R2/Comb)':<20} | {'Image R@1 (S/R2/Comb)':<20}")
print("-" * 80)

for method in results_table.keys():
    # Keys vary based on JSON content, usually 'text_retrieval_recall@1' and 'image_retrieval_recall@1'
    # Adapted from original print loop to be safer
    try:
        i2t_key = 'text_retrieval_recall@1'
        t2i_key = 'image_retrieval_recall@1'
        
        if i2t_key in results_table[method] and t2i_key in results_table[method]:
            i2t = results_table[method][i2t_key]
            t2i = results_table[method][t2i_key]
            # print(f'{method} & {i2t[0]:.2f} & {i2t[1]:.2f} & {i2t[2]:.2f} & {t2i[0]:.2f} & {t2i[1]:.2f} & {t2i[2]:.2f} \\\\')
            print(f"{method:<30} | {i2t[0]:5.2f} {i2t[1]:5.2f} {i2t[2]:5.2f} | {t2i[0]:5.2f} {t2i[1]:5.2f} {t2i[2]:5.2f}")
        else:
            print(f"{method:<30} | (Missing data)")
    except Exception as e:
        print(f"Error printing table for {method}: {e}")
print("="*80 + "\n")


# --- FIX FOR LEGEND AND LAYOUT ---

# 1. Get handles and labels from the first axis (assuming all lines are in both)
handles, labels = axes[0].get_legend_handles_labels()

# 2. Setup legend outside the plot
# loc='lower center' relative to the figure, bbox_to_anchor shifts it
# ncol depends on number of methods. Auto-adjust or fixed? 
# Let's use ncol=3 or 4 to spread it out horizontally below.
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), 
           ncol=min(4, len(labels)), frameon=True, fancybox=True, shadow=True, fontsize=10)

# 3. Adjust layout to make room at the bottom
plt.tight_layout()
plt.subplots_adjust(bottom=0.20) # Reserve 20% of height at bottom for legend

# 4. Save figure instead of showing
output_file = 'calibration_cifar100.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight') # bbox_inches='tight' helps too
print(f"Plot saved to {os.path.abspath(output_file)}")
