import os,sys,glob
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

title_dic = {'single_adversarial':'Adversarial',
             'single_adversarial_lin': 'Adversarial lin.',
             'single_top1similarity': 'Top1 similarity',
             'montecarlo_adversarial': 'Adversarial (MCD)',
             'montecarlo_adversarial_lin': 'Adversarial lin. (MCD)',
             'montecarlo_top1consistency': 'Top1 consistency (MCD)',
             'montecarlo_top1similarity': 'Top1 similarity (MCD)',
             'ensemble_adversarial': 'Adversarial (Ens.)',
             'ensemble_adversarial_lin': 'Adversarial lin. (Ens.)',
             'ensemble_top1consistency': 'Top1 consistency (Ens.)',
             'ensemble_top1similarity': 'Top1 similarity (Ens.)'}

data = {}
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

for fname in files:
    with open(fname) as f:
        stem = os.path.splitext(os.path.basename(fname))[0]
        parts = stem.split('_')
        # Expecting: metrics_rejection_{dataset}_{strategy}_{method}
        # We need keys like: {strategy}_{method} (e.g. single_adversarial)
        if len(parts) >= 4 and parts[0] == 'metrics' and parts[1] == 'rejection':
            k = '_'.join(parts[3:]) # Skip dataset name at parts[2]
        elif len(parts) >= 3 and parts[0] == 'metrics' and parts[1] == 'rejection':
             k = '_'.join(parts[2:])
        else:
            k = stem
        print(f"Loading {fname} as key: {k}")
        data[k] = json.load(f)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

aux = {}
for ax in axes.ravel():
    aux[ax] = []


# Filter out single_top1consistency specifically
methods = sorted(data.keys())
methods = [m for m in methods if m != "single_top1consistency"]

for method in methods:
  for model in data[method].keys():
    for ax,k in zip(axes.ravel(), data[method][model].keys()):
        d = np.array(data[method][model][k])
        print(d[:,0])
        # Compute the area under the curve using the trapezoidal rule
        if np.max(d[:,0]) == 0:
            x_norm = d[:,0]
        else:
            x_norm = d[:,0]/np.max(d[:,0])

        area = np.trapz(d[:,1], x=x_norm)

        x_plot = d[:,0]
        y_plot = d[:,1]
        
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
                marker = 'o'
        else:
            marker = 'o'

        if 'oracle' in method:
            label = f'Upper-bound ({area:0.2f})'
            line_handle, = ax.plot(x_plot, y_plot, label=label, color='black', linestyle='dashed', marker=marker)
        else:
            title = title_dic.get(method, method)
            label = f'{title} ({area:0.2f})'
            line_handle, = ax.plot(x_plot, y_plot, label=label, marker=marker)

        aux[ax].append((line_handle, label, area))
        ax.set_title(k)

for ax in axes.ravel():
    sorted_data = sorted(aux[ax], key=lambda tup: tup[2], reverse=True)
    sorted_handles = [tup[0] for tup in sorted_data]
    sorted_labels = [tup[1] for tup in sorted_data]
    ax.legend(sorted_handles, sorted_labels)

plt.tight_layout()
plt.savefig('rejection_cifar100.png')
