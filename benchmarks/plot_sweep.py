#!/usr/bin/env python3
"""
Plot recall-QPS curves from any sweep CSV.

Usage:
  python3 benchmarks/plot_sweep.py benchmark_results/sweep_*.csv
  python3 benchmarks/plot_sweep.py nytimes/benchmark_results/sweep_*.csv
"""
import sys
import pandas as pd
import numpy as np

# Matplotlib < 3.6 still reaches for legacy NumPy aliases like np.Inf and
# np.NaN, which were removed in NumPy 2.x. Provide the aliases here so the
# plotting helper remains usable on mixed older-matplotlib/newer-numpy setups.
if not hasattr(np, 'Inf'):
    np.Inf = np.inf
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

if len(sys.argv) < 2:
    print("Usage: plot_sweep.py <csv_path>")
    sys.exit(1)

csv_path = sys.argv[1]
df = pd.read_csv(csv_path)

# Derive dataset name from filename
_base = csv_path.rsplit('/', 1)[-1].lower()
if 'sift' in _base:
    DATASET_NAME = 'SIFT-1M'
elif 'nytimes' in _base or 'hnswlib' in _base:
    DATASET_NAME = 'NYTimes-256'
else:
    DATASET_NAME = 'Sweep'

# Identify pancake and baseline labels from data
labels = df['label'].unique().tolist()
pancake_label = next((l for l in labels if 'pancake' in l.lower()), None)
baseline_labels = [l for l in labels if l != pancake_label]

# Color palette: pancake warm, baselines cool
COLORS = ['#d9480f', '#0c8599', '#495057', '#5c940d', '#862e9c']
MARKERS = ['o', 's', '^', 'D', 'v']

style_map = {}
if pancake_label:
    style_map[pancake_label] = {'color': COLORS[0], 'marker': MARKERS[0], 'linestyle': '-'}
for i, bl in enumerate(baseline_labels):
    style_map[bl] = {'color': COLORS[1 + i], 'marker': MARKERS[1 + i], 'linestyle': '--'}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: recall-QPS curves ---
for label, g in df.groupby('label'):
    g = g.sort_values('recall')
    style = style_map.get(label, {'color': 'k', 'marker': 'x', 'linestyle': '-'})
    ax1.errorbar(g['recall'], g['qps'],
                 xerr=g.get('recall_std', 0), yerr=g.get('qps_std', 0),
                 label=label, capsize=3, markersize=7, linewidth=1.5,
                 **style)

ax1.set_xlabel('Recall@10', fontsize=12)
ax1.set_ylabel('QPS (single-thread)', fontsize=12)
ax1.set_yscale('log')
ax1.set_title(f'Recall-QPS tradeoff on {DATASET_NAME}', fontsize=13)
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(loc='lower left', framealpha=0.9)
ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())

# Annotate ef_search values on pancake curve
if pancake_label:
    pk = df[df['label'] == pancake_label].sort_values('ef_search')
    for _, row in pk.iterrows():
        ax1.annotate(f"ef={int(row.ef_search)}",
                     xy=(row.recall, row.qps),
                     xytext=(5, 5), textcoords='offset points',
                     fontsize=8, alpha=0.6)

# --- Plot 2: latency curves ---
for label, g in df.groupby('label'):
    g = g.sort_values('recall')
    style = style_map.get(label, {'color': 'k', 'marker': 'x', 'linestyle': '-'})
    ax2.plot(g['recall'].values, g['p50_ms'].values, label=f'{label} p50',
             markersize=7, linewidth=1.5, **style)
    ax2.plot(g['recall'].values, g['p99_ms'].values, markersize=5, linewidth=1.0,
             alpha=0.5,
             color=style['color'], marker=style['marker'],
             linestyle=':', label=f'{label} p99')

ax2.set_xlabel('Recall@10', fontsize=12)
ax2.set_ylabel('Latency (ms, single-thread)', fontsize=12)
ax2.set_yscale('log')
ax2.set_title(f'Recall-Latency tradeoff on {DATASET_NAME}', fontsize=13)
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(loc='upper left', framealpha=0.9, fontsize=9)
ax2.yaxis.set_major_formatter(mticker.ScalarFormatter())

fig.tight_layout()
out_path = csv_path.replace('.csv', '.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")

# --- Print matched-recall comparison table ---
if pancake_label and baseline_labels:
    print(f"\nMatched-recall QPS comparison:")
    header = f"{'recall':>8}  {pancake_label:>20}"
    for bl in baseline_labels:
        header += f"  {bl:>20}"
    print(header)

    pk_data = df[df['label'] == pancake_label][['recall', 'qps']].sort_values('recall')
    for _, row in pk_data.iterrows():
        target = row['recall']
        line = f"  {target*100:>5.1f}%  {row['qps']:>20.0f}"
        for bl in baseline_labels:
            b = df[df['label'] == bl].sort_values('recall')
            if len(b) and b['recall'].min() <= target <= b['recall'].max():
                interp = np.exp(np.interp(target, b['recall'], np.log(b['qps'])))
                line += f"  {interp:>20.0f}"
            else:
                line += f"  {'n/a':>20}"
        print(line)
elif pancake_label:
    print(f"\nPancake-only results (no baseline to compare):")
    pk_data = df[df['label'] == pancake_label][['recall', 'qps', 'ef_search']].sort_values('ef_search')
    print(f"  {'ef':>4}  {'recall':>8}  {'qps':>6}")
    for _, row in pk_data.iterrows():
        print(f"  {int(row.ef_search):>4}  {row.recall*100:>6.1f}%  {row.qps:>6.0f}")
