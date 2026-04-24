#!/usr/bin/env python3
"""
Plot recall-QPS curves from the sweep CSV.

Usage:
  python3 plot_sweep.py benchmark_results/sweep_nytimes_<timestamp>.csv
"""
import sys
import pandas as pd
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

# Consistent colors: pancake = warm, usearch-int8 = teal, usearch-f32 = gray
STYLE = {
    'pancake-int8-wasm':   {'color': '#d9480f', 'marker': 'o', 'linestyle': '-'},
    'usearch-int8-native': {'color': '#0c8599', 'marker': 's', 'linestyle': '-'},
    'usearch-f32-native':  {'color': '#495057', 'marker': '^', 'linestyle': '--'},
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- Plot 1: recall-QPS curves (the canonical ann-benchmarks view) ---
for label, g in df.groupby('label'):
    g = g.sort_values('recall')
    style = STYLE.get(label, {'color': 'k', 'marker': 'x', 'linestyle': '-'})
    ax1.errorbar(g['recall'], g['qps'],
                 xerr=g['recall_std'], yerr=g['qps_std'],
                 label=label, capsize=3, markersize=7, linewidth=1.5,
                 **style)

ax1.set_xlabel('Recall@10', fontsize=12)
ax1.set_ylabel('QPS (single-thread)', fontsize=12)
ax1.set_yscale('log')
ax1.set_title(f'Recall-QPS tradeoff on {DATASET_NAME}', fontsize=13)
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(loc='lower left', framealpha=0.9)
ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())

# Annotate ef_search values on the Pancake curve for reference
pancake = df[df['label'] == 'pancake-int8-wasm'].sort_values('ef_search')
for _, row in pancake.iterrows():
    ax1.annotate(f"ef={int(row.ef_search)}",
                 xy=(row.recall, row.qps),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=8, alpha=0.6)

# --- Plot 2: latency curves ---
for label, g in df.groupby('label'):
    g = g.sort_values('recall')
    style = STYLE.get(label, {'color': 'k', 'marker': 'x', 'linestyle': '-'})
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

# --- Print the matched-recall comparison table ---
print("\nMatched-recall QPS comparison (pancake / baseline):")
print(f"{'recall':>8}  {'pancake':>10}  {'usearch-i8':>12}  {'usearch-f32':>12}")

pancake_by_recall = df[df['label'] == 'pancake-int8-wasm'][['recall', 'qps']].sort_values('recall')
import numpy as np
for _, row in pancake_by_recall.iterrows():
    target_recall = row['recall']
    line = f"  {target_recall*100:>5.1f}%  {row['qps']:>10.0f}"
    for baseline in ['usearch-int8-native', 'usearch-f32-native']:
        b = df[df['label'] == baseline].sort_values('recall')
        if len(b) and b['recall'].min() <= target_recall <= b['recall'].max():
            # Log-linear interpolation in recall-QPS space
            interp = np.exp(np.interp(target_recall, b['recall'], np.log(b['qps'])))
            line += f"  {interp:>12.0f}"
        else:
            line += f"  {'n/a':>12}"
    print(line)