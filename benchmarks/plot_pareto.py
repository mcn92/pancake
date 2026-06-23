#!/usr/bin/env python3
"""
Plot Pareto-frontier (QPS-recall) curves and interpolated equal-recall curves
from a pareto_frontier.js run.

Usage:
  python3 benchmarks/plot_pareto.py benchmark_results/pareto_<ts>.csv

Expects, alongside the raw sweep CSV, the sibling files written by the harness:
  pareto_<ts>_frontier.csv      (label, ef_search, recall, qps)
  pareto_<ts>_equalrecall.csv   (recall, <one column per sweepable label>)
If those are missing, frontiers are recomputed from the raw CSV.
"""
import sys
import os
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
    print("Usage: plot_pareto.py <pareto_*.csv>")
    sys.exit(1)

raw_path = sys.argv[1]
base = raw_path[:-4] if raw_path.endswith('.csv') else raw_path
frontier_path = base + '_frontier.csv'
equalrecall_path = base + '_equalrecall.csv'

df = pd.read_csv(raw_path)

# Stable per-label styling. Pancake configs warm, baselines cool.
LABELS = list(df['label'].unique())
PALETTE = ['#d9480f', '#e8590c', '#f08c00', '#c2255c',  # pancake-ish warms
           '#0c8599', '#1098ad', '#1971c2', '#5c940d', '#862e9c', '#495057']
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '<']

def style_for(i):
    return dict(color=PALETTE[i % len(PALETTE)], marker=MARKERS[i % len(MARKERS)])

style = {l: style_for(i) for i, l in enumerate(LABELS)}


def pareto_frontier(g):
    """Non-dominated (recall, qps) points, sorted by recall."""
    pts = g[['recall', 'qps', 'ef_search']].to_dict('records')
    front = [a for a in pts
             if not any(b is not a and b['recall'] >= a['recall'] and b['qps'] >= a['qps']
                        and (b['recall'] > a['recall'] or b['qps'] > a['qps'])
                        for b in pts)]
    return pd.DataFrame(sorted(front, key=lambda r: r['recall']))


# Prefer the harness-written frontier CSV; else recompute.
if os.path.exists(frontier_path):
    fdf = pd.read_csv(frontier_path)
    frontiers = {l: g.sort_values('recall') for l, g in fdf.groupby('label')}
else:
    frontiers = {l: pareto_frontier(g) for l, g in df.groupby('label')}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

def plot_dtype_panel(ax, dtype_value, title):
    subset = df[df['dtype'] == dtype_value]
    for label, g in subset.groupby('label'):
        g = g.sort_values('recall')
        st = style[label]
        single = len(g) == 1
        ax.scatter(g['recall'].values, g['qps'].values, s=70 if single else 35,
                   alpha=0.9 if single else 0.25, **st,
                   label=None if not single else f'{label} (single pt, not tunable)')
        fr = frontiers.get(label)
        if fr is not None and len(fr) > 1:
            ax.plot(fr['recall'].values, fr['qps'].values, linewidth=2.0, alpha=0.95,
                    linestyle='-', label=label, **st)

    ax.set_xlabel('Recall@10', fontsize=12)
    ax.set_yscale('log')
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='lower left', framealpha=0.9, fontsize=8)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

plot_dtype_panel(ax1, 'i8', 'Pareto frontier: int8 / i8')
plot_dtype_panel(ax2, 'f32', 'Pareto frontier: f32')
ax1.set_ylabel('QPS (single-thread)', fontsize=12)

fig.tight_layout()
out_path = base + '.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
