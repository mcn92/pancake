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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# --- Plot 1: all sweep points (faint) + Pareto frontier (bold) ---
for i, (label, g) in enumerate(df.groupby('label')):
    g = g.sort_values('recall')
    st = style[label]
    single = len(g) == 1
    # Raw sweep points, faint.
    ax1.scatter(g['recall'].values, g['qps'].values, s=70 if single else 35,
                alpha=0.9 if single else 0.25, **st,
                label=None if not single else f'{label} (single pt, not tunable)')
    # Frontier, bold line.
    fr = frontiers.get(label)
    if fr is not None and len(fr) > 1:
        ax1.plot(fr['recall'].values, fr['qps'].values, linewidth=2.0, alpha=0.95,
                 linestyle='-', label=label, **st)

ax1.set_xlabel('Recall@10', fontsize=12)
ax1.set_ylabel('QPS (single-thread)', fontsize=12)
ax1.set_yscale('log')
ax1.set_title('Pareto frontier: QPS vs Recall@10', fontsize=13)
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(loc='lower left', framealpha=0.9, fontsize=8)
ax1.yaxis.set_major_formatter(mticker.ScalarFormatter())

# --- Plot 2: interpolated equal-recall curves ---
if os.path.exists(equalrecall_path):
    eq = pd.read_csv(equalrecall_path)
    cols = [c for c in eq.columns if c != 'recall']
    for label in cols:
        st = style.get(label, style_for(0))
        sub = eq[['recall', label]].dropna()
        if len(sub):
            ax2.plot(sub['recall'].values, sub[label].values, linewidth=1.8, marker=st['marker'],
                     markersize=6, color=st['color'], label=label)
    ax2.set_xlabel('Recall@10 (target)', fontsize=12)
    ax2.set_ylabel('Interpolated QPS', fontsize=12)
    ax2.set_yscale('log')
    ax2.set_title('Equal-recall QPS (log-linear interpolation)', fontsize=13)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.legend(loc='upper right', framealpha=0.9, fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.ScalarFormatter())
else:
    ax2.text(0.5, 0.5, 'no equal-recall CSV found', ha='center', va='center')

fig.tight_layout()
out_path = base + '.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
