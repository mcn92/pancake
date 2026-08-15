#!/usr/bin/env python3
"""Phase 1 of the per-category student experiment: cosine k-means over the
wiki pack's chunk embeddings (vectors.f32, unit-normalized) to define
topical categories. Writes cluster assignments and a size summary.

  python3 cluster-wiki.py [k=32] [iters=12]

Outputs (student-pilot/, gitignored):
  cluster-assignments.u16   one u16 per chunk row
  cluster-summary.json      sizes + sample titles per cluster
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "04-static-wiki-pack" / "data-full"
OUT = HERE / "student-pilot"

k = int(sys.argv[1]) if len(sys.argv) > 1 else 32
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 12
dim = 384
rng = np.random.default_rng(20260814)

vectors = np.memmap(DATA / "vectors.f32", dtype=np.float32, mode="r")
count = vectors.shape[0] // dim
vectors = vectors.reshape(count, dim)
print(f"[cluster] {count} chunks, {dim}D, k={k}, iters={iters}", flush=True)

# k-means++-ish seeding from a subsample, then Lloyd iterations with chunked
# matmuls (vectors are unit-normalized, so cosine similarity is a dot).
seed_pool = vectors[rng.choice(count, size=min(20000, count), replace=False)]
centroids = seed_pool[rng.choice(seed_pool.shape[0], size=k, replace=False)].astype(np.float32).copy()

BATCH = 65536
assignments = np.zeros(count, dtype=np.uint16)
for iteration in range(iters):
    moved = 0
    sums = np.zeros((k, dim), dtype=np.float64)
    counts = np.zeros(k, dtype=np.int64)
    for start in range(0, count, BATCH):
        block = np.asarray(vectors[start:start + BATCH])
        sims = block @ centroids.T
        best = sims.argmax(axis=1).astype(np.uint16)
        moved += int((assignments[start:start + BATCH] != best).sum())
        assignments[start:start + BATCH] = best
        np.add.at(sums, best, block)
        counts += np.bincount(best, minlength=k)
    for j in range(k):
        if counts[j] > 0:
            centroid = sums[j] / counts[j]
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroids[j] = (centroid / norm).astype(np.float32)
    print(f"[cluster] iter {iteration + 1}: moved {moved} ({moved / count:.1%})", flush=True)
    if moved / count < 0.005:
        break

OUT.mkdir(exist_ok=True)
assignments.tofile(OUT / "cluster-assignments.u16")

titles = []
with open(DATA / "corpus.jsonl", encoding="utf-8") as f:
    for line in f:
        titles.append(json.loads(line).get("title", ""))

summary = []
for j in range(k):
    members = np.flatnonzero(assignments == j)
    sample = [titles[i] for i in rng.choice(members, size=min(8, len(members)), replace=False)] if len(members) else []
    summary.append({"cluster": j, "chunks": int(len(members)), "sampleTitles": sample})
summary.sort(key=lambda s: -s["chunks"])
(OUT / "cluster-summary.json").write_text(json.dumps(summary, indent=1))
print(f"[cluster] wrote assignments + summary; sizes: "
      f"min={min(s['chunks'] for s in summary)} max={max(s['chunks'] for s in summary)}", flush=True)
