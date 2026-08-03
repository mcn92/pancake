#!/usr/bin/env python3
"""Which resident sketch config lets C shrink? For each config, compute how
often the exact-u8 top-10 (what the pack's rerank would return with the full
rows in hand) survives inside the sketch's top-C, over the eval queries.

Configs: (sketchDims, bits) — pooling is DIM/sketchDims, values quantized to
the pooled row's min/max affine grid, mirroring the builder.
"""
import numpy as np
import torch
from pathlib import Path

DIM = 384
K = 10
DATA = Path(__file__).parent / "data-full"
device = "cuda" if torch.cuda.is_available() else "cpu"

vecs = np.memmap(DATA / "vectors.f32", dtype=np.float32, mode="r").reshape(-1, DIM)
x = torch.from_numpy(np.ascontiguousarray(vecs)).to(device)
q = torch.from_numpy(np.fromfile(DATA / "eval-queries.f32", dtype=np.float32).reshape(-1, DIM)).to(device)

# Exact-u8 reference: row-affine quantization of the full 384-D vectors,
# cosine distance against the float query — the pack's rerank result.
mn = x.min(dim=1, keepdim=True).values
mx = x.max(dim=1, keepdim=True).values
scale = (mx - mn) / 255.0
u8 = torch.round((x - mn) / scale.clamp(min=1e-12)).clamp(0, 255)
xq = mn + scale * u8   # dequantized rows

ref_top = []
for i in range(q.shape[0]):
    scores = xq @ q[i]
    ref_top.append(torch.topk(scores, K).indices.cpu().numpy())

def sketch_matrix(sd, bits):
    # Pool + quantize the whole corpus once per config; per-query work is
    # just the pooled dot product.
    pool = DIM // sd
    pooled = x.reshape(-1, sd, pool).mean(dim=2)
    pmn = pooled.min(dim=1, keepdim=True).values
    pmx = pooled.max(dim=1, keepdim=True).values
    levels = (1 << bits) - 1
    pscale = (pmx - pmn) / levels
    pq = torch.round((pooled - pmn) / pscale.clamp(min=1e-12)).clamp(0, levels)
    return pmn + pscale * pq, pool

for sd, bits in [(96, 4), (96, 8), (192, 4), (192, 8)]:
    deq, pool = sketch_matrix(sd, bits)
    resident_mb = (x.shape[0] * (sd * bits // 8 + 8)) / 1e6
    hits = {c: 0 for c in (50, 100, 150, 200, 300, 600)}
    for i in range(q.shape[0]):
        qp = q[i].reshape(sd, pool).mean(dim=1)
        s = deq @ qp * pool   # pooled dot approximates full dot
        order = torch.argsort(s, descending=True).cpu().numpy()
        for c in hits:
            topc = set(order[:c].tolist())
            hits[c] += sum(1 for t in ref_top[i] if t in topc)
    n = q.shape[0] * K
    curve = "  ".join(f"C={c}:{hits[c]/n*100:5.1f}%" for c in sorted(hits))
    print(f"sketchDims={sd} bits={bits} resident={resident_mb:5.1f}MB  {curve}")
