#!/usr/bin/env python3
"""Quantify the cluster-layout win before rebuilding the pack: k-means the
corpus embeddings, order clusters by a nearest-centroid chain, and simulate
how many coalesced range requests the eval queries' top-C candidates need
under the permuted layout vs the current one.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

DIM = 384
ROW_BYTES = 384  # u8 row in the sketch artifact's lazy tier


def kmeans(x, k, iters=25, seed=0):
    g = torch.Generator(device=x.device).manual_seed(seed)
    centroids = x[torch.randperm(x.shape[0], generator=g, device=x.device)[:k]].clone()
    for it in range(iters):
        assign = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
        for i in range(0, x.shape[0], 65536):
            block = x[i:i + 65536]
            assign[i:i + 65536] = (block @ centroids.T).argmax(dim=1)
        sums = torch.zeros_like(centroids).index_add_(0, assign, x)
        counts = torch.bincount(assign, minlength=k).clamp(min=1).unsqueeze(1)
        newc = torch.nn.functional.normalize(sums / counts, dim=1)
        shift = (newc - centroids).norm(dim=1).max().item()
        centroids = newc
        if shift < 1e-4:
            break
    return assign.cpu().numpy(), centroids


def chain_order(centroids):
    # Greedy nearest-neighbor chain so adjacent clusters are similar too.
    k = centroids.shape[0]
    sims = (centroids @ centroids.T).cpu().numpy()
    np.fill_diagonal(sims, -np.inf)
    order = [0]
    used = {0}
    for _ in range(k - 1):
        last = order[-1]
        for cand in np.argsort(-sims[last]):
            if cand not in used:
                order.append(int(cand))
                used.add(int(cand))
                break
    return order


def coalesce(ids, gap):
    addrs = np.sort(np.asarray(ids, dtype=np.int64)) * ROW_BYTES
    ranges = 1
    filler = 0
    for i in range(1, len(addrs)):
        d = addrs[i] - (addrs[i - 1] + ROW_BYTES)
        if d <= gap:
            filler += d
        else:
            ranges += 1
    return ranges, filler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data-full")
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--C", type=int, default=600)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vecs = np.memmap(args.data / "vectors.f32", dtype=np.float32, mode="r").reshape(-1, DIM)
    x = torch.from_numpy(np.ascontiguousarray(vecs)).to(device)
    k = min(args.k, x.shape[0])
    if k != args.k:
        print(f"k-means: reducing k from {args.k} to {k} for {x.shape[0]} vectors")

    print(f"k-means: {x.shape[0]} vectors -> {k} clusters on {device}")
    assign, centroids = kmeans(x, k)
    order = chain_order(centroids)
    rank_of_cluster = np.empty(k, dtype=np.int64)
    for rank, c in enumerate(order):
        rank_of_cluster[c] = rank
    # Permutation: stable sort by chained cluster rank; new_id[old_id]
    perm = np.argsort(rank_of_cluster[assign], kind="stable")   # new position -> old id
    new_id = np.empty_like(perm)
    new_id[perm] = np.arange(len(perm))

    q = np.fromfile(args.data / "eval-queries.f32", dtype=np.float32).reshape(-1, DIM)
    qt = torch.from_numpy(q).to(device)
    candidate_count = min(args.C, x.shape[0])
    for gap in (4096, 16384, 65536):
        tot_old = tot_new = fill_old = fill_new = 0
        for i in range(q.shape[0]):
            scores = (x @ qt[i]).cpu().numpy()
            top = np.argpartition(-scores, candidate_count - 1)[:candidate_count]
            r_old, f_old = coalesce(top, gap)
            r_new, f_new = coalesce(new_id[top], gap)
            tot_old += r_old; tot_new += r_new; fill_old += f_old; fill_new += f_new
        n = q.shape[0]
        print(f"gap={gap:6d}: requests {tot_old/n:6.1f} -> {tot_new/n:6.1f}   "
              f"filler/query {fill_old/n/1024:7.1f} KiB -> {fill_new/n/1024:7.1f} KiB")

    np.save(args.data / "layout-perm.npy", perm)
    print("saved layout-perm.npy (new position -> old id)")


if __name__ == "__main__":
    main()
