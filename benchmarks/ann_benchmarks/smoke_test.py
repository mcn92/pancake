#!/usr/bin/env python3
"""Small correctness/recall smoke test for the installed ANN-Benchmarks adapter."""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np


def normalize(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norms, 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=pathlib.Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.checkout.resolve()))

    from ann_benchmarks.algorithms.pancake.module import Pancake

    rng = np.random.default_rng(20260712)
    dimensions = 32
    count = 2500
    centers = normalize(rng.normal(size=(32, dimensions)).astype(np.float32))
    assignments = np.arange(count) % len(centers)
    data = centers[assignments] + rng.normal(0, 0.12, size=(count, dimensions)).astype(np.float32)
    queries = data[rng.choice(count, size=40, replace=False)] + rng.normal(
        0, 0.015, size=(40, dimensions)
    ).astype(np.float32)

    for metric in ("euclidean", "angular"):
        metric_data = normalize(data) if metric == "angular" else data
        metric_queries = normalize(queries) if metric == "angular" else queries
        for quantized in (True, False):
            adapter = Pancake(metric, quantized, {"M": 8, "efConstruction": 100})
            adapter.fit(metric_data)
            adapter.set_query_arguments(200)
            recalls = []
            for query in metric_queries:
                found = adapter.query(query, 10)
                if metric == "angular":
                    scores = metric_data @ query
                    truth = np.argpartition(-scores, 10)[:10]
                else:
                    distances = np.sum((metric_data - query) ** 2, axis=1)
                    truth = np.argpartition(distances, 10)[:10]
                recalls.append(len(set(found.tolist()).intersection(truth.tolist())) / 10)
            recall = float(np.mean(recalls))
            if recall < 0.75:
                raise RuntimeError(f"Unexpectedly low {adapter}: recall@10={recall:.3f}")
            print(f"{adapter}: recall@10={recall:.3f}, memory={adapter.get_additional()['engine_memory_bytes']} bytes")
            adapter.done()

    metric_data = normalize(data)
    metric_data[7] = 0
    metric_data[211] = 0
    metric_queries = normalize(queries)
    metric_queries[3] = 0
    adapter = Pancake("angular", True, {"M": 8, "efConstruction": 100})
    adapter.fit(metric_data)
    adapter.set_query_arguments(200)
    if adapter.get_additional()["filtered_base_vectors"] != 0:
        raise RuntimeError("Angular zero-vector retention should not report filtered base rows")
    if adapter.get_additional()["retained_zero_vectors"] != 2:
        raise RuntimeError("Angular zero-vector retention did not report the expected count")
    zero_query_results = adapter.query(metric_queries[3], 10).tolist()
    if zero_query_results[:2] != [7, 211]:
        raise RuntimeError("Zero angular query should return retained zero-vector IDs first")
    found = adapter.query(metric_queries[0], 10)
    if 7 in found.tolist() or 211 in found.tolist():
        raise RuntimeError("Zero angular base vector should not displace normal nonzero results")
    print(f"{adapter}: zero-vector retention smoke passed")
    adapter.done()


if __name__ == "__main__":
    main()
