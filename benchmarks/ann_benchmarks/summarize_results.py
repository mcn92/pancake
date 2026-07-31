#!/usr/bin/env python3
"""Summarize Pancake HDF5 files produced by the official ANN-Benchmarks runner."""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import re
import shutil

import h5py
import numpy as np


def recall_at_k(truth_distances: np.ndarray, run_distances: np.ndarray, k: int) -> float:
    thresholds = truth_distances[:, k - 1] + 1e-3
    hits = np.sum(run_distances[:, :k] <= thresholds[:, None], axis=1)
    return float(np.mean(hits / k))


def scalar(value):
    return value.item() if hasattr(value, "item") else value


def int_attr(attrs, key, fallback=0):
    if key in attrs and attrs[key] is not None:
        return int(attrs[key])
    return fallback


def parse_int_field(attrs, filename, key):
    if key in attrs and attrs[key] is not None:
        return int(attrs[key])
    name = str(attrs.get("name", ""))
    for text in (filename.name, name):
        match = re.search(rf"{re.escape(key)}['\"]?\s*[:_=]\s*([0-9]+)", text)
        if match:
            return int(match.group(1))
    return 0


def parse_first_int_field(attrs, filename, keys):
    for key in keys:
        value = parse_int_field(attrs, filename, key)
        if value:
            return value
    return 0


def parse_ef_search(attrs, filename):
    if "efSearch" in attrs and attrs["efSearch"] is not None:
        return int(attrs["efSearch"])
    name = str(attrs.get("name", ""))
    for text in (filename.name, name):
        for pattern in (r"efSearch['\"]?\s*[:_=]\s*([0-9]+)", r"efQuery['\"]?\s*[:_=]\s*([0-9]+)"):
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))
    match = re.search(r"_([0-9]+)\.hdf5$", filename.name)
    if match:
        return int(match.group(1))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkout", type=pathlib.Path)
    parser.add_argument("--dataset", default="sift-128-euclidean")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--algorithm", default="pancake-u8")
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--copy-hdf5", action="store_true")
    args = parser.parse_args()

    checkout = args.checkout.resolve()
    dataset_path = checkout / "data" / f"{args.dataset}.hdf5"
    result_dir = checkout / "results" / args.dataset / str(args.count) / args.algorithm
    files = sorted(result_dir.glob("*.hdf5"))
    if not files:
        raise SystemExit(f"No result files found under {result_dir}")

    with h5py.File(dataset_path, "r") as dataset:
        truth_distances = np.asarray(dataset["distances"])

    rows = []
    for filename in files:
        with h5py.File(filename, "r") as result:
            attrs = {key: scalar(value) for key, value in result.attrs.items()}
            times = np.asarray(result["times"])
            distances = np.asarray(result["distances"])
            rows.append({
                "algorithm": attrs["algo"],
                "storage": attrs.get("storage"),
                "M": parse_first_int_field(attrs, filename, ("M", "R")),
                "efConstruction": parse_first_int_field(attrs, filename, ("efConstruction", "L")),
                "efSearch": parse_ef_search(attrs, filename),
                "recallAt10": recall_at_k(truth_distances, distances, args.count),
                "queriesPerSecond": 1.0 / float(attrs["best_search_time"]),
                "p50Ms": float(np.percentile(times, 50) * 1000),
                "p95Ms": float(np.percentile(times, 95) * 1000),
                "p99Ms": float(np.percentile(times, 99) * 1000),
                "buildSeconds": float(attrs["build_time"]),
                "runnerIndexSizeKiB": float(attrs["index_size"]),
                "engineMemoryBytes": int(attrs.get("engine_memory_bytes", 0)),
                "filteredBaseVectors": int(attrs.get("filtered_base_vectors", 0)),
                "retainedZeroVectors": int(attrs.get("retained_zero_vectors", 0)),
                "queries": int(times.shape[0]),
                "runs": int(attrs["run_count"]),
                "sourceFile": filename.name,
            })

    rows.sort(key=lambda row: (row["M"], row["efConstruction"], row["efSearch"]))
    output = args.output.resolve() if args.output else result_dir / "summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "ann-benchmarks-summary",
        "upstream": "https://github.com/erikbern/ann-benchmarks",
        "upstreamCommit": "2e081ad32c1eccab72dcb739ad886c310b90f715",
        "dataset": args.dataset,
        "count": args.count,
        "algorithm": args.algorithm,
        "runnerMode": "local-single-query",
        "rows": rows,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    if args.copy_hdf5:
        receipt_dir = output.parent / "hdf5"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        for filename in files:
            shutil.copy2(filename, receipt_dir / filename.name)

    for row in rows:
        print(
            f"ef={row['efSearch']:>3} recall@10={row['recallAt10']:.4f} "
            f"qps={row['queriesPerSecond']:.0f} p99={row['p99Ms']:.3f}ms"
        )
    print(f"Wrote {output}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
