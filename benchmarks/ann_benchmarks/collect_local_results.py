#!/usr/bin/env python3
"""Collect local ANN-Benchmarks HDF5 results into tables and Pareto plots."""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import pathlib
import sys
from collections import defaultdict
from typing import Any

import h5py
import numpy as np

if not hasattr(np, "Inf"):
    np.Inf = np.inf  # type: ignore[attr-defined]


METRIC_COLUMNS = [
    "k-nn",
    "epsilon",
    "largeepsilon",
    "rel",
    "qps",
    "p50",
    "p95",
    "p99",
    "p999",
    "distcomps",
    "build",
    "candidates",
    "indexsize",
    "queriessize",
]

BASE_COLUMNS = [
    "dataset",
    "count",
    "distance",
    "batch_mode",
    "algorithm",
    "parameters",
]

ATTR_COLUMNS = [
    "run_count",
    "best_search_time",
    "build_time",
    "index_size",
    "source_file",
]


def json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def load_truth_distances(checkout: pathlib.Path, dataset_name: str) -> np.ndarray:
    dataset_path = checkout / "data" / f"{dataset_name}.hdf5"
    if not dataset_path.exists():
        raise FileNotFoundError(f"missing ground-truth dataset: {dataset_path}")
    with h5py.File(dataset_path, "r") as dataset:
        return np.asarray(dataset["distances"])


def result_files(results_dir: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path for path in results_dir.glob("*/*/*/*.hdf5") if path.is_file())


def relative_source(checkout: pathlib.Path, filename: pathlib.Path) -> str:
    try:
        return str(filename.relative_to(checkout))
    except ValueError:
        return str(filename)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkout",
        type=pathlib.Path,
        default=pathlib.Path("ann-benchmarks"),
        help="ANN-Benchmarks checkout containing data/ and results/.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("benchmarks/ann_benchmarks/local_results"),
        help="Directory for the combined table and generated plots.",
    )
    parser.add_argument("--x-axis", default="k-nn", help="Pareto plot x-axis metric.")
    parser.add_argument("--y-axis", default="qps", help="Pareto plot y-axis metric.")
    parser.add_argument("--x-scale", default="linear", help="Pareto plot x-axis scale.")
    parser.add_argument("--y-scale", default="log", help="Pareto plot y-axis scale.")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Include faded non-frontier points in each Pareto plot.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Clear cached metric groups in result files before recomputing.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print ANN-Benchmarks metric logs.")
    args = parser.parse_args()

    checkout = args.checkout.resolve()
    sys.path.insert(0, str(checkout))
    os.environ.setdefault("MPLCONFIGDIR", str(pathlib.Path("/tmp/matplotlib-codex")))

    import plot as ann_plot
    from ann_benchmarks.plotting.metrics import all_metrics
    from ann_benchmarks.plotting.utils import compute_all_metrics, create_linestyles

    if args.x_axis not in all_metrics:
        raise SystemExit(f"unknown x-axis metric: {args.x_axis}")
    if args.y_axis not in all_metrics:
        raise SystemExit(f"unknown y-axis metric: {args.y_axis}")

    files = result_files(checkout / "results")
    if not files:
        raise SystemExit(f"No HDF5 result files found under {checkout / 'results'}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    plot_groups: dict[tuple[str, int, bool], dict[str, list[tuple[str, str, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    truth_cache: dict[str, np.ndarray] = {}

    for filename in files:
        with h5py.File(filename, "r+") as run:
            attrs = {key: json_scalar(value) for key, value in run.attrs.items()}
            dataset_name = str(attrs["dataset"])
            if dataset_name not in truth_cache:
                truth_cache[dataset_name] = load_truth_distances(checkout, dataset_name)

            algorithm = str(attrs["algo"])
            parameters = str(attrs.get("name", algorithm))
            count = int(attrs["count"])
            batch_mode = bool(attrs["batch_mode"])

            if args.verbose:
                algo, algo_name, metric_values = compute_all_metrics(
                    truth_cache[dataset_name], run, attrs, args.recompute
                )
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    algo, algo_name, metric_values = compute_all_metrics(
                        truth_cache[dataset_name], run, attrs, args.recompute
                    )

            row: dict[str, Any] = {
                "dataset": dataset_name,
                "count": count,
                "distance": attrs.get("distance", ""),
                "batch_mode": batch_mode,
                "algorithm": algo,
                "parameters": algo_name,
                "source_file": relative_source(checkout, filename),
                "attrs_json": json.dumps(attrs, sort_keys=True),
            }
            for column in METRIC_COLUMNS:
                row[column] = json_scalar(metric_values.get(column, ""))
            for column in ATTR_COLUMNS:
                if column != "source_file":
                    row[column] = json_scalar(attrs.get(column, ""))
            rows.append(row)

            plot_groups[(dataset_name, count, batch_mode)][algorithm].append(
                (algorithm, parameters, metric_values[args.x_axis], metric_values[args.y_axis])
            )

    rows.sort(key=lambda row: (row["dataset"], row["count"], row["batch_mode"], row["algorithm"], row["parameters"]))

    csv_path = args.output_dir / "ann_benchmarks_local_results.csv"
    json_path = args.output_dir / "ann_benchmarks_local_results.json"
    fieldnames = BASE_COLUMNS + METRIC_COLUMNS + ATTR_COLUMNS + ["attrs_json"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    linestyles = create_linestyles(sorted({row["algorithm"] for row in rows}))
    plot_paths = []
    for (dataset_name, count, batch_mode), all_data in sorted(plot_groups.items()):
        suffix = "-batch" if batch_mode else ""
        plot_path = args.output_dir / f"{dataset_name}-{count}{suffix}-{args.x_axis}-vs-{args.y_axis}.png"
        ann_plot.create_plot(
            dict(all_data),
            args.raw,
            args.x_scale,
            args.y_scale,
            args.x_axis,
            args.y_axis,
            str(plot_path),
            linestyles,
            batch_mode,
        )
        plot_paths.append(plot_path)

    print(f"Wrote {len(rows)} rows to {csv_path}")
    print(f"Wrote {json_path}")
    for plot_path in plot_paths:
        print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
