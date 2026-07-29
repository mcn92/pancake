# Benchmark Results

This directory contains Pareto-frontier benchmark outputs.
The generated ground-truth cache files under `benchmark_results/cache/` are
local cache artifacts and are intentionally not part of the committed result
set.

## Run Environment

- Host: AMD Ryzen 9 4900HS with Radeon Graphics, 8 logical CPUs
- RAM: 12.5 GB
- Platform: linux/x64
- Node: v22.19.0
- Benchmark parameters: `k=10`,
  `efSearch=10,20,40,60,80,100,150,200,300,500,800`, 3 repetitions.
  The current DBpedia-50K release run used `M=16`, `efConstruction=50`;
  NYTimes, SIFT, and GloVe used `M=12`, `efConstruction=75`.

## Current Result Sets

| Dataset | Vectors | Dim | Metric | Queries | Result prefix |
|:--------|--------:|----:|:-------|--------:|:--------------|
| DBpedia-50K | 50,000 | 1536 | L2 | 1,000 | `release/pareto_dbpedia_2026-07-15T08-18-46-756Z` |
| NYTimes-256 | 290,000 | 256 | cosine | 1,000 | `release/pareto_nytimes_2026-07-26T05-33-25-537Z` |
| SIFT-1M | 1,000,000 | 128 | L2 | 1,000 | `release/pareto_sift_2026-07-26T06-12-28-697Z` |
| GloVe-100 | 1,183,514 | 100 | cosine | 1,000 | `release/pareto_glove_2026-07-26T08-55-02-282Z` |

Each result set includes:

- `.csv`: every measured backend and `efSearch` point
- `_frontier.csv`: non-dominated QPS/recall points
- `_equalrecall.csv`: interpolated equal-recall comparison table
- `.json`: full structured results with system and memory metadata
- `.log`: console output from the benchmark run
- `.png`: Pareto frontier plot

## Scaled churn confirmation

`churn_scale_100k.json` records a deterministic 100K-vector clustered cosine
run with five complete population turnovers (`M=8`, `efConstruction=100`,
`efSearch=200`). Recall@10 fell from 96.0% at baseline to 7.2% at 83.3%
deleted nodes, with only 4.8 results returned on average. The high-deletion
rebuild compaction took 17.6s and restored 99.2% recall with a full top-10; a
fresh index over the identical final population reached 97.2%. The WASM heap
was 171.6 MB immediately before and after compaction, confirming that the
replacement graph reused released engine allocations rather than retaining a
two-graph peak. Reproduce with `node benchmarks/churn_scale.js`.

## DBpedia-50K (50,000 x 1536D, L2)

| Backend | Build | Memory | ef=100 recall | ef=100 QPS | ef=800 recall | ef=800 QPS |
|:--|--:|--:|--:|--:|--:|--:|
| pancake-wasm-int8 | 73.4s | 86.3 MB | 96.70% | 1,134 | 97.58% | 214 |
| pancake-wasm-fp32 | 155.1s | 299.4 MB | 98.83% | 829 | 99.90% | 156 |
| pancake-native-int8 | 37.4s | 86.3 MB | 96.68% | 1,396 | 97.56% | 274 |
| pancake-native-fp32 | 93.3s | 299.4 MB | 98.83% | 1,025 | 99.90% | 203 |
| usearch-int8 | 15.6s | 80.3 MB (file size) | 88.91% | 1,649 | 89.80% | 308 |
| usearch-f16 | 29.5s | 153.6 MB (file size) | 98.14% | 1,042 | 99.89% | 206 |
| usearch-fp32 | 44.4s | 300.1 MB (file size) | 98.19% | 794 | 99.90% | 154 |
| hnswlib-fp32 | 40.5s | 305.3 MB (RSS delta) | 98.20% | 1,016 | 99.86% | 210 |

## NYTimes-256 (290,000 x 256D, cosine)

| Backend | Build | Memory | ef=100 recall | ef=100 QPS | ef=800 recall | ef=800 QPS |
|:--|--:|--:|--:|--:|--:|--:|
| pancake-wasm-int8 | 168.9s | 129.1 MB | 79.75% | 2,050 | 91.86% | 320 |
| pancake-wasm-fp32 | 242.6s | 311.5 MB | 79.91% | 1,514 | 92.43% | 240 |
| pancake-native-int8 | 119.8s | 129.1 MB | 80.03% | 2,570 | 92.04% | 386 |
| pancake-native-fp32 | 174.2s | 311.5 MB | 79.91% | 1,871 | 92.43% | 311 |
| usearch-int8 | 109.9s | 103.1 MB (file size) | 73.75% | 2,031 | 86.13% | 334 |
| usearch-f16 | 130.3s | 173.9 MB (file size) | 77.39% | 1,708 | 89.50% | 288 |
| usearch-fp32 | 178.9s | 315.5 MB (file size) | 76.99% | 1,361 | 89.59% | 229 |
| hnswlib-fp32 | 109.1s | 336.3 MB (RSS delta) | 77.01% | 2,709 | 90.07% | 460 |

## SIFT-1M (1,000,000 x 128D, L2)

| Backend | Build | Memory | ef=100 recall | ef=100 QPS | ef=800 recall | ef=800 QPS |
|:--|--:|--:|--:|--:|--:|--:|
| pancake-wasm-int8 | 421.9s | 323.0 MB | 96.58% | 2,212 | 99.20% | 372 |
| pancake-wasm-fp32 | 499.1s | 585.9 MB | 96.97% | 1,821 | 99.80% | 315 |
| pancake-native-int8 | 324.0s | 323.1 MB | 96.60% | 2,558 | 99.19% | 432 |
| pancake-native-fp32 | 321.3s | 585.9 MB | 96.97% | 2,869 | 99.80% | 476 |
| usearch-int8 | 209.3s | 233.4 MB (file size) | 89.20% | 3,132 | 91.65% | 538 |
| usearch-f16 | 313.8s | 355.5 MB (file size) | 95.74% | 1,886 | 99.69% | 327 |
| usearch-fp32 | 373.3s | 599.6 MB (file size) | 95.74% | 1,651 | 99.69% | 283 |
| hnswlib-fp32 | 232.9s | 629.7 MB (RSS delta) | 95.89% | 3,340 | 99.82% | 587 |

## GloVe-100 (1,183,514 x 100D, cosine)

| Backend | Build | Memory | ef=100 recall | ef=100 QPS | ef=800 recall | ef=800 QPS |
|:--|--:|--:|--:|--:|--:|--:|
| pancake-wasm-int8 | 601.7s | 350.7 MB | 75.38% | 1,894 | 90.41% | 320 |
| pancake-wasm-fp32 | 690.2s | 567.0 MB | 76.24% | 1,512 | 90.93% | 258 |
| pancake-native-int8 | 480.9s | 350.7 MB | 74.98% | 2,145 | 90.37% | 366 |
| pancake-native-fp32 | 456.9s | 567.0 MB | 76.27% | 2,365 | 90.94% | 395 |
| usearch-int8 | 522.7s | 244.6 MB (file size) | 71.21% | 1,684 | 85.28% | 297 |
| usearch-f16 | 555.0s | 357.4 MB (file size) | 72.63% | 1,459 | 89.99% | 250 |
| usearch-fp32 | 631.9s | 583.2 MB (file size) | 73.49% | 1,169 | 90.19% | 216 |
| hnswlib-fp32 | 349.7s | n/a (RSS delta) | 74.02% | 3,255 | 90.00% | 536 |

## Memory Notes

Pancake memory is reported by the index. USearch does not expose an equivalent
runtime memory counter in this harness, so its table value is the saved index
file size. hnswlib is reported as process RSS delta. RSS deltas are useful for
order-of-magnitude comparisons but can include allocator and runtime effects
outside the index itself.

## Comparison Scope

The Pareto harness includes USearch `f16` alongside `i8` and `f32`. The installed
USearch npm package (`2.25.1`) exposes native Node prebuilds but does not ship a
USearch WASM package entrypoint. Custom local USearch WASM builds are tracked in
the root README and may appear in newer ad hoc result sets, but the release
tables above keep the cross-library comparisons to artifacts captured in each
CSV.

Repeated 1M-vector runs on the current laptop show substantial system noise at
some points (maximum observed QPS coefficient of variation up to roughly 56%).
Treat differences below about 10% as inconclusive unless they reproduce on a
quieter machine or across additional repetitions.
