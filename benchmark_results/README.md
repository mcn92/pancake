# Benchmark Results

This directory contains the latest committed Pareto-frontier benchmark outputs.
The generated ground-truth cache files under `benchmark_results/cache/` are
local cache artifacts and are intentionally not part of the committed result
set.

## Run Environment

- Host: AMD Ryzen 9 4900HS with Radeon Graphics, 8 logical CPUs
- RAM: 12.5 GB
- Platform: linux/x64
- Node: v22.19.0
- Benchmark parameters: `k=10`, `M=8`, `efConstruction=50`,
  `efSearch=10,20,40,60,80,100,150,200,300,500,800`, 3 repetitions

## Current Result Sets

| Dataset | Vectors | Dim | Metric | Queries | Result prefix |
|:--------|--------:|----:|:-------|--------:|:--------------|
| DBpedia-50K | 50,000 | 1536 | L2 | 1,000 | `pareto_dbpedia_2026-07-07T19-04-37-320Z` |
| NYTimes-256 | 289,761 | 256 | cosine | 999 | `pareto_nytimes_2026-07-07T19-19-12-061Z` |
| SIFT-1M | 1,000,000 | 128 | L2 | 1,000 | `pareto_sift_2026-07-07T19-37-32-563Z` |
| GloVe-100 | 1,183,514 | 100 | cosine | 1,000 | `pareto_glove_2026-07-07T21-33-55-060Z` |

Each result set includes:

- `.csv`: every measured backend and `efSearch` point
- `_frontier.csv`: non-dominated QPS/recall points
- `_equalrecall.csv`: interpolated equal-recall comparison table
- `.json`: full structured results with system and memory metadata
- `.log`: console output from the benchmark run
- `.png`: Pareto frontier plot

## DBpedia-50K (50,000 x 1536D, L2)

| Backend | Build | Memory | ef=100 recall | ef=100 QPS | ef=800 recall | ef=800 QPS |
|:--|--:|--:|--:|--:|--:|--:|
| pancake-wasm-int8 | 30.6s | 80.3 MB | 95.11% | 1,704 | 97.45% | 303 |
| pancake-wasm-fp32 | 56.6s | 296.3 MB | 96.76% | 1,244 | 99.65% | 222 |
| pancake-native-int8 | 19.7s | 80.3 MB | 95.12% | 2,008 | 97.46% | 387 |
| pancake-native-fp32 | 38.9s | 296.3 MB | 96.76% | 1,352 | 99.65% | 287 |
| usearch-int8 | 12.2s | 77.3 MB (file size) | 86.98% | 2,480 | 89.53% | 433 |
| usearch-fp32 | 34.3s | 297.0 MB (file size) | 95.23% | 1,140 | 99.38% | 217 |
| hnswlib-fp32 | 32.3s | 313.0 MB (RSS delta) | 95.10% | 1,399 | 99.37% | 298 |

## NYTimes-256 (289,761 x 256D, cosine)

| Backend | Build | Memory | ef=100 recall | ef=100 QPS | ef=800 recall | ef=800 QPS |
|:--|--:|--:|--:|--:|--:|--:|
| pancake-wasm-int8 | 76.7s | 111.4 MB | 64.56% | 2,908 | 81.78% | 484 |
| pancake-wasm-fp32 | 107.4s | 302.5 MB | 63.48% | 1,825 | 81.84% | 359 |
| pancake-native-int8 | 56.6s | 111.4 MB | 64.38% | 3,462 | 81.51% | 590 |
| pancake-native-fp32 | 80.7s | 302.5 MB | 63.65% | 2,101 | 81.88% | 433 |
| usearch-int8 | 58.8s | 94.2 MB (file size) | 58.71% | 2,541 | 76.09% | 460 |
| usearch-fp32 | 94.3s | 306.5 MB (file size) | 59.47% | 1,791 | 78.34% | 310 |
| hnswlib-fp32 | 62.1s | 320.3 MB (RSS delta) | 59.96% | 3,415 | 78.46% | 628 |

## SIFT-1M (1,000,000 x 128D, L2)

| Backend | Build | Memory | ef=100 recall | ef=100 QPS | ef=800 recall | ef=800 QPS |
|:--|--:|--:|--:|--:|--:|--:|
| pancake-wasm-int8 | 214.8s | 262.4 MB | 91.52% | 2,972 | 98.20% | 488 |
| pancake-wasm-fp32 | 250.5s | 555.6 MB | 91.86% | 2,502 | 98.81% | 414 |
| pancake-native-int8 | 176.4s | 262.4 MB | 91.35% | 2,834 | 98.17% | 585 |
| pancake-native-fp32 | 174.7s | 555.6 MB | 91.86% | 3,320 | 98.81% | 580 |
| usearch-int8 | 137.0s | 203.2 MB (file size) | 84.63% | 3,551 | 90.89% | 628 |
| usearch-fp32 | 230.9s | 569.4 MB (file size) | 90.71% | 2,169 | 98.62% | 348 |
| hnswlib-fp32 | 153.4s | 614.0 MB (RSS delta) | 90.76% | 3,654 | 99.13% | 672 |

## GloVe-100 (1,183,514 x 100D, cosine)

| Backend | Build | Memory | ef=100 recall | ef=100 QPS | ef=800 recall | ef=800 QPS |
|:--|--:|--:|--:|--:|--:|--:|
| pancake-wasm-int8 | 307.3s | 278.9 MB | 62.42% | 2,455 | 81.22% | 427 |
| pancake-wasm-fp32 | 348.4s | 531.1 MB | 63.75% | 2,074 | 82.20% | 364 |
| pancake-native-int8 | 253.4s | 278.9 MB | 62.90% | 2,798 | 81.35% | 500 |
| pancake-native-fp32 | 240.1s | 531.1 MB | 63.87% | 2,979 | 82.65% | 517 |
| usearch-int8 | 316.2s | 208.9 MB (file size) | 58.70% | 1,830 | 77.29% | 323 |
| usearch-fp32 | 411.8s | 547.5 MB (file size) | 60.14% | 1,377 | 79.94% | 255 |
| hnswlib-fp32 | 212.8s | 586.4 MB (RSS delta) | 58.65% | 3,828 | 79.89% | 664 |

## Memory Notes

Pancake memory is reported by the index. USearch does not expose an equivalent
runtime memory counter in this harness, so its table value is the saved index
file size. hnswlib is reported as process RSS delta. RSS deltas are useful for
order-of-magnitude comparisons but can include allocator and runtime effects
outside the index itself.
