# Pancake docs-search POC report

Date: <!-- -->
Authors: <!-- -->
Verdict: **GO / NO-GO / ITERATE** — <!-- one sentence -->

## Setup

| | |
|---|---|
| Corpus | <!-- source, page count, chunk count --> |
| Artifact | <!-- kind (1 student-inline / 2 declared / 3 inline-transformer), file size, identity prefix --> |
| Build | <!-- command used, build wall time --> |
| Host | <!-- local serve.mjs / S3 / CDN, region --> |
| Query set | <!-- N queries, where they came from --> |
| Harness | `poc/harness.mjs` <!-- exact invocation --> |

## Quality

Ground truth: <!-- brute-force over the same corpus / current managed results -->

| Metric | Measured | Target | Pass? |
|---|---|---|---|
| recall@10 | | >= 0.70 | |
| golden match-quality labels reproduced | | all | |
| abstention rate (matchQuality = none) | | | |

Failure modes observed: <!-- query families that miss, and why -->

## Performance

From `harness.mjs --out results.json` (attach the file).

| Metric | First pass (cold reader) | Repeat pass (warm reader) | Target |
|---|---|---|---|
| latency median (ms) | | | <= 250 warm |
| latency p95 (ms) | | | |
| bytes/query mean (KiB) | | | <= 100 |
| bytes/query p95 (KiB) | | | |
| range reads/query median | | | |

Open cost: <!-- ms, range reads, KiB from the harness "open:" line -->
CDN cold-cache latency (if measured against a real CDN with cache flush): <!-- -->

## Cost

From `poc/cost.mjs` (paste the invocation and output).

| | $/month |
|---|---|
| Pancake total (egress + storage + requests + builds + ops) | |
| Current managed spend | |
| Savings | |
| Breakeven on integration cost | <!-- months --> |

Assumptions: <!-- queries/month, pricing source, rebuild cadence -->

## Risks and next steps

- <!-- e.g. rebuild cadence vs docs update frequency; incremental builds needed? -->
- <!-- e.g. cold-cache prewarming, sketch tuning if bytes/query high -->
- <!-- e.g. kind-1 vs kind-3 tradeoff if quality short -->
