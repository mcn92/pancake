# Relevance regression suite

Labeled query sets with expected sections, run against compiled artifacts
with `scripts/bakeoff-retrieval.mjs`. This is the permanent record of what
"retrieval works" means: when ranking, chunking, calibration, or the
encoder changes, rerun the suite and compare against the baselines
committed beside each query set.

## Query sets

- `nodeapi-queries.json` — nodejs.org/api, crawled with
  `compile --source https://nodejs.org/api/ --max-pages 30`. A real,
  substantial docsite: ~2,000 sections, dense API identifiers, authored
  anchor ids. Mixes natural-language questions, API identifiers
  (`fs.readFileSync`), error codes, CLI flags, and HTTP-ish exact
  lookups.
- `nodeapi-baseline.json` — the metrics recorded when the set was
  authored, with the artifact identity they were measured against.

## Running

```bash
# recompile the corpus (network; page content drifts over time)
npx pikelet compile --source https://nodejs.org/api/ \
  --max-pages 30 --out nodeapi.pikelet

# measure all three retrieval modes
node scripts/bakeoff-retrieval.mjs nodeapi.pikelet \
  test/relevance/nodeapi-queries.json
```

Not part of `npm test`: it needs the network and ~20 minutes of local
embedding. Treat a drop against the committed baseline as a regression to
explain, not noise to re-record — the corpus can drift when the site
publishes new docs, so re-author expectations only when a target section
verifiably moved or was renamed upstream.
