# POC measurement kit

Everything the docs-search POC needs beyond what the example already ships:
an instrumented query harness, a cost model, and a report template. The
build/host/reader pieces are the parent example (`compile.mjs`, `serve.mjs`,
`pancake-file-reader.mjs`); building from your own docs is
`pikelet` (`--runtime artifact --mode student`, or the
Docusaurus plugin's `completeProfile` mode — see `docs-site/` in the repo
root for a working config).

```bash
# smoke it on the shipped example artifact (goldens come from the file itself)
cd examples/05-one-file-search
node compile.mjs
node poc/harness.mjs pancake-docs.pancake --k 5 --out poc/results.json

# over HTTP range requests (real egress accounting)
node serve.mjs &
node poc/harness.mjs http://127.0.0.1:8790/pancake-docs.pancake --k 5

# your artifact, your queries, with ground truth for recall@10
node poc/harness.mjs /path/to/your.pancake --queries queries.json --k 10 --out poc/results.json

# cost estimate from the measurements
node poc/cost.mjs --results poc/results.json --queries-per-month 1000000 \
    --current-monthly 500 --integration-cost 8000
```

`queries.json` is either `["query text", ...]` or
`[{ "text": "...", "ids": [12, 40, 7] }, ...]` where `ids` are ground-truth
top-k record ids (build them brute-force over the same corpus — see
`../test-wiki.mjs` for the pattern). With no `--queries`, the harness uses
the artifact's embedded evaluation goldens and checks their match-quality
labels reproduce.

Kind-2 artifacts (declared external encoder) need `options.encodeQuery` and
are out of the harness's scope; use kind-1 (student) or kind-3 (inline
transformer) artifacts, which query self-contained.

Write up the numbers in `REPORT_TEMPLATE.md`.
