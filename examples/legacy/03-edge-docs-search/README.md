# Zero-dependency distilled docs search

This is the Worker-backed Pancake docs-search demo. The deployed Cloudflare
Worker restores a bundled quantized Pancake snapshot, embeds each query locally,
and searches the in-memory index. The public UI is served separately from
Cloudflare Pages under `pages-ui/`.

Hosting status: the Worker is deployed in private mode. `wrangler.toml` sets
`workers_dev = false`, so there is no public `workers.dev` endpoint — requests
to one return Cloudflare error 1042. The Pages UI at
`https://pancake-docs-search.pages.dev` remains live and expects the operator
to supply a Worker API base URL (it stores the value locally in the browser).
To host your own public endpoint, deploy with `workers_dev = true` (or a
custom route) and either set the `DEMO_SEARCH_KEY` secret or deploy
`PRIVATE_SEARCH=0` to open `/search`.

The Worker has no service bindings and makes no outbound requests. In the
private-mode configuration, `/search` requires the `DEMO_SEARCH_KEY` secret
unless `PRIVATE_SEARCH=0` is deployed.

There is no embedding API and no runtime ML framework. The complete query path is:

```text
text
  -> hashed word and character n-grams
  -> 1.08 MB int8 distilled student
  -> normalized 384D query vector
  -> query-quality score from bundled calibration data
  -> Pancake WASM
  -> matching documentation chunks
```

The large teacher model is an offline build dependency. It is never bundled
with the Worker or contacted at query time.

## What the demo proves

- Pancake runs inside a Cloudflare Worker without a native addon.
- A snapshot can be restored directly from a Worker data module on cold start.
- Query embedding and vector retrieval run without outbound API requests or storage bindings.
- The embedding model is application data, not a Pancake dependency.
- Embedding, Pancake search, and cold-restore latency are reported separately.
- `efSearch` and source filters are applied per query through the public API.
- Off-domain or noise queries are visibly marked as weak/no-match without a second retrieval pass.

This is a domain-specific encoder, not a general-purpose replacement for a
large embedding model. It is deliberately small enough to make the entire demo
self-contained.

## Distillation design

Documents are embedded offline with `sentence-transformers/all-MiniLM-L6-v2`,
pinned to revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
The student learns to place queries in that same 384D space. Its loss combines:

- cosine imitation of the teacher query vector; and
- retrieval-distribution imitation against every documentation chunk.

The runtime student is:

```text
8,192 hashed n-gram buckets x 128 hidden dimensions
  -> mean pool
  -> tanh
  -> 128 x 384 projection
  -> L2 normalization
```

Both parameter matrices use symmetric per-row int8 quantization. The exported
`PSTU` artifact is approximately 1.08 MB.

On the current 208-chunk corpus, the selected model was evaluated on 226 query
forms that were not used for training or model selection:

- mean cosine to teacher query vectors: `0.915`
- teacher top-result agreement: `81.9%`
- overlap with the teacher's top five: `82.8%`
- JavaScript/Python exported-model parity: `226/226` top results

Validation queries are separate from this final test set and select the early-
stopping checkpoint. The full per-query results are written to
`student-evaluation.json` during training.

## Query abstention

The demo also exports `docs-abstention.json`, a small application-layer
confidence model trained after the student is quantized. It combines top-hit
distance, top-5 margin, pre-normalization query magnitude, and corpus bucket
coverage plus a hidden-vector probe into `match_quality`:

- `strong` returns normal results.
- `weak` returns results with a no-strong-match banner.
- `none` returns an empty result list for hard-floor noise inputs.
- `unscored` is used only when an older asset bundle has no abstention block.

The committed calibration catches the motivating collision query
`banana pikelet recipe` as off-domain, while obvious noise such as emoji-only
input is refused as `none`. Current heldout abstention metrics are recorded in
`docs-manifest.json`; the model's pooled OOD AUC is `0.986`, collision catch
rate is `1.000`, shuffled-negative catch rate is `0.727`, nonsense refusal rate
is `1.000`, and test false-abstain rate is `0.018` against a `0.02` validation
target.

## Runtime assets

The build produces five runtime assets that Wrangler bundles into the Worker:

- `docs-index.bin` — standard Pancake snapshot
- `docs-corpus.json` — result metadata and previews
- `docs-manifest.json` — dimensions, construction config, hashes, and evaluation summary
- `docs-student.bin` — quantized query encoder
- `docs-abstention.json` — query-quality model, thresholds, and corpus bucket tables

`docs-student-evaluation.json` is also produced for inspection but is not
imported by the Worker.

The current artifacts are committed under `assets/`, so deployment does not
require Python, PyTorch, or retraining. The build instructions below regenerate
them when the documentation or encoder changes.

## Build the demo

Build Pancake at the repository root first:

```bash
npm run build:all
```

Extract the documentation corpus:

```bash
node examples/legacy/03-edge-docs-search/build_demo.mjs \
  --out /tmp/pancake-docs-demo \
  --corpus-only
```

Create an isolated training environment. These packages are offline build tools,
not runtime dependencies:

```bash
python3 -m venv /tmp/pancake-student-venv
/tmp/pancake-student-venv/bin/pip install \
  -r examples/legacy/03-edge-docs-search/requirements-train.txt
```

Train and export the student. The first run downloads the teacher checkpoint to
the selected Hugging Face cache:

```bash
HF_HOME=/tmp/pancake-hf \
/tmp/pancake-student-venv/bin/python \
  examples/legacy/03-edge-docs-search/train_student.py \
  --corpus /tmp/pancake-docs-demo/docs-corpus.json \
  --out /tmp/pancake-student
```

Verify that plain JavaScript reproduces the exported Python model:

```bash
node examples/legacy/03-edge-docs-search/verify_student.mjs \
  --student-dir /tmp/pancake-student
```

Build the Pancake snapshot and bundled asset directory:

```bash
node examples/legacy/03-edge-docs-search/build_demo.mjs \
  --out examples/legacy/03-edge-docs-search/assets \
  --student-dir /tmp/pancake-student
```

The build prints preview results for the sample queries before deployment.

Build and exercise the actual workerd bundle with an in-process Worker and its
bundled binary data modules:

```bash
cd examples/legacy/03-edge-docs-search
npx wrangler deploy --dry-run \
  --outdir ../../.tmp-test-work/student-worker-distilled
cd ../..
node examples/legacy/03-edge-docs-search/test_worker.mjs
```

## Run locally

```bash
cd examples/legacy/03-edge-docs-search
npx wrangler dev --port 8787
```

Open `http://localhost:8787/`. The local Worker serves its built-in UI and API
from the same origin.

If Wrangler cannot write logs in your environment, redirect its config path:

```bash
XDG_CONFIG_HOME=/tmp/wrangler-config npx wrangler dev --port 8787
```

## Deploy the Worker

Deploy from this directory:

```bash
npx wrangler deploy
```

The committed configuration sets:

```toml
READ_ONLY = "1"
ALLOWED_ORIGIN = "https://pancake-docs-search.pages.dev"
DISABLE_SEARCH = "0"
PRIVATE_SEARCH = "1"
```

`DEMO_SEARCH_KEY` is a Cloudflare Worker secret, not a repo file:

```bash
npx wrangler secret put DEMO_SEARCH_KEY
```

Mode switches:

- Private search: `PRIVATE_SEARCH = "1"` and `DEMO_SEARCH_KEY` set.
- Public search: `PRIVATE_SEARCH = "0"`.
- Fully disabled search: `DISABLE_SEARCH = "1"`.

Redeploy after changing `wrangler.toml`.

## Deploy the Pages UI

The static Pages UI lives in `examples/legacy/03-edge-docs-search/pages-ui/`.

From the repo root:

```bash
npm run build:pages-demo
npx wrangler pages deploy examples/legacy/03-edge-docs-search/pages-ui/dist \
  --project-name pancake-docs-search
```

The Pages UI stores the Worker API URL in browser `localStorage` and keeps the
optional access key in tab-scoped `sessionStorage` after the user clicks Save.
The deployed static config currently does not bake in the Worker URL or demo key.

## Endpoints

- `GET /` — Worker-local interactive webpage
- `GET /health` — public liveness and coarse mode status
- `GET /readiness` — authenticated bundled-asset metadata
- `POST /search` — `{ query, k?, ef?, source? }`
- `POST /reset_cache` — authenticated and disabled in read-only deployments

Search accepts only POST bodies so query text and demo keys are not placed in
URLs. Default responses do not echo the query and report only user-facing results,
coarse `embedding_ms`, `search_ms`, `restore_ms`, `match_quality`, optional
`confidence`, and result metadata. Set `DEBUG_TELEMETRY=1` only for private
diagnostic deployments if you need cache state, encoder metadata, or detailed
timing fields.

## Security and deployment boundary

The bundled snapshot is immutable deployment data; the in-memory index is a
disposable query-optimized copy. Concurrent cold requests share one restore
promise. Snapshot and student-model byte limits are checked before restoration,
and the student artifact is checked against the manifest SHA-256 before use.

The demo intentionally does not provide live index mutation. Applications that
need authoritative writes should build and publish new snapshots or place their
write coordination in an appropriate durable system.

The private demo key is a shared bearer secret. Anyone with the Worker URL and
key can search. CORS restricts browser calls to the Pages origin, but CORS is
not authentication; non-browser clients can call the Worker URL directly.
