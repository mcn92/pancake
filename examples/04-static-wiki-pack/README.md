# Wiki Knowledge Pack

All of Simple English Wikipedia as one searchable Pancake artifact: ~242k
articles chunked into 456,153 passages, embedded with MiniLM, compiled into a
range-readable sketch artifact, and queried entirely client-side — the
encoder, the WASM scan, the exact rerank, and calibrated abstention all run
in the browser tab against static files. No search server, no embedding API,
no third-party requests.

Measured on the built pack (2026-08-03):

- **Pack**: 222.6 MB sketch artifact (47.4 MB resident tier, hash-verified
  at open) + 332 MB corpus + 1.8 MB offsets + 45 MB fp16 encoder.
- **Recall@10 vs exact float brute force** (200 queries): 95.65% at the
  shipped rerank C=200, 96.95% at C=300 — hand-written natural questions
  100% at both.
- **Browser query** (Chromium, local HTTP/2): ~27 ms encode + ~216 ms
  search + ~35 ms hydrate, ~66 range requests / ~390 KiB per query.
- **Cold boot** is the honest cost: ~93 MB before the first query (47 MB
  resident tier + 45 MB fp16 encoder + ONNX runtime), all
  browser-cacheable. Measured on the live Pages/R2 deployment
  (pancake-wiki-pack-demo.pages.dev, ~100 Mbit/s client): ~8 s cold
  time-to-first-query (encoder and resident tier load in parallel), ~2 s on
  repeat visits (service-worker cache; the residual is WASM compile). The economics favor
  repeat-query contexts — a docs site, an installed pack — over drive-by
  pageloads.
- **Live queries** (same deployment): 0.6-1.5 s search on a cold edge,
  ~20 ms on browser-warm repeats. Two deployment lessons are baked into
  the code: the Pages Function edge-caches every range (identical reads —
  the resident open, the encoder — skip R2 after first touch per colo),
  and every range read carries its range in the query string, because
  Chromium serializes concurrent fetches of one cacheable URL on its
  HTTP-cache entry lock — same-URL range reads ran one at a time
  (5.7-18.6 s per query) until the URLs were made distinct. The remaining
  gap to the ~280 ms localhost number is per-request edge latency; the
  durable fix is fewer, larger reads (see the residency note below).
- **Abstention**: Calibrated from a generated held-out title shard rather than
  hand-written questions. The script holds out about 2% of titles by chunk
  count, builds a reduced pack without them, treats title questions from that
  shard as hard in-domain negatives, verifies retained-title positives by
  source-title retrieval, adds synthetic out-of-vocabulary gibberish negatives,
  adds generated real-English negatives from an unrelated title bank,
  mines a weak band from retained-title queries whose source lands at rank
  5-50, and reports per-negative-kind AUC plus reduced-vs-full shift before
  shipping the scorer with the full pack. Negative rows that retrieve as close
  as a median verified positive are dropped as semantic overlap.

## Pipeline

Everything under `data*/` is generated; the scripts are the pack compiler.
Requirements: Python 3.10+ with `torch` (CUDA strongly recommended),
`transformers`, `datasets`, `numpy`; Node 20+. Install the JS deps with:

```bash
ONNXRUNTIME_NODE_INSTALL_CUDA=skip npm install
```

Then, in order:

```bash
# 1. Read JSONL docs, chunk (~800 chars, title-prefixed), embed fp32 on GPU.
#    Input rows are { "id": "...", "title": "...", "text": "...", "url": "..." }.
python3 embed_corpus.py --input sample-corpus.jsonl --out data-sample

#    Or use the Wikipedia adapter. ~103 min on an RTX 2060 for the full corpus;
#    --limit 1000 for a smoke run.
python3 embed_corpus.py --limit 1000 --out data-full
python3 embed_corpus.py --out data-full

# 2. Cluster the embeddings (k-means, k=1024, nearest-centroid chain order)
#    and write the layout permutation. Also prints the coalescing simulation
#    that motivates the layout.
python3 eval_queries.py --data data-full        # wiki eval set + exact ground truth
python3 layout_sim.py --data data-full          # -> data-full/layout-perm.npy

#    For a non-Wikipedia pack, use title-sampled eval only unless you provide
#    your own corpus-specific hand probes.
python3 eval_queries.py --data data-sample --sampled 5 --no-hand
python3 layout_sim.py --data data-sample --k 5

# 3. Reorder the corpus by cluster. Chunk ids are positional, so this is the
#    entire layout change — vectors, corpus text, and offsets stay in lockstep.
python3 permute_corpus.py --src data-full --out data-perm
python3 permute_corpus.py --src data-sample --out data-sample-perm

# 4. Build the pack: cosine u8 HNSW index -> sketch artifact (192-dim 4-bit
#    sketches, 2:1 pooling) + corpus.bin + corpus-offsets.u32.  ~10 min.
node build_pack.mjs data-perm
node build_pack.mjs data-sample-perm

# 5. Evaluate: recall vs exact float truth at the shipped settings.
python3 eval_queries.py --data data-perm         # ground truth for the permuted ids
node eval_recall.mjs data-perm 200
python3 eval_queries.py --data data-sample-perm --sampled 5 --no-hand
node eval_recall.mjs data-sample-perm 200

# 6. Calibrate abstention: builds a reduced held-out calibration pack, fits the
#    scorer, writes wiki-abstention.json, the vocabulary bloom, and generated
#    probes; prints the confusion table and reduced-vs-full shift.
node calibrate_abstention.mjs data-perm
```

Step 6 is intentionally not a quick metadata pass: it writes a reduced copy of
`corpus.jsonl`/`vectors.f32`, rebuilds a second pack in-process, and embeds the
generated fit queries. On the 456k-chunk wiki pack, expect another pack-build-
sized job and peak memory that includes the source vectors file.

The default abstention fit is agnostic and generated, but it is not a substitute
for corpus-owner judgment. For a pack users will rely on, add owner probes for
in-domain answerable queries and hard out-of-domain queries; those probes are
the per-corpus ground truth that catches over-abstention and false confidence.
The built-in foreign-title bank is used for fit pressure and diagnostics, not
as a hard generated probe set, because some real-English "foreign" queries are
semantically close enough that retrieval cannot separate them without losing
verified positives.

Calibration knobs:

```bash
HOLDOUT_RATE=0.02 FIT_QUERIES=160 HOLDOUT_QUERIES=160 \
FOREIGN_QUERIES=80 GIBBERISH_QUERIES=80 WEAK_QUERIES=80 \
ABSTENTION_SEED=424242 \
  node calibrate_abstention.mjs data-perm
```

Pack owners can add corpus-specific correctness probes without changing the
default agnostic fit:

```bash
node calibrate_abstention.mjs data-perm --probes probes.json
```

`probes.json` is an array of `{ "text": "...", "expect": "answer" }` rows;
`expect` may also be an array such as `["weak", "answer"]`.

Pack owners can also replace the default foreign-title bank:

```bash
node calibrate_abstention.mjs data-perm --foreign-titles foreign-titles.txt
```

`foreign-titles.txt` can be newline-delimited text; `.json` files are parsed as
an array of title strings. The calibrator logs dropped foreign entries so domain
overlap is visible instead of silently poisoning labels.

`query_pack.mjs` is a CLI smoke test of the full flow (JS-side encoding,
search, hydration) against whichever data directory you pass it.

## Demo page

`web/` is a Vite page that runs the same flow in the browser, with a network
meter and per-query timings. Build and serve locally:

```bash
npx vite build web
node web/serve2.mjs          # HTTP/2 + self-signed TLS on :8932; needs web/.cert
PACK_DATA=data-perm node web/serve2.mjs   # choose the data directory
```

`web/serve.mjs` is the plain-HTTP variant. Both serve real 206 range
responses; the model and ONNX runtime are served from the same origin
(`/models/`, `/ort/`), so the page makes no external requests. Generate the
local cert once with:

```bash
mkdir -p web/.cert
openssl req -x509 -newkey rsa:2048 -keyout web/.cert/key.pem \
  -out web/.cert/cert.pem -days 30 -nodes -subj "/CN=localhost"
```

Headless hooks for measurement: `window.__bench([queries])` returns boot and
per-query timings; `window.__probes()` runs the committed abstention golden
probes in the live page. Search knobs via URL: `?C=200&gap=16384&p=32`.

## Design notes (the measured reasons things are the way they are)

- **fp16 encoder, not q8.** fp16 is end-to-end identical to fp32 (recall
  unchanged at every rerank depth) at half the download; q8 costs 3.9 recall
  points. Corpus and queries must share one embedding space, so the exact
  weights ship beside the pack.
- **2:1 pooling, not 4:1.** 96-dim sketches (4:1) capture only 84.8% of the
  exact top-10 even at C=600; 192-dim 4-bit captures 95.9% at C=200. Bit
  depth is irrelevant at both widths — pooling is the lever.
- **Cluster-ordered layout.** Browsers pay ~4 ms of overhead per fetch, so
  scattered 384-byte rerank reads cost seconds regardless of bandwidth or
  parallelism. Writing chunks in k-means cluster order makes a query's
  candidates physically adjacent: ~66 coalesced requests instead of ~430,
  with no format changes — the layout is a build-time permutation.
- **Abstention is pack-local, not wiki-specific.** The scorer is still just
  logistic regression over top-distance, margin, mean-of-10, and known-
  vocabulary fraction from a 256 KB bloom filter, but the labels now come from
  the pack itself. Calibration removes a deterministic title shard, generates
  questions from those absent titles as hard negatives, verifies positives from
  retained titles, adds generated real-English foreign-title negatives,
  synthesizes gibberish from tokens outside the vocabulary bloom, mines weak
  examples from rank-5-to-50 source-title hits, then measures how much the
  fitted scores move when evaluated against the final full pack. Gibberish rows
  are downweighted in both standardization and fit, and AUC is reported by
  negative kind so the easy vocabulary failures cannot hide the harder in-
  vocabulary cases. Negative labels are verified against this pack: title
  identity is only a proxy, so held-out and foreign-bank negatives are dropped
  when their retrieval distance is as strong as a median verified positive.
  Holding out by title rather than by individual chunk matters: if another
  chunk with the same title remains, the negative is no longer provably absent.
- **The query path never touches the HNSW graph.** The sketch profile is
  edge-free — resident scan plus one fetch round beats graph traversal over
  ranged reads (one round-trip per hop) by design. The pipeline's index
  build exists only because the pack builder consumes engine snapshots; a
  direct vectors-to-pack path would drop that ~10-minute step.
- **Residency scales linearly** at ~104 bytes/chunk (96 B sketch + 8 B
  scale/offset): this corpus costs 47 MB resident; full English Wikipedia
  (~25M chunks) would cost ~2.5 GB, out of scope for a tab. Bounded corpora
  are the point of the pack framing; a two-level tier (resident centroids,
  ranged per-cluster sketches) is the plausible path past the ceiling.

## Licensing

Corpus text: [Simple English Wikipedia](https://simple.wikipedia.org)
(`wikimedia/wikipedia`, `20231101.simple`), CC BY-SA 4.0 — the demo page
and any redistribution of `corpus.bin` must keep attribution. Encoder:
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
Apache-2.0.
