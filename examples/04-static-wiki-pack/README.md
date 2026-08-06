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
- **Abstention**: AUC 1.0000 on the 122-point fit set (96 answerable, 26
  unanswerable — a small sample, so read the clean score gap between the
  classes, not the headline AUC); 14/14 golden probes pass in Node and in
  Chromium.

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

# 6. Calibrate abstention: fits the scorer, writes wiki-abstention.json,
#    the vocabulary bloom, and the golden probes; prints the confusion table.
node calibrate_abstention.mjs data-perm
```

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
- **Abstention verdicts measure match strength, not intent.** The scorer
  (logistic over top-distance, margin, mean-of-10, and known-vocabulary
  fraction from a 256 KB bloom filter) cleanly separates "nothing useful
  here" from real matches. It deliberately does not try to detect temporal
  or intent gaps — "who won the 2026 world cup" surfaces the real pre-event
  article and is scored as the strong match it is. The vocabulary feature is
  what catches gibberish, which lands at cosine distances inside the
  answerable band and is invisible to distance alone.
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
