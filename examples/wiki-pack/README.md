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
- **Abstention**: AUC 1.0000 on the calibration set; 14/14 golden probes
  pass in Node and in Chromium.

## Pipeline

Everything under `data*/` is generated; the scripts are the pack compiler.
Requirements: Python 3.10+ with `torch` (CUDA strongly recommended),
`transformers`, `datasets`, `numpy`; Node 20+. Install the JS deps with:

```bash
ONNXRUNTIME_NODE_INSTALL_CUDA=skip npm install
```

Then, in order:

```bash
# 1. Stream the dump, chunk (~800 chars, title-prefixed), embed fp32 on GPU.
#    ~103 min on an RTX 2060 for the full corpus; --limit 1000 for a smoke run.
python3 embed_corpus.py --out data-full

# 2. Cluster the embeddings (k-means, k=1024, nearest-centroid chain order)
#    and write the layout permutation. Also prints the coalescing simulation
#    that motivates the layout.
python3 eval_queries.py --data data-full        # eval set + exact ground truth
python3 layout_sim.py                            # -> data-full/layout-perm.npy

# 3. Reorder the corpus by cluster. Chunk ids are positional, so this is the
#    entire layout change — vectors, corpus text, and offsets stay in lockstep.
python3 permute_corpus.py                        # -> data-perm/

# 4. Build the pack: cosine u8 HNSW index -> sketch artifact (192-dim 4-bit
#    sketches, 2:1 pooling) + corpus.bin + corpus-offsets.u32.  ~10 min.
node build_pack.mjs data-perm

# 5. Evaluate: recall vs exact float truth at the shipped settings.
python3 eval_queries.py --data data-perm         # ground truth for the permuted ids
node eval_recall.mjs data-perm 200

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

## Licensing

Corpus text: [Simple English Wikipedia](https://simple.wikipedia.org)
(`wikimedia/wikipedia`, `20231101.simple`), CC BY-SA 4.0 — the demo page
and any redistribution of `corpus.bin` must keep attribution. Encoder:
[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
Apache-2.0.
