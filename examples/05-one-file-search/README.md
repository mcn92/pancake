# One-file search

A search engine as a single static file. This example compiles the five
Search Artifact components — corpus, index, encoder, evaluation,
calibration — into one content-addressed `.pancake`
(`spec/COMPLETE_PROFILE.md`), and serves natural-language queries from it
in Node and in the browser over HTTP range requests, with no backend and
no WASM in the query path.

```bash
node compile.mjs                      # 03's assets -> pancake-docs.pancake (1.4 MiB)
node compile.mjs --inspect pancake-docs.pancake
node test-file.mjs                    # the file proves itself (goldens live inside it)
npx vite build web                    # build the browser page
node serve.mjs                        # http://127.0.0.1:8790 — search in the browser
node test-browser.mjs                 # Chromium acceptance (needs playwright chromium)
```

```js
import { openPancakeFile } from './pancake-file-reader.mjs';
const search = await openPancakeFile('pancake-docs.pancake');   // or a range source
const out = await search.query('how do workers restore snapshots');
// { matchQuality: 'strong', confidence: 0.94, results: [{ title, text, sourcePath, ... }] }
```

## Files

- `compile.mjs` — assembles a `.pancake` per the spec: 64 B header,
  canonical-JSON manifest (its SHA-256 is the artifact identity), segment
  table, and four segments — the index is an embedded `.pancake-sketch`,
  the corpus is an offsets table + JSON records (one range read per
  hydration), encoder+calibration share one query-interpretation segment,
  and the evaluation segment carries the golden queries. `--inspect`
  verifies every digest.
- `pancake-file-reader.mjs` — the one-file reader. Environment-neutral:
  Node opens a path, the browser passes an HTTP range source. Verifies the
  manifest identity and eager segments at open; the sketch tier and corpus
  records stay lazy. Pure JS query path (the WASM engine is compile-time
  only).
- `web/` — the browser host: an input box over the reader, showing per-query
  range requests and bytes. `serve.mjs` is the entire hosting requirement:
  static files + `Range` support.
- `search-reader.mjs`, `demo.mjs`, `test.mjs` — the original composition
  spike over 03's six separate asset files, kept as the reference the
  one-file reader is tested against.
- `abstention.mjs` — calibration scoring shared by both readers (extracted
  from 03's worker.js).

## What the acceptance tests establish

- `test-file.mjs` (15 checks): the golden queries come from the file's own
  evaluation segment — the artifact carries its conformance fixtures — and
  all 10 reproduce their labels; hydrated results byte-match the source
  corpus; the one-file reader returns identical results to the six-asset
  spike; a tampered segment fails its digest on read.
- `test-browser.mjs` (6 checks, real Chromium): the page opens the file
  over HTTP ranges, verifies the resident hash, answers a docs question
  with hydrated results and calibrated confidence, abstains on an
  out-of-domain query, and provably fetches by range rather than
  downloading the file.

## Numbers (docs corpus: 208 chunks, 384D)

- File: 1.40 MiB total — encoder 1.15 MiB, corpus 147 KiB, index 99 KiB,
  evaluation 4.5 KiB.
- Open: ~23 KiB resident (sketch tier + corpus offsets) plus the encoder;
  browser cold open is a handful of range requests.
- Query: sketch scan + one parallel rerank round + one range read per
  hydrated result; single-digit-millisecond embeds.

## Still out of scope

Promotion into the published `pancake-wasm` package, the
`create-pancake-search` compile frontend, per-chunk range verification
(spec section 6), and large-corpus compilation (the wiki pack is the
intended flagship).
