# Pikelet

Pikelet compiles a document corpus into a single file that answers
search queries on its own. No database, no server, no embedding
service. The file carries its index, its text, its query encoder, and a
confidence model. Query it straight off a CDN or a local disk over
byte-range reads.

> Renamed from **Pancake** (2026-09). Same wire format. Existing
> artifacts and their identities remain valid; `pancake-wasm` and
> `create-pancake-search` are deprecated pointers to `pikelet-wasm` and
> `pikelet`.

Looking for the [Pikelet programming language](https://github.com/pikelet-lang/pikelet)?
That's Brendan Zabarauskas's dependently-typed systems language. An
unrelated project that had the name first.

## Quick start

Compile a folder of docs into one `.pikelet` file:

```bash
npx pikelet compile --source ./docs --out search.pikelet
```

Query it:

```js
import { openPikeletFile } from 'pikelet-wasm/complete';

const search = await openPikeletFile('search.pikelet');
const out = await search.query('how do I get started', { k: 5 });

console.log(out.matchQuality, out.results[0]?.title);
await search.close();
```

`matchQuality` is `'strong'`, `'weak'`, or `'none'`. A pack that doesn't
contain the answer says so instead of guessing.

## Knowledge packs (for LLMs)

The same file works as a retrieval tool for an LLM. `pikelet mcp` serves
one or more packs over the Model Context Protocol, so Claude Code or any
MCP client can attach one with a single line:

```bash
npx pikelet mcp install --client claude-code --pack ./search.pikelet
```

Every result carries provenance: pack identity, source, heading.
Identities can be pinned (`--pack <url>#<sha256>`) so a client refuses a
pack that isn't the exact bytes it expects. See `packs/README.md` for
hosting a pack and `pikelet/README.md` for the full MCP reference.

## What's in this repo

- **`src/`** the vector search engine (HNSW, WASM), with a quantized and
  a float backend. No native dependencies. Runs in Node, browsers, and
  Cloudflare Workers.
- **`complete/`, `pikelet-artifact*.js`** the artifact formats: the
  `.pikelet` container and the sketch index it embeds. Both are
  content-addressed and integrity-checked. See `spec/` for the byte-level
  contracts.
- **`pikelet/`** the CLI (`compile`, `create`, `mcp`, `doctor`) and the
  Docusaurus search plugin.
- **`examples/`** worked demos, from a minimal browser search widget to
  a wiki-scale deployment. Start with `examples/05-one-file-search/`.

## The trick underneath it

Corpus vectors are stored as one scale and offset per row plus a byte
array (`value ≈ offset + scale × byte`) instead of raw floats. That's
ordinary scalar quantization. Less obvious: the same identity holds if
you average groups of those bytes together, because averaging commutes
with an affine transform. `mean(offset + scale·b)` equals `offset +
scale·mean(b)`. So a compressed "sketch" of a row, used to scan millions
of candidates before touching the real vectors, costs no retraining and
no codebook. Just fewer bytes averaged with the row's own scale and
offset.

The same fact shows up again inside the query encoder. Its transformer
weights are quantized the same way, so a matrix-vector multiply against
the compressed weights can be computed directly from the integer bytes
and corrected by one scale and offset afterward, without reconstructing
the floats first. Full-precision throughput at a quarter of the memory
traffic. One cheap arithmetic fact, reused for approximate search, exact
search, and running a small language model, is most of why this fits in
a file instead of a service.

## Why it's built this way

Most vector search tools assume a long-running server. Pikelet assumes
a static file: build the index once, host it anywhere that serves byte
ranges, and let the reader fetch only what a query needs. That trade
buys reproducibility (an artifact's identity is a hash of its contents)
and portability (the same file opens in a browser, a Worker, or Node).
The cost is easy mutation. This isn't a fit for a corpus that changes
every few minutes.

## Install

```bash
npm install pikelet-wasm
```

or from a checkout:

```bash
git clone https://github.com/mcn92/pikelet.git && cd pikelet && npm install
```

## Tests

```bash
npm test              # engine + artifact + complete-profile conformance
npm run test:fuzz      # adversarial/malformed-input suite
npm run test:browser   # browser smoke test
```

Building the WASM engine from source (only needed if you edit `src/`)
requires the Emscripten SDK: `npm run build`.

## Status and license

The engine and artifact readers/builders (`pikelet-wasm`) are published
and stable. The complete `.pikelet` profile's spec
(`spec/COMPLETE_PROFILE.md`) is still Draft 2, under review. The format
is usable today, but expect revisions before 1.0.

Apache-2.0. See [LICENSE](LICENSE).
