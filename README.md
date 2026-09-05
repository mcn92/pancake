# Pikelet

Vector search that ships as a file.

A `.pikelet` bundles a corpus, its retrieval machinery, its query encoder, integrity metadata, and evaluation fixtures into one immutable artifact. Search is the first thing you do with it; distributing knowledge is what it's for.

You compile a folder of documents into one `.pikelet` file. That file contains the index, the text, the model that turns a question into a vector, a keyword index, and a set of hashes that let a reader prove none of it has been altered. Put it on S3, a CDN, or a static host. A reader — in a browser, a Cloudflare Worker, or Node — opens it over HTTP range requests, fetches a few hundred kilobytes, and answers the query. No server. No vector database. No embedding API.

```
npx pikelet compile --source ./docs --out search.pikelet
```

## What you build with it

- Search on a documentation site with no backend — compile, upload one file, done.
- Search embedded in a browser app, a Cloudflare Worker, an Electron app, or an offline tool, with no native dependencies.
- A versioned knowledge base distributed as a single asset: publish a new file, pin the hash, old readers keep working.
- A searchable corpus you hand to a customer instead of giving them access to your infrastructure.
- A frozen, hash-pinned body of knowledge that an LLM agent can mount by URL and cite by section.
- Public or private knowledge packs — a product manual, a legal code, a research corpus — that anyone with the URL can query without you running anything.

The common shape: the corpus is built offline, changes on a release cycle rather than by the second, and you'd rather host it than operate it.

Renamed from *Pancake* in September 2026. Same project, same file formats; `pancake-wasm` and `create-pancake-search` are deprecated pointers to `pikelet-wasm` and `pikelet`. Not related to the [Pikelet programming language](https://github.com/pikelet-lang/pikelet), which had the name first.

## One decision

Everything in this repo follows from how a vector is stored. Each row is 8-bit integers plus two floats:

```
x[d] ≈ offset + scale · q[d]        q[d] ∈ 0..255
```

That's a per-row affine map. It costs 4× less memory than float32, which is the ordinary reason to do it. The reason it runs through the whole project is that every operation search needs — dot products, sums, averages — is linear, and affine terms factor out of linear operations. So:

**Query against a stored row.** The query stays float32 and is never quantized.

```
y·x = offset·Σy + scale·(y·q)
```

The inner loop multiplies floats by bytes. The two constants come in once at the end. No decompressed copy of the row ever exists.

**Stored row against stored row.** This is what graph construction needs, thousands of times per insert.

```
x_i·x_j = D·o_i·o_j + o_i·s_j·Σq_j + o_j·s_i·Σq_i + s_i·s_j·(q_i·q_j)
```

`D` is the vector's dimensionality (384 for the bundled encoder, 128 for SIFT1M below). `q_i·q_j` is an integer dot of two byte arrays — exact, and the cheapest SIMD instruction there is. `Σq` is stored per row. The HNSW graph is built and repaired entirely on compressed data.

**Pooling.** Average adjacent groups of `p` bytes and you get a shorter row. Because the mean of `offset + scale·q` over a group equals `offset + scale·mean(q)`, the shorter row keeps the *same two constants*. That's the sketch tier (below), and the micro tier is the same thing done twice.

**Weights.** The bundled query encoder is a 6-layer MiniLM whose matrices are stored the same way, one (scale, offset) per 64-column block. Activations stay float32, weights stay bytes, dequantization happens inside the matmul. 24 MB instead of 86, and the same widen-and-multiply kernel that scores vectors runs the transformer.

The consequence for the file format: a row decodes from its own bytes and its own two floats and nothing else. No codebook, no global statistics. So every row is a fixed-size byte range at a computable offset, which is what makes it fetchable on its own, hashable on its own, and verifiable on the read that fetches it.

Product quantization would compress harder. It would also need a codebook, a training pass, table lookups in the distance kernel, and rows that mean nothing without the codebook. Four-to-one with no shared state was the better trade for a file meant to be read in pieces.

## How a query runs

Over a network, what makes search slow is not bytes; it's sequential round trips. Graph traversal is a chain of dependent reads. Measured against object storage over the public internet on SIFT1M (2026-08-01), a flat scan plus one parallel fetch beat graph traversal by about 5× at equal recall. So for the remote artifact path there is no graph in the file. There's a scan. (The HNSW graph still exists — it's the in-memory engine, used when the whole index is local and round trips are free.)

1. **Resident tier.** A pooled, 4-bit version of every row (`sketch`), loaded once when the file opens and kept in memory. For 456k Wikipedia passages at 384 dimensions that's ~52 MiB. A SIMD kernel scans all of it in ~17 ms per million rows.
2. **One fetch round.** The scan picks the top `C` candidates. Their full 8-bit rows are fetched by byte range, in parallel, with nearby ranges coalesced (gap 2 KB — larger gaps were measured to fetch 17× the bytes for zero fewer round trips). Keyword hits from the BM25 index join the same candidate set.
3. **Exact rerank.** Each fetched row is scored with the float query, verified against its digest, and the top `k` is returned with distance, title, section, source, and the file's content hash.

`C` is the only knob. The compiler measures recall against brute force on held-out queries at build time and writes the operating point into the file, so readers don't guess.

For very large files, a **micro tier** — the sketch pooled again — lets the reader open on a smaller resident slice, serve queries from it immediately with a wider candidate pool, and swap to the full sketch when it has streamed in. Results say which tier answered.

## What's in the file

```
manifest      identity hash, segment offsets, format versions
index         sketch rows (resident) + 8-bit rows (lazy), per-row digests
corpus        text records behind an offsets table, per-record digests
encoder       WordPiece vocab + quantized MiniLM weights   (~24 MB, fixed)
lexical       BM25 inverted index
calibration   abstention model, or a stated reason it was skipped
evaluation    golden queries and expected results
```

The manifest hash is the file's identity. Pin it (`file.pikelet#<sha256>`) and a reader refuses to serve anything else.

**Integrity.** The resident prefix is verified at open. Every lazily fetched index row and every hydrated corpus record is verified against its own digest before it can affect a result; the digests ride inside reads the reader was already making (+4.2% file size, ~0% latency, measured 2026-08-25). Bytes no query reads are not verified — that's the trade for not downloading the file. `verifyVectors()` does the full pass if you want it. A tampered file fails the query. It does not skew it.

**Abstention.** At build time the compiler fits a small model on retrieval signals (best distance, margin, term coverage) and cross-validates it against in-domain unanswerable queries. When that fit separates supported from unsupported retrievals reliably, results carry `matchQuality: strong | weak | none` — an estimate of whether the corpus supports the query, not a guarantee. When it doesn't — small corpora, or ones where every chunk looks alike, such as a single novel — calibration is skipped, the reason is logged, and results report `unscored`. The raw distances are always there either way.

**Encoder cost.** The default (`kind 3`) bundles the full encoder: ~24 MB regardless of corpus size. That is what makes the file self-contained. A corpus-distilled encoder (`kind 1`, ~1 MB, lower quality) or a host-supplied one (`kind 2`, verified against embedded test vectors at open) are the smaller options — see [`pikelet/README.md`](pikelet/README.md) for the tradeoffs.

## What a `.pikelet` is

A `.pikelet` is a self-contained, immutable knowledge artifact. It carries the source text, semantic and keyword retrieval over it, the encoder that turns a question into a query, per-record and per-row integrity commitments, whatever calibration the corpus could support, and the evaluation fixtures that show what it was tested against. Everything needed to interrogate the corpus, and nothing that needs a service.

You publish one like a static asset. You pin it by hash. You mount it from a local path or a URL. The same file serves a browser, an edge worker, a Node process, or an agent, and every answer it gives carries the file's identity and the section it came from.

Search is how you interrogate it. The file is the thing.

## What's in this repo

- **`src/`** the vector search engine (HNSW, WASM), quantized and float backends. No native dependencies; runs in Node, browsers, and Cloudflare Workers.
- **`complete/`, `pikelet-artifact*.js`** the artifact formats: the `.pikelet` container and the sketch index it embeds. Both are content-addressed and integrity-checked. See `spec/` for the byte-level contracts.
- **`pikelet/`** the CLI (`compile`, `create`, `mcp`, `doctor`) and the Docusaurus search plugin.
- **`examples/`** worked demos, from a minimal browser search widget to a wiki-scale deployment. Start with [`examples/06-mcp-knowledge-pack/`](examples/06-mcp-knowledge-pack/) (compile, mount, query over MCP) or [`examples/05-one-file-search/`](examples/05-one-file-search/) (the file itself).

## Attaching a pack to an LLM

For an agent, a pack is a bounded knowledge source: a frozen, hash-pinned body of text with searchable evidence and a stable identity. It can't drift between calls, it can't be quietly edited, and when support for a query looks weak, the retrieval metadata exposes that uncertainty rather than hiding it. An agent can mount one or several — a product's docs, a codebase's design notes, a reference corpus — and cite by pack identity and section.

`pikelet mcp` serves packs over the Model Context Protocol on stdio:

```
npx pikelet mcp install --client claude-code \
  --pack https://example.com/docs.pikelet#<sha256>
```

The client gets `search`, `get_record`, `list_packs`, and `verify_pack` (which runs the golden queries stored in the file). Every result carries the pack identity and source location, so citations are pinnable. URL packs are range-read, never downloaded whole. See [`pikelet/README.md`](pikelet/README.md) for the full MCP reference and [`packs/README.md`](packs/README.md) for hosting a pack.

## Install

```
npm install pikelet-wasm        # engine + artifact readers/builders
npm install -g pikelet          # compile / create / mcp CLI
```

```
git clone https://github.com/mcn92/pikelet.git && cd pikelet && npm install
```

The checkout includes prebuilt WASM in `dist/`; rebuilding needs Emscripten and is only necessary if you change `src/`.

```js
import { openPikeletFile } from 'pikelet-wasm/complete';

const search = await openPikeletFile('search.pikelet');   // path, URL, or a { size, read(offset, len) } source
const out = await search.query('how do workers restore snapshots', { k: 5 });
console.log(out.matchQuality, out.results[0]?.title);
await search.close();
```

If you already have vectors and just want the in-memory engine:

```js
import Pikelet from 'pikelet-wasm';
const index = await Pikelet.create({ dim: 384, maxElements: 100000, metric: 'cosine', quantized: true });
index.add(vec); index.search(query, 10); index.export();
```

## Numbers

Stated with their conditions, because they don't mean anything without them.

- **Simple English Wikipedia**, 456,153 passages, 384-dim, kind-3 encoder inline: 649 MiB file. Opens on ~52 MiB. ~127 range reads per query. recall@10 95.2% at the recommended `C` (97.0% at `C=600`) against float32 brute force on a 200-query eval fixed before the measurement. ~110 ms/query end to end locally (query encoding + SIMD scan + rerank; the scan itself is ~8 ms of that at this corpus size).
- **A 700-record novel**, same settings: 26 MB file, 21 range reads and 140 KB per query. Abstention skipped (AUC 0.695 < 0.75).
- **SIFT1M**, 128-dim, 64-dim 4-bit sketch: 38 MiB resident, 96% recall@10 at `C=300`.

Benchmarks against faiss, hnswlib, and usearch are in `benchmarks/`. They're honest about the fact that a WASM engine loses to native on raw in-memory throughput; that isn't what this is for.

## When to use it, when not to

Use it if you want search on a docs site or a fixed corpus with no infrastructure; if your deployment is static files or the edge; if you need to hand an LLM a frozen, hash-pinned body of text; if you'd rather host an index than operate one.

Don't use it if your corpus changes faster than you can recompile; if you need a multi-tenant server; if you need exact nearest neighbours at scale; if you're already running pgvector or a native ANN service and portability isn't a constraint. The resident scan is linear in corpus size and is comfortable to a few million rows, not beyond.

## Status

One author. The engine and readers are published and tested (`npm test` runs the engine, sketch, complete-profile, MCP, and ingestion conformance suites). The `.pikelet` format spec is Draft 2 and not frozen. The encoder kernel's C++ source lives at [`examples/05-one-file-search/encoder-spike/encoder.cpp`](examples/05-one-file-search/encoder-spike/encoder.cpp); the compiled `.wasm` shipped in `complete/encoder-kernels/` is built from it, and its weights are verified against embedded test vectors at open. Bare `.pnck` engine snapshots carry no checksum; integrity starts at the artifact layer.

Specs: [`spec/SEARCH_ARTIFACT_CONTRACT.md`](spec/SEARCH_ARTIFACT_CONTRACT.md), [`spec/SKETCH_PROFILE.md`](spec/SKETCH_PROFILE.md), [`spec/COMPLETE_PROFILE.md`](spec/COMPLETE_PROFILE.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
