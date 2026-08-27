# create-pancake-search

Scaffold the product version of the Pancake demo path: ingest docs, build the
search assets offline, ship them with a Worker/UI shell, and serve retrieval at
the edge. The lower-level `/examples` directory shows the same lifecycle piece
by piece; this package is the fast path when you want the story to become a
deployable app.

The 15-second version:

```bash
npm create pancake-search -- --name my-docs-search --source ./docs --no-deploy --yes
cd my-docs-search
npm run dev
```

The generated project contains a bundled Pancake snapshot, corpus metadata, a
Workers AI search worker, and a static UI. For a pure Search Artifact demo that
matches the artifact examples more directly, use:

```bash
npm create pancake-search -- --name my-docs-search --source ./docs --runtime artifact --no-deploy --yes
```

In both runtimes, the story is the same: the expensive work happens at build
time; query-time code embeds the query, searches Pancake, and hydrates result
metadata. Query embedding comes from Workers AI by default, or from a bundled
corpus-distilled encoder with `--mode student` (see below), which removes the
Cloudflare AI dependency entirely. For workers-ai projects, `LOCAL_STUB_AI=1`
can exercise the endpoint mechanics locally without Workers AI.

You can also provide a prebuilt `.pancake-range` file with `--artifact`. External
artifacts must have dimension, count, and IDs that match the generated corpus.

## Compiling a complete `.pancake` artifact

When you want the search *file* rather than a search *app*, `compile` builds
a complete kind-3 artifact and stops — no project, no Worker, no Cloudflare:

```bash
npx create-pancake-search compile --source ./docs --out search.pancake
```

The output is one self-contained file carrying the corpus records, sketch
index, inline MiniLM query encoder, and evaluation data. Open it from any
runtime with the complete reader:

```js
import { openPancakeFile } from 'pancake-wasm/complete';
const search = await openPancakeFile('search.pancake');
const out = await search.query('how do I configure auth', { k: 5 });
```

Passage embedding runs locally through the same inline encoder the artifact
carries (the ~24 MiB weight blob is fetched once, digest-pinned, when the
package copy is absent — registry installs ship without it). `compile`
accepts `--source` (folder or URL), `--out`, `--name` (corpus name recorded
in the artifact), `--max-pages`, `--include`/`--exclude`, and `--force` to
overwrite the output file. The scaffold-only flags (`--mode`, `--runtime`,
`--artifact`, deploy and student options) are rejected: compile always
builds the complete kind-3 profile.

## Where this sits

This package is the **product layer** of the Pancake stack. It consumes the
two layers below it — the `pancake-wasm` ANN engine and the Search Artifact
readers/builders (`spec/SEARCH_ARTIFACT_CONTRACT.md` in the main repo) — and
emits a project that is *yours*: the generated Worker, UI, and config are
application code with `pancake-wasm` as a dependency, not part of this
package. Engine and artifact behavior are documented in the main repo;
this README covers only scaffolding, generation options, and the generated
project's layout.

### URL ingestion trust boundary

`--source <url>` crawls a website from your machine at build time. The
crawler runs locally under your account, follows the URL *you* typed, keeps
the crawl frontier on that URL's origin, skips redirects, enforces timeouts
and per-page/body caps, and never runs at query time — the deployed Worker
makes no outbound fetches at all. It deliberately does not block
private-network addresses: it is a local developer tool, and pointing it at
your own intranet docs is a supported use. Do not lift the crawl code into a
deployed service without adding SSRF protections (scheme allowlist,
private/link-local IP rejection, redirect pinning).

## Self-contained query embedding (`--mode student`)

`--mode student` removes the Workers AI dependency entirely. At build time the
CLI distills a corpus-specific teacher-student (PSTU) query encoder — the same
one the Docusaurus plugin and the edge docs-search demo use — and bundles it
into the Worker (~1.1 MiB). Queries embed in-process in single-digit
milliseconds, the generated `wrangler.toml` has no `[ai]` binding, and
`wrangler dev` runs fully local with no Cloudflare account:

```bash
npm create pancake-search -- --name my-docs-search --source ./docs --runtime artifact --mode student --no-deploy --yes
```

Training requires a Python 3 environment with `torch` and `transformers`
(`PANCAKE_SEARCH_PYTHON` selects the interpreter). The trainer also calibrates
the abstention scorer and enforces acceptance gates; on small or noisy corpora
those gates can fail, in which case pass `--skip-abstention` to ship the
encoder without a match-quality scorer (responses report
`match_quality: "unscored"`), or improve the source corpus. When calibration
succeeds, `/search` reports `match_quality` and returns no results for
out-of-domain queries.

To reuse a previously trained encoder instead of retraining, pass
`--student-model <model.bin> --student-vectors <docs-vectors.f32>`
(optionally `--student-abstention <scorer.json>`). The teacher document
vectors must come from the same training run so the index geometry matches
the query encoder.

For local endpoint testing without Cloudflare Workers AI, generated Workers
support `LOCAL_STUB_AI=1`. It uses deterministic hash embeddings and is meant
only for testing the Worker/search path. If you build with
`PANCAKE_SEARCH_STUB_EMBEDDINGS=1`, rebuild with real Workers AI embeddings
before deploy; stub-built indexes contain hash embeddings, not semantic
embeddings.

In Search Artifact mode, `/search` responses include per-query and cumulative
range-read stats so cold-load and warm-cache behavior are visible directly.

URL ingestion is intentionally conservative: crawls stay on the seed origin,
skip redirects, and cap HTML response bodies before parsing.

## Package layout

`bin/create-pancake-search.mjs` calls `main()` in `src/cli.mjs`, which owns
argument parsing and the `create` / `rebuild` / `doctor` commands and the
config a scaffold is generated from. The work lives beside it:

| module | responsibility |
| --- | --- |
| `src/common.mjs` | package paths and version, config defaults, the model table, `CliError`, loaders that resolve `pancake-wasm` (engine, `/artifact`, `/complete`) from npm or the monorepo |
| `src/ingest.mjs` | folder walk and URL crawl, HTML/Markdown/MDX extraction, chunking, dedupe, Docusaurus route mapping, the public chunk shape |
| `src/embed.mjs` | build-time embeddings: transformers.js, the student trainer, the inline transformer, precomputed vectors, the deterministic stub, self-recall |
| `src/complete-build.mjs` | kind-3 complete artifact assembly, the inline-encoder declaration, the pinned weights download |
| `src/scaffold.mjs` | generated-project files: runtime modules, templates, `wrangler.toml` / `package.json`, student input staging, deploy |
| `src/build.mjs` | `buildAssets` (ingest → chunk → embed → index → artifact), config validation, `manifest.json`, student asset publishing, bundle sizing |
| `src/doctor.mjs` | the `doctor <url>` hosting probe |
| `docusaurus/` | the Docusaurus plugin and its browser client, built on `buildSearchAssets` |

## Checking a host: `doctor`

Range-read artifacts depend on transport properties that hosts get wrong
silently, and the symptom is "the demo is slow", not an error. Before (or
after) deploying a `.pancake`, `.pancake-sketch`, or `.pancake-range` file,
probe the URL it is served from:

```bash
npx create-pancake-search doctor https://example.com/search/search.pancake
```

It prints a pass/warn/fail line per check — `HEAD` (size, `Accept-Ranges`,
`ETag`), a real 64-byte `Range` GET (206 vs full-body 200), the same range
with a `?r=start-end` cache-key query (the form every browser read uses, to
defeat Chromium's same-URL cache-entry lock), the negotiated protocol
(HTTP/1.1 serializes parallel rerank reads at ~6 connections; h2/h3
multiplex), median RTT over three small reads, and the artifact's magic and
identity from its first 64 bytes — and exits 1 if any check fails.

## Docusaurus

Docusaurus sites can build a static Pancake Search Artifact through the package
subpath plugin:

```js
// docusaurus.config.js
import pancakeSearch from 'create-pancake-search/docusaurus';

export default {
  plugins: [
    [
      pancakeSearch,
      {
        assetBase: 'pancake-search',
        name: 'my-docs-search',
      },
    ],
  ],
};
```

On `docusaurus build`, the plugin indexes the rendered HTML in the build output
directory, trains a corpus-specific teacher-student distilled encoder for those
rendered chunks, then writes static artifact assets into `build/pancake-search/`:

- `index.pancake-range` — range-readable Pancake Search Artifact
- `corpus.json` — result metadata and snippets
- `manifest.json` — embedding/index/runtime metadata and URLs
- `student-model.bin` — the PSTU student encoder used by browser queries
- `student-abstention.json` — the generated match-quality scorer

That means docs, blog posts, pages, and rendered MDX all flow through the same
folder ingestion, chunking, teacher-vector indexing, and Search Artifact builder
as the CLI, without generating or deploying a Worker. The teacher model runs at
build time; the built site only serves the compact student model for browser
query vectors and abstention scorer.

By default, the plugin injects a floating, draggable search panel into the page
and exposes `window.PancakeDocusaurusSearch` for custom UI code. The panel's JS
and CSS are bundled through Docusaurus; the generated static directory only
contains the search artifact assets. To ship only the assets and mount your own
UI, disable the default mount:

```js
[pancakeSearch, { assetBase: 'pancake-search', mount: false }]
```

The default build expects a Python environment with `torch` and `transformers`.
Set `trainStudent.python` if Docusaurus should call a specific interpreter:

```js
[pancakeSearch, { trainStudent: { python: '.venv/bin/python', epochs: 60 } }]
```

Advanced users can provide pre-trained assets, but the model has to travel with
the matching teacher document vectors for the rendered corpus:

```js
[
  pancakeSearch,
  {
    studentModel: './pancake-student.bin',
    studentVectors: './docs-vectors.f32',
    studentAbstention: './student-abstention.json',
  },
]
```

`studentModel` is query-side only. The plugin refuses to build passages from a
bare student model because that silently changes the index geometry. A
Wikipedia-trained student is only useful for smoke testing the mechanics; it is
not a general-purpose docs encoder.

### Complete profile (one `search.pancake` file, query-interp kind 3)

`completeProfile.enabled` switches the plugin from the range artifact plus
student encoder to the complete profile: a single `search.pancake` that
carries the corpus, index, WordPiece vocab, quantized MiniLM encoder weights,
calibration, and evaluation data, read in the browser by the kind-3 reader
from `pancake-wasm/complete`. No student training, no Python, no hosted
encoder — the reader supplies kernels, the file supplies data.

```js
[
  pancakeSearch,
  {
    assetBase: 'pancake-search',
    sourcePath: 'docs',              // index markdown/MDX sources instead of built HTML
    sourceRouteBase: 'docs',
    completeProfile: {
      enabled: true,
      vocab: 'node_modules/create-pancake-search/src/inline-encoder/vocab.txt',
      weights: './pancake-search/encoder-weights.bin',   // fetched here on first build
      model: 'sentence-transformers/all-MiniLM-L6-v2',
      maxTokens: 128,
      // vectors: './docs-vectors.f32',       // optional precomputed document vectors
      // calibration: './calibration.json',   // optional abstention calibration
    },
  },
]
```

Paths resolve against the site directory. `vocab.txt` ships in this package
(`src/inline-encoder/vocab.txt`). The 24.3 MiB `encoder-weights.bin` does
not: when the configured path is missing
and its basename is `encoder-weights.bin`, the plugin (and the CLI's
`runtime.mode: "complete"` path) downloads it once from the
`inline-encoder-v1` GitHub release, verifies the pinned SHA-256, and writes
it to that path for reuse. Set `PANCAKE_ENCODER_WEIGHTS_URL` to fetch from a
mirror; custom-named weights are never fetched. Without `vectors`, the build
embeds every chunk through the packaged encoder at build time (inputs longer
than `maxTokens` are windowed and mean-pooled, and the build logs how many).

The output is `build/pancake-search/search.pancake` (plus `corpus.json` and
`manifest.json`);
the plugin's search panel opens it over HTTP range reads, so the host must
honor `Range` — check with `create-pancake-search doctor <url>`.

## Limitations

The bundled student encoder featurizes `[a-z0-9']` tokens only, and chunking
counts whitespace-separated tokens. English and other Latin-script,
whitespace-delimited content works; unsegmented scripts (CJK and similar) do
not — the build fails with an explicit 0-chunks error, and queries with no
recognized terms return a graceful no-match instead of results.
