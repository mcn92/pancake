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

## Limitations

The bundled student encoder featurizes `[a-z0-9']` tokens only, and chunking
counts whitespace-separated tokens. English and other Latin-script,
whitespace-delimited content works; unsegmented scripts (CJK and similar) do
not — the build fails with an explicit 0-chunks error, and queries with no
recognized terms return a graceful no-match instead of results.
