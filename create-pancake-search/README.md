# create-pancake-search

Scaffold a Pancake-backed Cloudflare Worker search app.

```bash
npm create pancake-search -- --name my-docs-search --source ./docs --no-deploy --yes
cd my-docs-search
npm run dev
```

The generated project contains a bundled Pancake snapshot, corpus metadata, a Workers AI search worker, and a static UI.

Experimental Search Artifact runtime scaffolding is available with:

```bash
npm create pancake-search -- --name my-docs-search --source ./docs --runtime artifact --no-deploy --yes
```

You can also provide a prebuilt `.pancake-range` file with `--artifact`. External
artifacts must have dimension, count, and IDs that match the generated corpus.

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

That means docs, blog posts, pages, and rendered MDX all flow through the same
folder ingestion, chunking, teacher-vector indexing, and Search Artifact builder
as the CLI, without generating or deploying a Worker. The teacher model runs at
build time; the built site only serves the compact student model for browser
query vectors.

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

Advanced users can provide a pre-trained model with `studentModel`, but it must
be trained for this site or a very close corpus. A Wikipedia-trained student is
only useful for smoke testing the mechanics; it is not a general-purpose docs
encoder.
