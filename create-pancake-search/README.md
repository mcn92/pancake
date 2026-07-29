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
only for testing the Worker/search path.

In Search Artifact mode, `/search` responses include per-query and cumulative
range-read stats so cold-load and warm-cache behavior are visible directly.
