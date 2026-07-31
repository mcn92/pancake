# Browser Search Artifact Demo

This is a static browser demo for Pancake range-readable artifacts. It opens a
`.pancake-range` file through HTTP range requests, keeps the v2 router resident,
and lazily materializes base-layer nodes as the search runs.

From the repository root:

```bash
npx vite build examples/search-artifact-demo/static
npx vite preview examples/search-artifact-demo/static --host 127.0.0.1
```

Then open the printed local URL.

The demo includes a tiny SIFT-shaped v2 artifact:

```text
examples/search-artifact-demo/static/public/artifacts/pancake-smoke-split.pancake-range
```

That artifact is intentionally small so the page can run immediately. Replace
it with a larger artifact using the same filename, or change `ARTIFACT_URL` in
`src/main.js`.

The first command writes static files to:

```text
examples/search-artifact-demo/static/dist
```

The second command serves that built directory locally.

For deployment, upload the contents of `dist` to any static host that preserves
range requests for `.pancake-range` files.

To use a larger artifact:

```bash
cp path/to/index.pancake-range \
  examples/search-artifact-demo/static/public/artifacts/pancake-smoke-split.pancake-range
npx vite build examples/search-artifact-demo/static
```
