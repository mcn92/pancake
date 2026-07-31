# Pancake docs search Pages UI

Static Cloudflare Pages shell for the Worker-backed semantic-search demo.

Hosted UI:

```text
https://pancake-docs-search.pages.dev
```

Worker API: none public at the moment. The companion Worker is deployed in
private mode (`workers_dev = false` in
`../worker-semantic-search/wrangler.toml`), so its `workers.dev` endpoint is
disabled. Point the UI at your own deployment of the Worker.

The UI does not bake in the Worker URL or demo access key. Enter them in the
Connection section and click Save; the Worker URL is stored in browser
`localStorage`, and the access key is stored only in tab-scoped `sessionStorage`.

## Deploy

From the repository root:

```bash
npm run build:pages-demo
npx wrangler pages deploy examples/worker-semantic-search-pages/dist \
  --project-name pancake-docs-search
```

The Worker backend lives in `../worker-semantic-search/`. Search is currently
private when the Worker is deployed with `PRIVATE_SEARCH=1` and a
`DEMO_SEARCH_KEY` secret.
