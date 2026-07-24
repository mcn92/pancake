# Pancake docs search Pages UI

Static Cloudflare Pages shell for the Worker-backed semantic-search demo.

Hosted UI:

```text
https://pancake-docs-search.pages.dev
```

Worker API:

```text
https://pancake-docs-search.mcn9284.workers.dev
```

The UI does not bake in the Worker URL or demo access key. Enter them in the
Connection section and click Save; they are stored in browser `localStorage`.

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
