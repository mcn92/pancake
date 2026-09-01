# The pack shelf

`packs.json` is a shelf: a static listing of `.pancake` knowledge packs.
There is no registry service and no accounts — the registry is also just a
file. Mount everything on a shelf in one line:

```bash
npx create-pancake-search mcp --shelf https://raw.githubusercontent.com/mcn92/pancake/main/packs/packs.json
```

or write the config for your MCP client instead of running it by hand:

```bash
npx create-pancake-search mcp install --shelf <shelf-url> --client claude-code
```

Each entry names a pack, where it lives (`url`, or `path` relative to the
shelf file), and optionally its immutable manifest `identity` — the sha256
every mount verifies before serving, so a shelf pins exact knowledge
states, not just locations. Packs are range-read off dumb HTTP: mounting
the 649 MiB Wikipedia pack transfers ~52 MiB and each question costs ~127
range requests; the file is never downloaded whole.

## Adding a pack

Compile one and host it anywhere that serves HTTP ranges — R2, S3, any
CDN; `create-pancake-search doctor <url>` certifies a host. GitHub
release assets work but rate-limit sustained range bursts (HTTP 429)
under heavy per-IP use; the reader retries with backoff, but a
high-traffic shelf entry belongs on object storage or a CDN:

```bash
npx create-pancake-search compile --source https://docs.example.com \
  --out example-docs.pancake --license MIT
```

The compile output prints the manifest identity; put it in the shelf entry.
Set `--license` on anything meant for redistribution — it is recorded in
the pack manifest and surfaced by `list_packs`, and result provenance
(title, heading path, source URL) carries attribution through to answers.
