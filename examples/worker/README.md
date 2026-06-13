# Pancake Worker Reference Architecture

This directory contains a reference Cloudflare Worker deployment built on top of `pancake-wasm`.

The Worker keeps a Pancake index in-process when hot and persists snapshots to
Cloudflare R2. Treat it as a **snapshot-first ANN serving layer** rather than
as a durable mutable vector database.

Best fit:

- read-heavy search workloads
- modest-sized indexes that can be restored into Worker memory
- explicit import/export flows
- periodic snapshot rebuilds published to R2

Less suitable:

- high-write authoritative online mutation
- strict cross-isolate read-after-write guarantees
- relying on in-memory Worker state as the only source of truth

## Recommended deployment shape

The safest default is:

1. build or update the index outside the Worker
2. publish a clean snapshot to R2
3. deploy the Worker in `READ_ONLY=1`
4. expose `/search` and `/health` publicly
5. keep mutation routes for admin-only workflows or local development

That keeps the Worker in the role it fits best: snapshot-backed search at the edge.

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /search` | k-NN search (`{ query: float[], k?, ef? }`) |
| `GET /health` | Health check (public, no auth required) |
| `GET /readiness` | Authenticated readiness check with snapshot visibility |
| `GET /stats` | Index count, memory, ghost ratio |
| `GET /export` | Serialize index to binary blob |
| `POST /init` | Create an index (`{ dims, maxElements, M?, efConstruction?, efSearch?, vectors? }`) |
| `POST /import` | Restore from binary (`?dims=N` required) |
| `POST /add` | Insert a vector (`{ vector: float[] }`) |
| `POST /add_batch` | Insert multiple vectors (`{ vectors: float[][] }`) |
| `POST /delete` | Soft-delete by ID (`{ id: number }`) |
| `POST /compact` | Rebuild graph without deleted entries |
| `POST /reset_cache` | Drop the warm in-memory index so the next query restores from R2 |

`/health` stays cheap and public. It reports whether the current isolate has an
index loaded, but it does not trigger a restore. `/readiness` is the
authenticated route for checking whether the Worker is loaded now and whether a
snapshot is available to restore.

## Running locally

```bash
cd examples/worker
npx wrangler dev --port 8787
```

## Deploying

```bash
cd examples/worker
wrangler r2 bucket create pancake-indexes
wrangler deploy
```

See [`wrangler.toml`](wrangler.toml) for the full configuration. Key env vars:

- `API_KEY` for bearer auth
- `ALLOWED_ORIGIN` for CORS
- `RATE_LIMIT_RPM` for per-IP rate limiting
- `READ_ONLY=1` to reject mutation/admin routes

For a safer internet-facing deployment, set at least:

- `API_KEY`
- `ALLOWED_ORIGIN`
- `READ_ONLY=1`

## Deployment notes

**Memory.** Workers have a 128 MB memory ceiling per isolate. Rough formula for quantized index memory:

```txt
~(dim + 8 + 7 * M) * num_vectors bytes
```

At `M=16` this is `(dim + 120)` bytes per vector. Examples: `30k x 256D = 10 MB`, `200k x 384D = 100 MB`. The fp32 backend uses `(4*dim + 8 + 7*M)` bytes per vector, roughly 4x more.

**CPU time.** Workers paid plan allows 50ms CPU per request (free tier: 10ms). Search comfortably fits within both tiers. Heavy operations such as `/import`, `/compact`, and large `/add_batch` requests can exceed free-tier limits on larger indexes.

**Cold starts and R2 restore.** Worker isolates are not persistent. On cold start, the Worker fetches the index from R2 and deserializes it lazily on the first request. For a large index, the first request after idle will be slow.

**Persistence.** The current Worker example writes snapshots to R2 and restores
from R2 on cold start. This is the durable boundary. In-memory state is only a
warm cache for the current isolate. Snapshot writes are append-only and restore
loads the latest saved snapshot, which avoids older async writes overwriting a
newer one under the same R2 key.

Mutation routes exist for demos, local validation, and administrative flows,
but they should not be treated as the primary production write path for a
stateful vector database. If you need stronger mutation semantics, put an
authoritative layer elsewhere and use the Worker as the search-serving
frontend.

**Read-only mode.** When `READ_ONLY=1`, the Worker rejects `/init`, `/import`,
`/add`, `/add_batch`, `/delete`, `/compact`, `/export`, `/reset_cache`, and
`/search_debug` with `403`. This is the recommended production setting for a
public search endpoint backed by snapshots.

**Rate limiting.** Rate limiting is in-memory per isolate. Each isolate tracks its own sliding-window counter, so the effective limit across multiple isolates is approximately `limit * number_of_isolates`.
