# Pancake Worker Reference Architecture

This directory contains a reference Cloudflare Worker deployment built on top of `pancake-wasm`.

The Worker keeps a Pancake index in-process when hot and persists snapshots to Cloudflare R2. It is best suited to read-mostly, modest-sized indexes where cold restore is acceptable and where the Worker acts as an ephemeral ANN serving layer rather than a durable vector database.

## Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /init` | Create an index (`{ dims, maxElements, M?, efConstruction?, efSearch?, vectors? }`) |
| `POST /add` | Insert a vector (`{ vector: float[] }`) |
| `POST /add_batch` | Insert multiple vectors (`{ vectors: float[][] }`) |
| `POST /delete` | Soft-delete by ID (`{ id: number }`) |
| `POST /compact` | Rebuild graph without deleted entries |
| `POST /search` | k-NN search (`{ query: float[], k?, ef? }`) |
| `GET /export` | Serialize index to binary blob |
| `POST /import` | Restore from binary (`?dims=N` required) |
| `GET /stats` | Index count, memory, ghost ratio |
| `GET /health` | Health check (public, no auth required) |

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

## Deployment notes

**Memory.** Workers have a 128 MB memory ceiling per isolate. Rough formula for quantized index memory:

```txt
~(dim + 8 + 7 * M) * num_vectors bytes
```

At `M=16` this is `(dim + 120)` bytes per vector. Examples: `30k x 256D = 10 MB`, `200k x 384D = 100 MB`. The fp32 backend uses `(4*dim + 8 + 7*M)` bytes per vector, roughly 4x more.

**CPU time.** Workers paid plan allows 50ms CPU per request (free tier: 10ms). Search comfortably fits within both tiers. Heavy operations such as `/import`, `/compact`, and large `/add_batch` requests can exceed free-tier limits on larger indexes.

**Cold starts and R2 restore.** Worker isolates are not persistent. On cold start, the Worker fetches the index from R2 and deserializes it lazily on the first request. For a large index, the first request after idle will be slow.

**Persistence.** The current Worker example debounces R2 writes with a 2-second timer and uses `ctx.waitUntil()` to complete writes after the response is sent. If the isolate is terminated before the timer fires, recent mutations may be lost. For workflows where every write must be durable, use `/export` explicitly after critical mutations or redesign the persistence path.

**Rate limiting.** Rate limiting is in-memory per isolate. Each isolate tracks its own sliding-window counter, so the effective limit across multiple isolates is approximately `limit * number_of_isolates`.
