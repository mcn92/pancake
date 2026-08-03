#!/usr/bin/env node
// HTTP/2 variant of the demo server (self-signed TLS) — the transport real
// hosts speak, so browser-side fetch parallelism isn't capped at h1's six
// connections. Same route map and range semantics as serve.mjs.
import fs from 'node:fs';
import http2 from 'node:http2';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const roots = [
    ['/pack/', path.join(here, '..', process.env.PACK_DATA || 'data-full')],
    ['/models/', path.join(here, '..', 'node_modules', '@huggingface', 'transformers', '.cache')],
    ['/ort/', path.join(here, '..', 'node_modules', 'onnxruntime-web', 'dist')],
    ['/', path.join(here, 'dist')],
];
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm', '.onnx': 'application/octet-stream' };

// Whole files are preloaded into RAM on first touch: the point of this
// server is measuring the BROWSER side, so server-side I/O latency (sync
// reads on a slow /mnt/c mount) must not pollute the numbers.
const fileCache = new Map();
function openCached(filePath) {
    let entry = fileCache.get(filePath);
    if (!entry) {
        if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) return null;
        const data = fs.readFileSync(filePath);
        entry = { data, size: data.byteLength, path: filePath };
        fileCache.set(filePath, entry);
    }
    return entry;
}

const server = http2.createSecureServer({
    key: fs.readFileSync(path.join(here, '.cert', 'key.pem')),
    cert: fs.readFileSync(path.join(here, '.cert', 'cert.pem')),
});

server.on('stream', (stream, headers) => {
    const url = decodeURIComponent(new URL(headers[':path'], 'http://x').pathname);
    let file = null;
    for (const [prefix, root] of roots) {
        if (url.startsWith(prefix)) {
            const rel = url === '/' ? 'index.html' : url.slice(prefix.length);
            const candidate = path.normalize(path.join(root, rel));
            if (!candidate.startsWith(root + path.sep)) break;
            file = openCached(candidate);
            if (file) break;
        }
    }
    if (!file) { stream.respond({ ':status': 404 }); stream.end('not found'); return; }

    const type = TYPES[path.extname(file.path)] || 'application/octet-stream';
    const range = headers.range && /^bytes=(\d+)-(\d*)$/.exec(headers.range);
    const start = range ? Number(range[1]) : 0;
    const end = range ? (range[2] ? Math.min(Number(range[2]), file.size - 1) : file.size - 1) : file.size - 1;
    const length = end - start + 1;
    const h = {
        ':status': range ? 206 : 200,
        'content-type': type,
        'content-length': length,
        'accept-ranges': 'bytes',
        'cache-control': 'no-store',
    };
    if (range) h['content-range'] = `bytes ${start}-${end}/${file.size}`;
    stream.respond(h);
    stream.end(file.data.subarray(start, end + 1));
});

const port = Number(process.env.PORT || 8932);
server.listen(port, '127.0.0.1', () => console.log(`wiki-pack demo (h2) on https://127.0.0.1:${port}`));
