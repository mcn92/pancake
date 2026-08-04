#!/usr/bin/env node
// Local server for the wiki-pack demo skeleton with real 206 Range support:
//   /pack/*    -> ../data-full/*          (artifact, corpus, offsets, manifest)
//   /models/*  -> transformers.js cache   (self-hosted MiniLM assets)
//   /ort/*     -> onnxruntime-web dist    (ONNX runtime wasm)
//   /*         -> ./dist/*                (vite build output)
import fs from 'node:fs';
import http from 'node:http';
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

// WSL note: stat/open on /mnt/c cost tens of ms, so cache fd + size per file
// and serve ranges with positional reads instead of streams.
const fileCache = new Map();
function openCached(filePath) {
    let entry = fileCache.get(filePath);
    if (!entry) {
        if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) return null;
        entry = { fd: fs.openSync(filePath, 'r'), size: fs.statSync(filePath).size };
        fileCache.set(filePath, entry);
    }
    return entry;
}

const server = http.createServer((req, res) => {
    const url = decodeURIComponent(new URL(req.url, 'http://x').pathname);
    let file = null;
    for (const [prefix, root] of roots) {
        if (url.startsWith(prefix)) {
            let rel = url === '/' ? 'index.html' : url.slice(prefix.length);
            // Pack URLs carry a cache-busting version segment (/pack/vXXXX/…)
            // that the Pages Function strips before lookup; mirror that here.
            if (prefix === '/pack/') rel = rel.replace(/^v[0-9a-f]{6,}\//, '');
            const candidate = path.normalize(path.join(root, rel));
            if (!candidate.startsWith(root + path.sep)) break;
            file = openCached(candidate);
            if (file) { file.path = candidate; break; }
        }
    }
    if (!file) { res.writeHead(404); res.end('not found'); return; }

    const type = TYPES[path.extname(file.path)] || 'application/octet-stream';
    const range = req.headers.range && /^bytes=(\d+)-(\d*)$/.exec(req.headers.range);
    const start = range ? Number(range[1]) : 0;
    const end = range ? (range[2] ? Math.min(Number(range[2]), file.size - 1) : file.size - 1) : file.size - 1;
    const length = end - start + 1;
    const headers = {
        'Content-Type': type,
        'Content-Length': length,
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'no-store',
    };
    if (range) headers['Content-Range'] = `bytes ${start}-${end}/${file.size}`;
    res.writeHead(range ? 206 : 200, headers);
    if (length <= 4 * 1024 * 1024) {
        const buf = Buffer.allocUnsafe(length);
        fs.readSync(file.fd, buf, 0, length, start);
        res.end(buf);
    } else {
        fs.createReadStream(file.path, { start, end }).pipe(res);
    }
});

const port = Number(process.env.PORT || 8931);
server.listen(port, '127.0.0.1', () => console.log(`wiki-pack demo on http://127.0.0.1:${port}`));
