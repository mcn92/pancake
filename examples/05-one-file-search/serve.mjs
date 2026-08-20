#!/usr/bin/env node
// Minimal static server with HTTP Range support — the only thing a .pancake
// host needs. Serves the built page from web/dist and the compiled
// .pancake from the example root.
//
//   node serve.mjs [port]     (default 8790)

import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';

const here = path.dirname(new URL(import.meta.url).pathname);
const distDir = path.join(here, 'web', 'dist');
const artifactPath = path.join(here, 'pancake-docs.pancake');
const TYPES = { '.html': 'text/html; charset=utf-8', '.mjs': 'text/javascript', '.js': 'text/javascript', '.css': 'text/css', '.pancake': 'application/octet-stream' };

// The artifact's identity is its manifest sha256, already sitting in the
// header — so the content-addressed URL costs one 64-byte read, no hashing.
function artifactIdentity() {
    const fd = fs.openSync(artifactPath, 'r');
    try {
        const header = Buffer.alloc(64);
        fs.readSync(fd, header, 0, 64, 0);
        return header.subarray(24, 56).toString('hex');
    } finally {
        fs.closeSync(fd);
    }
}

export function createServer() {
    return http.createServer((req, res) => {
        const url = new URL(req.url, 'http://localhost');
        // Content-addressed serving, the wiki pack's pattern (DEPLOY.md):
        // the artifact lives at an identity-hashed immutable URL and a tiny
        // short-lived pointer names the current one, so a rebuild mid-session
        // can never hand a client mixed bytes — stale pointers 404 and the
        // client refetches the pointer.
        if (url.pathname === '/pancake-latest.json') {
            const identity = artifactIdentity();
            const body = JSON.stringify({ identity, url: `/p/${identity}.pancake` });
            res.writeHead(200, {
                'Content-Type': 'application/json',
                'Cache-Control': 'max-age=60',
                'Content-Length': Buffer.byteLength(body),
            }).end(body);
            return;
        }
        const addressed = /^\/p\/([0-9a-f]{64})\.pancake$/.exec(url.pathname);
        let immutable = false;
        let filePath;
        if (addressed) {
            if (addressed[1] !== artifactIdentity()) {
                res.writeHead(404).end('stale artifact url — refetch /pancake-latest.json');
                return;
            }
            filePath = artifactPath;
            immutable = true;
        } else {
            filePath = url.pathname === '/' ? path.join(distDir, 'index.html')
                : url.pathname === '/pancake-docs.pancake' ? artifactPath
                : path.join(distDir, path.normalize(url.pathname).replace(/^([.][.][/\\])+/, ''));
        }
        if (!filePath.startsWith(distDir) && filePath !== artifactPath) {
            res.writeHead(403).end();
            return;
        }
        if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
            res.writeHead(404).end('not found');
            return;
        }
        const size = fs.statSync(filePath).size;
        const type = TYPES[path.extname(filePath)] || 'application/octet-stream';
        const cacheControl = immutable ? 'public, max-age=31536000, immutable' : 'no-cache';
        const range = /^bytes=(\d+)-(\d+)?$/.exec(req.headers.range || '');
        if (range) {
            const start = Number(range[1]);
            const end = Math.min(range[2] !== undefined ? Number(range[2]) : size - 1, size - 1);
            if (start > end || start >= size) {
                res.writeHead(416, { 'Content-Range': `bytes */${size}` }).end();
                return;
            }
            res.writeHead(206, {
                'Content-Type': type,
                'Content-Length': end - start + 1,
                'Content-Range': `bytes ${start}-${end}/${size}`,
                'Accept-Ranges': 'bytes',
                'Cache-Control': cacheControl,
            });
            fs.createReadStream(filePath, { start, end }).pipe(res);
            return;
        }
        res.writeHead(200, { 'Content-Type': type, 'Content-Length': size, 'Accept-Ranges': 'bytes', 'Cache-Control': cacheControl });
        fs.createReadStream(filePath).pipe(res);
    });
}

if (import.meta.url === `file://${process.argv[1]}`) {
    const port = Number(process.argv[2]) || 8790;
    createServer().listen(port, '127.0.0.1', () => {
        console.log(`serving http://127.0.0.1:${port} (range-capable)`);
    });
}
