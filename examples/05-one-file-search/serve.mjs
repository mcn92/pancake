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
const TYPES = { '.html': 'text/html; charset=utf-8', '.mjs': 'text/javascript', '.js': 'text/javascript', '.css': 'text/css', '.pancake': 'application/octet-stream' };

export function createServer() {
    return http.createServer((req, res) => {
        const url = new URL(req.url, 'http://localhost');
        let filePath = url.pathname === '/' ? path.join(distDir, 'index.html')
            : url.pathname === '/pancake-docs.pancake' ? path.join(here, 'pancake-docs.pancake')
            : path.join(distDir, path.normalize(url.pathname).replace(/^([.][.][/\\])+/, ''));
        if (!filePath.startsWith(distDir) && !filePath.endsWith('pancake-docs.pancake')) {
            res.writeHead(403).end();
            return;
        }
        if (!fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
            res.writeHead(404).end('not found');
            return;
        }
        const size = fs.statSync(filePath).size;
        const type = TYPES[path.extname(filePath)] || 'application/octet-stream';
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
            });
            fs.createReadStream(filePath, { start, end }).pipe(res);
            return;
        }
        res.writeHead(200, { 'Content-Type': type, 'Content-Length': size, 'Accept-Ranges': 'bytes' });
        fs.createReadStream(filePath).pipe(res);
    });
}

if (import.meta.url === `file://${process.argv[1]}`) {
    const port = Number(process.argv[2]) || 8790;
    createServer().listen(port, '127.0.0.1', () => {
        console.log(`serving http://127.0.0.1:${port} (range-capable)`);
    });
}
