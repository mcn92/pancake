// Service worker for the wiki knowledge pack: cache-first for the immutable
// heavy assets — versioned pack paths (the version segment is a content
// hash, so a rebuilt pack gets new keys automatically), the encoder, and
// the ONNX runtime. Repeat visits boot from disk instead of re-downloading
// ~93 MB.
//
// Range reads arrive with distinct ?r=start-end URLs (the page adds them to
// defeat Chromium's same-URL cache-entry lock), so each range is its own
// cache key. The Cache API rejects 206 responses, so ranges are stored as
// synthetic 200s with the original status noted in a header and restored on
// the way out.
const CACHE_NAME = 'wiki-pack-assets-v1';
const CACHEABLE = /^\/(pack\/v[0-9a-f]{6,}\/|models\/|ort\/)/;

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);
    if (event.request.method !== 'GET' || url.origin !== self.location.origin
        || !CACHEABLE.test(url.pathname)) return;
    const key = url.pathname + url.search;

    event.respondWith((async () => {
        const cache = await caches.open(CACHE_NAME);
        const hit = await cache.match(key);
        if (hit) {
            const status = Number(hit.headers.get('x-sw-status') || 200);
            return new Response(hit.body, { status, headers: hit.headers });
        }
        const response = await fetch(event.request);
        if (response.status === 200 || response.status === 206) {
            const copy = response.clone();
            const headers = new Headers(copy.headers);
            headers.set('x-sw-status', String(response.status));
            event.waitUntil(cache.put(key, new Response(copy.body, { status: 200, headers })));
        }
        return response;
    })());
});
