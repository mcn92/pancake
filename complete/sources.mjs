export function httpRangeSource(url, options = {}) {
    const stats = { requests: 0, bytes: 0, acceptRanges: null, etag: null, fullFallback: false, retries: 0, redirects: 0 };
    const maxFullFallbackBytes = options.maxFullFallbackBytes ?? 64 * 1024 * 1024;
    // Transient statuses are retried with capped exponential backoff
    // (Retry-After honored when sane): CDNs rate-limit the parallel range
    // bursts this reader legitimately issues — GitHub release assets answer
    // 429 under an open's prefetch + first-query fan-out — and failing a
    // whole query over a pressure signal wastes everything already fetched.
    const maxRetries = options.maxRetries ?? 6;
    const RETRYABLE = new Set([429, 502, 503, 504]);
    const retryDelayMs = (attempt, response) => {
        // GitHub's release CDN answers rate-limited range bursts with 429
        // and a Retry-After of up to a minute; honoring it (capped) beats
        // guessing. Without one, exponential backoff capped at 15 s.
        const after = Number(response.headers.get('retry-after'));
        if (Number.isFinite(after) && after >= 0 && after <= 90) return after * 1000;
        return Math.min(15000, 250 * 2 ** attempt);
    };
    const cacheKeyParam = options.cacheKeyParam === undefined ? 'r'
        : (options.cacheKeyParam === null || options.cacheKeyParam === false) ? null
            : String(options.cacheKeyParam);
    let full = null;

    // Redirect memoization. A host that answers with a redirect (GitHub
    // release assets: github.com 302s every hit to a signed CDN URL) would
    // otherwise charge its rate-limited front door once per range read —
    // ~130 hits for one wiki query, and github.com's frontend throttling,
    // not the CDN, is what 429s the burst; the blob host behind the
    // redirect serves parallel 206s happily. Resolve the chain once, pin
    // the final URL, and re-resolve only when it stops being accepted
    // (401/403 — signed URLs expire). Browsers cannot read Location from a
    // cross-origin redirect (opaqueredirect), so resolution degrades to
    // the un-pinned auto-follow behavior there.
    let resolvedTarget = null; // { target, viaRedirect } | null (unresolved or unresolvable)
    let resolving = null;
    let sizeFromHead = null;
    const resolveTarget = async () => {
        if (resolvedTarget) return resolvedTarget;
        resolving ??= (async () => {
            let target = url;
            let viaRedirect = false;
            try {
                for (let hop = 0; hop < 5; hop++) {
                    const probe = await fetch(target, { method: 'HEAD', redirect: 'manual' });
                    const location = probe.headers.get('location');
                    if (probe.status >= 300 && probe.status < 400 && location) {
                        target = new URL(location, target).href;
                        viaRedirect = true;
                        stats.redirects += 1;
                        continue;
                    }
                    if (probe.type === 'opaqueredirect' || probe.status === 0) {
                        // Browser CORS: the chain is invisible; let fetch
                        // auto-follow per read as before.
                        target = url;
                        viaRedirect = false;
                    } else if (probe.ok) {
                        const len = Number(probe.headers.get('content-length'));
                        if (Number.isFinite(len) && len > 0) sizeFromHead = len;
                        stats.acceptRanges = probe.headers.get('accept-ranges');
                        stats.etag = probe.headers.get('etag');
                    }
                    break;
                }
            } catch {
                // HEAD unsupported or blocked: fall back to auto-follow.
                target = url;
                viaRedirect = false;
            }
            resolvedTarget = { target, viaRedirect };
            resolving = null;
            return resolvedTarget;
        })();
        return resolving;
    };

    return {
        stats,
        size: undefined,
        preferredParallelism: options.preferredParallelism ?? 32,
        preferredGapBytes: options.preferredGapBytes ?? 16 * 1024,

        async init() {
            await resolveTarget();
            if (sizeFromHead !== null) this.size = sizeFromHead;
        },

        async read(offset, length) {
            // After a one-time full download (host ignored Range) every read
            // is served from memory; no further requests go out.
            if (full) return full.subarray(offset, offset + length);
            stats.requests += 1;
            stats.bytes += length;
            const end = offset + length - 1;
            const headers = { Range: `bytes=${offset}-${end}` };
            if (stats.etag) headers['If-Range'] = stats.etag;
            const pinned = await resolveTarget();
            // The per-range cache-key query defeats Chromium's same-URL
            // cache-entry lock (concurrent fetches of one cacheable URL
            // serialize on it). Hosts that sign the full query string
            // (S3 presigned URLs and similar) can turn it off with
            // options.cacheKeyParam: null — range reads then share the
            // unmodified URL. A redirect-pinned target never gets the
            // param: signed URLs sign their query string, and appending
            // anything invalidates them.
            const targetFor = ({ target, viaRedirect }) => (cacheKeyParam && !viaRedirect
                ? `${target}${target.includes('?') ? '&' : '?'}${cacheKeyParam}=${offset}-${end}`
                : target);
            let response = await fetch(targetFor(pinned), { headers });
            if ((response.status === 401 || response.status === 403) && pinned.viaRedirect) {
                // The pinned signed URL expired; resolve a fresh one once
                // and retry. Concurrent reads share the re-resolution.
                if (resolvedTarget === pinned) resolvedTarget = null;
                stats.retries += 1;
                response = await fetch(targetFor(await resolveTarget()), { headers });
            }
            for (let attempt = 0; RETRYABLE.has(response.status) && attempt < maxRetries; attempt++) {
                stats.retries += 1;
                await new Promise((resolve) => setTimeout(resolve, retryDelayMs(attempt, response)));
                response = await fetch(targetFor(resolvedTarget || pinned), { headers });
            }

            if (response.status === 206) {
                // A 206 must be the range we asked for, and must say so: a
                // host (or cache) answering with a different slice would
                // otherwise be caught only downstream, by the digest on
                // whatever that slice happens to be part of — and the lazy
                // sketch rows have no per-row digest yet. RFC 9110 requires
                // Content-Range on a 206, so its absence fails closed too.
                const contentRange = response.headers.get('content-range');
                const m = contentRange && /^bytes (\d+)-(\d+)\//.exec(contentRange);
                if (!m || Number(m[1]) !== offset || Number(m[2]) > end) {
                    throw new Error(`range read returned Content-Range ${contentRange || '(missing)'}, requested bytes ${offset}-${end}`);
                }
                const body = new Uint8Array(await response.arrayBuffer());
                if (body.length !== Number(m[2]) - Number(m[1]) + 1) {
                    throw new Error(`range read body is ${body.length} bytes for Content-Range ${contentRange}`);
                }
                return body;
            }
            if (response.status === 200) {
                // An unknown size must refuse, not pass the gate: with no
                // Content-Length (chunked body) and init() never called,
                // size would be 0 and the cap check silently vacuous.
                const size = Number(response.headers.get('content-length')) || this.size || 0;
                if (size <= 0) {
                    throw new Error('host ignores Range and reports no size; refusing unbounded full download (call init() first or fix the host)');
                }
                if (size > maxFullFallbackBytes) {
                    throw new Error(`host ignores Range and the file is ${size} bytes; refusing full download`);
                }
                // Declared sizes are advisory (a chunked body can stream more
                // than any header claimed), so the cap is enforced on the
                // bytes actually received, aborting as soon as it is crossed.
                const reader = response.body?.getReader?.();
                if (!reader) {
                    const bytes = new Uint8Array(await response.arrayBuffer());
                    if (bytes.length > maxFullFallbackBytes) {
                        throw new Error(`host ignores Range and streamed ${bytes.length} bytes; refusing full download over ${maxFullFallbackBytes}`);
                    }
                    full = bytes;
                } else {
                    const chunks = [];
                    let received = 0;
                    for (;;) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        received += value.byteLength;
                        if (received > maxFullFallbackBytes) {
                            await reader.cancel().catch(() => {});
                            throw new Error(`host ignores Range and streamed over ${maxFullFallbackBytes} bytes; aborting full download`);
                        }
                        chunks.push(value);
                    }
                    full = new Uint8Array(received);
                    let at = 0;
                    for (const chunk of chunks) { full.set(chunk, at); at += chunk.byteLength; }
                }
                stats.fullFallback = true;
                console.warn('.pancake host does not honor Range; fell back to a one-time full download');
                return full.subarray(offset, offset + length);
            }
            throw new Error(`range read failed: HTTP ${response.status}`);
        },
    };
}
