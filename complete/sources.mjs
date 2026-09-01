export function httpRangeSource(url, options = {}) {
    const stats = { requests: 0, bytes: 0, acceptRanges: null, etag: null, fullFallback: false, retries: 0 };
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

    return {
        stats,
        size: undefined,
        preferredParallelism: options.preferredParallelism ?? 32,
        preferredGapBytes: options.preferredGapBytes ?? 16 * 1024,

        async init() {
            const head = await fetch(url, { method: 'HEAD' });
            if (head.ok) {
                const len = Number(head.headers.get('content-length'));
                if (Number.isFinite(len) && len > 0) this.size = len;
                stats.acceptRanges = head.headers.get('accept-ranges');
                stats.etag = head.headers.get('etag');
            }
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
            // The per-range cache-key query defeats Chromium's same-URL
            // cache-entry lock (concurrent fetches of one cacheable URL
            // serialize on it). Hosts that sign the full query string
            // (S3 presigned URLs and similar) can turn it off with
            // options.cacheKeyParam: null — range reads then share the
            // unmodified URL.
            const target = cacheKeyParam
                ? `${url}${url.includes('?') ? '&' : '?'}${cacheKeyParam}=${offset}-${end}`
                : url;
            let response = await fetch(target, { headers });
            for (let attempt = 0; RETRYABLE.has(response.status) && attempt < maxRetries; attempt++) {
                stats.retries += 1;
                await new Promise((resolve) => setTimeout(resolve, retryDelayMs(attempt, response)));
                response = await fetch(target, { headers });
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
