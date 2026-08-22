export function httpRangeSource(url, options = {}) {
    const stats = { requests: 0, bytes: 0, acceptRanges: null, etag: null, fullFallback: false };
    const maxFullFallbackBytes = options.maxFullFallbackBytes ?? 64 * 1024 * 1024;
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
            const sep = url.includes('?') ? '&' : '?';
            const headers = { Range: `bytes=${offset}-${end}` };
            if (stats.etag) headers['If-Range'] = stats.etag;
            const response = await fetch(`${url}${sep}r=${offset}-${end}`, { headers });

            if (response.status === 206) {
                // A 206 must be the range we asked for: a host (or cache)
                // that answers with a different slice would otherwise be
                // caught only downstream, by the digest on whatever that
                // slice happens to be part of.
                const contentRange = response.headers.get('content-range');
                const m = contentRange && /^bytes (\d+)-(\d+)\//.exec(contentRange);
                if (m && (Number(m[1]) !== offset || Number(m[2]) > end)) {
                    throw new Error(`range read returned bytes ${m[1]}-${m[2]}, requested ${offset}-${end}`);
                }
                return new Uint8Array(await response.arrayBuffer());
            }
            if (response.status === 200) {
                if (!full) {
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
                    full = new Uint8Array(await response.arrayBuffer());
                    stats.fullFallback = true;
                    console.warn('.pancake host does not honor Range; fell back to a one-time full download');
                }
                return full.subarray(offset, offset + length);
            }
            throw new Error(`range read failed: HTTP ${response.status}`);
        },
    };
}
