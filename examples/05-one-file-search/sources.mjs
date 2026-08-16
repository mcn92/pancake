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
            stats.requests += 1;
            stats.bytes += length;
            const end = offset + length - 1;
            const sep = url.includes('?') ? '&' : '?';
            const headers = { Range: `bytes=${offset}-${end}` };
            if (stats.etag) headers['If-Range'] = stats.etag;
            const response = await fetch(`${url}${sep}r=${offset}-${end}`, { headers });

            if (response.status === 206) {
                return new Uint8Array(await response.arrayBuffer());
            }
            if (response.status === 200) {
                if (!full) {
                    const size = Number(response.headers.get('content-length')) || this.size || 0;
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
