'use strict';

function createHttpRangeSource(url, fetchImpl = fetch) {
    return {
        async read(offset, length) {
            const response = await fetchImpl(url, {
                headers: {
                    Range: `bytes=${offset}-${offset + length - 1}`,
                },
            });
            if (response.status !== 206 && response.status !== 200) {
                throw new Error(`HTTP range read failed: ${response.status} ${response.statusText}`.trim());
            }
            const bytes = new Uint8Array(await response.arrayBuffer());
            if (bytes.byteLength !== length) {
                throw new Error(`HTTP range read returned ${bytes.byteLength} bytes; expected ${length}`);
            }
            return bytes;
        },
    };
}

function createR2RangeSource(bucket, key) {
    return {
        async read(offset, length) {
            const object = await bucket.get(key, {
                range: { offset, length },
            });
            if (!object) {
                throw new Error(`R2 object not found: ${key}`);
            }
            const bytes = new Uint8Array(await object.arrayBuffer());
            if (bytes.byteLength !== length) {
                throw new Error(`R2 range read returned ${bytes.byteLength} bytes; expected ${length}`);
            }
            return bytes;
        },
    };
}

module.exports = {
    createHttpRangeSource,
    createR2RangeSource,
};
