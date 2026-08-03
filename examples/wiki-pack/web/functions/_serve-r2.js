// Shared handler: serve an R2 object with real 206 Range support, immutable
// cache headers, edge caching, and CORS.
//
// Edge cache: identical ranges (the resident-tier open, the encoder, the
// offsets table) are cached at the colo keyed by url+range, so only the
// first visitor per colo pays the R2 read. The Cache API refuses 206
// responses, so ranges are stored as synthetic 200s under a range-suffixed
// key and re-wrapped on hit.
//
// CORS: the demo shards range traffic across Pages branch aliases to escape
// the browser's one-connection-one-isolate serialization (a single h2
// connection funnels every subrequest through one isolate's ~6-connection
// cap to R2). Cross-origin fetches from the shard origins need these
// headers; Range is not a CORS-safelisted header, so OPTIONS is answered
// and cached for a day per path.
//
// Objects over wrangler's 300 MiB upload ceiling are stored as fixed-size
// parts (<key>.p0, <key>.p1, ...) and stitched here; SPLIT_OBJECTS records
// the split geometry chosen at upload time.
const SPLIT_OBJECTS = {
    'pack/corpus.bin': { partSize: 209715200, total: 332294781, parts: 2 },
};

const CORS = {
    'Access-Control-Allow-Origin': '*',
    'Access-Control-Expose-Headers': 'Content-Range, Content-Length, Accept-Ranges, ETag',
};

function parseRange(request) {
    const header = request.headers.get('Range');
    const match = header && /^bytes=(\d+)-(\d*)$/.exec(header);
    if (!match) return null;
    return { offset: Number(match[1]), end: match[2] === '' ? undefined : Number(match[2]) };
}

function baseHeaders(contentType, etag) {
    return new Headers({
        'Content-Type': contentType || 'application/octet-stream',
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'public, max-age=31536000, immutable',
        ...(etag ? { ETag: etag } : {}),
        ...CORS,
    });
}

export function onOptions() {
    return new Response(null, {
        status: 204,
        headers: {
            ...CORS,
            'Access-Control-Allow-Methods': 'GET, OPTIONS',
            'Access-Control-Allow-Headers': 'Range',
            'Access-Control-Max-Age': '86400',
        },
    });
}

function cacheKeyFor(request, range) {
    const url = new URL(request.url);
    url.search = range ? `?r=${range.offset}-${range.end ?? ''}` : '';
    return new Request(url.toString(), { method: 'GET' });
}

async function serveSplit(env, key, spec, range, waitUntil) {
    const offset = range ? range.offset : 0;
    const end = range ? Math.min(range.end ?? spec.total - 1, spec.total - 1) : spec.total - 1;
    if (offset > end || offset >= spec.total) return new Response('range not satisfiable', { status: 416, headers: CORS });

    const slices = [];
    for (let p = Math.floor(offset / spec.partSize); p <= Math.floor(end / spec.partSize) && p < spec.parts; p++) {
        const partStart = p * spec.partSize;
        const from = Math.max(offset, partStart) - partStart;
        const to = Math.min(end, partStart + spec.partSize - 1) - partStart;
        slices.push({ key: `${key}.p${p}`, offset: from, length: to - from + 1 });
    }

    const headers = baseHeaders('application/octet-stream');
    headers.set('Content-Length', String(end - offset + 1));
    if (range) headers.set('Content-Range', `bytes ${offset}-${end}/${spec.total}`);

    if (slices.length === 1) {
        const object = await env.PACK.get(slices[0].key, { range: { offset: slices[0].offset, length: slices[0].length } });
        if (!object) return new Response('not found', { status: 404, headers: CORS });
        return new Response(object.body, { status: range ? 206 : 200, headers });
    }

    const { readable, writable } = new TransformStream();
    const pump = (async () => {
        try {
            for (const slice of slices) {
                const object = await env.PACK.get(slice.key, { range: { offset: slice.offset, length: slice.length } });
                if (!object) throw new Error(`missing part ${slice.key}`);
                await object.body.pipeTo(writable, { preventClose: true });
            }
            await writable.close();
        } catch (e) {
            await writable.abort(e);
        }
    })();
    waitUntil(pump);
    return new Response(readable, { status: range ? 206 : 200, headers });
}

export async function serveFromR2(context, prefix) {
    const { env, params, request } = context;
    const key = `${prefix}/${Array.isArray(params.path) ? params.path.join('/') : params.path}`;
    const range = parseRange(request);

    // Edge cache first: hits skip R2 (and the isolate connection cap) entirely.
    const cache = caches.default;
    const cacheKey = cacheKeyFor(request, range);
    const hit = await cache.match(cacheKey);
    if (hit) {
        const headers = new Headers(hit.headers);
        if (range) {
            return new Response(hit.body, { status: 206, headers });
        }
        return new Response(hit.body, { status: 200, headers });
    }

    let response;
    const split = SPLIT_OBJECTS[key];
    if (split) {
        response = await serveSplit(env, key, split, range, context.waitUntil.bind(context));
    } else {
        let object;
        if (range) {
            object = await env.PACK.get(key, {
                range: range.end === undefined ? { offset: range.offset } : { offset: range.offset, length: range.end - range.offset + 1 },
            });
        } else {
            object = await env.PACK.get(key);
        }
        if (!object) return new Response('not found', { status: 404, headers: CORS });

        const headers = baseHeaders(object.httpMetadata?.contentType, object.httpEtag);
        if (range) {
            const end = range.end === undefined ? object.size - 1 : Math.min(range.end, object.size - 1);
            headers.set('Content-Range', `bytes ${range.offset}-${end}/${object.size}`);
            headers.set('Content-Length', String(end - range.offset + 1));
            response = new Response(object.body, { status: 206, headers });
        } else {
            headers.set('Content-Length', String(object.size));
            response = new Response(object.body, { status: 200, headers });
        }
    }

    if (response.status === 200 || response.status === 206) {
        // Store as a synthetic 200 (the Cache API rejects 206); the range
        // lives in the key. Content-Range survives in the stored headers and
        // is re-sent on hits.
        const [body, cacheBody] = response.body.tee();
        context.waitUntil(cache.put(cacheKey, new Response(cacheBody, { status: 200, headers: response.headers })));
        return new Response(body, { status: response.status, headers: response.headers });
    }
    return response;
}
