#!/usr/bin/env node
/**
 * Search HNSW index with text queries
 *
 * Usage:
 *   node search_index.js --index index.bin --metadata metadata.json --query "search text" [options]
 *
 * Options:
 *   --index <path>      Index file (required)
 *   --metadata <path>   Metadata JSON (required)
 *   --query <text>      Search query (required for single search)
 *   --k <number>        Number of results (default: 10)
 *   --ef <number>       Search quality parameter (default: 100)
 *   --interactive       Interactive mode
 */

const fs = require('fs');
const readline = require('readline');

async function loadWASM() {
    const path = require('path');
    const Pancake = require('../../dist/engine.js');
    const wasmBuf = fs.readFileSync(path.join(__dirname, '..', '..', 'dist', 'engine.wasm'));
    return await Pancake({ wasmBinary: wasmBuf.buffer.slice(wasmBuf.byteOffset, wasmBuf.byteOffset + wasmBuf.byteLength) });
}

function loadMetadata(path) {
    return JSON.parse(fs.readFileSync(path, 'utf8'));
}

function loadIndex(engine, path, quantized, dim) {
    const indexData = fs.readFileSync(path);
    const dataPtr = engine._emsc_malloc(indexData.length);
    engine.HEAPU8.set(indexData, dataPtr);

    // Initialize the backend before importing to set g_float_dim (required for float_import_index)
    if (!quantized) {
        // Initialize with correct dimension and reasonable defaults for capacity/graph params
        // The actual graph structure will be loaded from the serialized data
        const initStatus = engine.ccall('float_init', 'number',
            ['number', 'number', 'number', 'number', 'number', 'number'],
            [dim, 100000, 32, 200, 128, 0]);  // dim, max_elem, M, ef_construction, ef_search, metric
        if (initStatus !== 0) {
            engine._emsc_free(dataPtr);
            throw new Error('Failed to initialize float backend');
        }
    }

    const importFn = quantized ? 'i8_import_index' : 'float_import_index';
    const status = quantized
        ? engine.ccall(importFn, 'number',
            ['number', 'number', 'number'],
            [dataPtr, indexData.length, dim])
        : engine.ccall(importFn, 'number',
            ['number', 'number'],
            [dataPtr, indexData.length]);

    engine._emsc_free(dataPtr);

    if (status !== 0) {
        throw new Error('Failed to load index');
    }
}

function embedQuery(engine, text) {
    const textPtr = engine._emsc_malloc(text.length + 1);
    const dim = engine.ccall('emb_dimension', 'number', [], []);
    const sizePtr = engine._emsc_malloc(4);

    for (let i = 0; i < text.length; i++) {
        engine.HEAPU8[textPtr + i] = text.charCodeAt(i) & 0x7F;
    }
    engine.HEAPU8[textPtr + text.length] = 0;

    // emb_encode returns pointer to embedding data, not error code
    const resultPtr = engine.ccall('emb_encode', 'number',
        ['number', 'number'],
        [textPtr, sizePtr]);

    if (resultPtr === 0) {
        engine._emsc_free(textPtr);
        engine._emsc_free(sizePtr);
        throw new Error('Failed to embed query');
    }

    const embedding = new Float32Array(dim);
    for (let i = 0; i < dim; i++) {
        embedding[i] = engine.HEAPF32[resultPtr / 4 + i];
    }

    engine._emsc_free(textPtr);
    engine._emsc_free(sizePtr);

    return embedding;
}

function search(engine, queryVec, k, ef, quantized) {
    const dims = queryVec.length;
    const vecPtr = engine._emsc_malloc(dims * 4);
    const idsPtr = engine._emsc_malloc(k * 8);
    const distsPtr = engine._emsc_malloc(k * 4);

    engine.HEAPF32.set(queryVec, vecPtr >> 2);

    // Set ef_search using runtime tuning API (does not reinitialize the index)
    const setEfFn = quantized ? 'i8_set_ef' : 'float_set_ef';
    engine.ccall(setEfFn, 'void', ['number'], [ef]);

    const queryFn = quantized ? 'i8_query' : 'float_query';
    const found = engine.ccall(queryFn, 'number',
        ['number', 'number', 'number', 'number'],
        [vecPtr, k, idsPtr, distsPtr]);

    const results = [];
    const dv = new DataView(engine.HEAPU8.buffer);

    for (let i = 0; i < found; i++) {
        const lo = dv.getUint32(idsPtr + i * 8, true);
        const hi = dv.getUint32(idsPtr + i * 8 + 4, true);
        const id = Number(hi * 0x100000000 + lo);
        const distance = engine.HEAPF32[distsPtr / 4 + i];
        results.push({ id, distance });
    }

    engine._emsc_free(vecPtr);
    engine._emsc_free(idsPtr);
    engine._emsc_free(distsPtr);

    return results;
}

function displayResults(query, results, metadata, embedTime, searchTime) {
    console.log('\n' + '='.repeat(70));
    console.log(`Query: "${query}"`);
    console.log(`Embedding: ${embedTime.toFixed(2)}ms | Search: ${searchTime.toFixed(2)}ms | Total: ${(embedTime + searchTime).toFixed(2)}ms`);
    console.log('='.repeat(70));
    console.log('');

    if (results.length === 0) {
        console.log('No results found.');
        return;
    }

    results.forEach((result, idx) => {
        const doc = metadata[result.id];
        console.log(`${idx + 1}. [score: ${result.distance.toFixed(4)}] ${doc.text || doc.content || ''}`);
        if (doc.metadata) {
            console.log(`   ${JSON.stringify(doc.metadata)}`);
        }
        console.log('');
    });
}

async function interactiveMode(engine, metadata, k, ef, quantized) {
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout,
        prompt: '\nQuery (or "quit"): '
    });

    console.log('\n' + '='.repeat(70));
    console.log('Interactive Search Mode');
    console.log('='.repeat(70));

    rl.prompt();

    rl.on('line', (line) => {
        const query = line.trim();

        if (query.toLowerCase() === 'quit' || query.toLowerCase() === 'exit') {
            console.log('\nGoodbye!');
            rl.close();
            return;
        }

        if (query.length === 0) {
            rl.prompt();
            return;
        }

        try {
            const embedStart = Date.now();
            const queryVec = embedQuery(engine, query);
            const embedTime = Date.now() - embedStart;

            const searchStart = Date.now();
            const results = search(engine, queryVec, k, ef, quantized);
            const searchTime = Date.now() - searchStart;

            displayResults(query, results, metadata, embedTime, searchTime);
        } catch (err) {
            console.error('Error:', err.message);
        }

        rl.prompt();
    });

    rl.on('close', () => process.exit(0));
}

async function main() {
    const args = process.argv.slice(2);

    const getArg = (flag, defaultValue) => {
        const idx = args.indexOf(flag);
        return idx !== -1 ? args[idx + 1] : defaultValue;
    };

    const hasFlag = (flag) => args.includes(flag);

    const indexPath = getArg('--index', null);
    const metadataPath = getArg('--metadata', null);
    const query = getArg('--query', null);
    const k = parseInt(getArg('--k', '10'));
    const ef = parseInt(getArg('--ef', '100'));
    const interactive = hasFlag('--interactive');
    const quantized = !hasFlag('--float');

    if (!indexPath || !metadataPath) {
        console.log('Usage: node search_index.js --index <file> --metadata <file> [options]');
        console.log('');
        console.log('Options:');
        console.log('  --index <path>      Index file (required)');
        console.log('  --metadata <path>   Metadata JSON (required)');
        console.log('  --query <text>      Search query (single search)');
        console.log('  --k <number>        Results to return (default: 10)');
        console.log('  --ef <number>       Search quality (default: 100)');
        console.log('  --interactive       Interactive mode');
        console.log('  --float             Use float index (default: int8)');
        process.exit(1);
    }

    console.log('='.repeat(70));
    console.log('Pancake Search');
    console.log('='.repeat(70));
    console.log('');

    console.log('Loading metadata...');
    const metadata = loadMetadata(metadataPath);
    console.log(`  ${metadata.length} documents`);
    console.log('');

    console.log('Initializing WASM engine...');
    const engine = await loadWASM();

    console.log('Initializing embedding model...');
    const status = engine.ccall('emb_init', 'number', [], []);
    if (status !== 0) throw new Error('Failed to initialize embedding model');
    const dim = engine.ccall('emb_dimension', 'number', [], []);
    console.log(`  Embedding dimension: ${dim}D`);
    console.log('');

    console.log(`Loading ${quantized ? 'int8' : 'float32'} index from ${indexPath}...`);
    loadIndex(engine, indexPath, quantized, dim);
    console.log('  Index loaded!');
    console.log('');

    if (interactive || !query) {
        await interactiveMode(engine, metadata, k, ef, quantized);
    } else {
        const embedStart = Date.now();
        const queryVec = embedQuery(engine, query);
        const embedTime = Date.now() - embedStart;

        const searchStart = Date.now();
        const results = search(engine, queryVec, k, ef, quantized);
        const searchTime = Date.now() - searchStart;

        displayResults(query, results, metadata, embedTime, searchTime);
    }
}

main().catch(err => {
    console.error('Error:', err.message);
    process.exit(1);
});
