#!/usr/bin/env node
/**
 * Build HNSW index from embeddings
 *
 * Usage:
 *   node build_index.js --embeddings embeddings.bin --output index.bin [options]
 *
 * Options:
 *   --embeddings <path>    Input embeddings file (required)
 *   --output <path>        Output index file (default: index.bin)
 *   --m <number>           HNSW M parameter (default: 16)
 *   --ef-construction <n>  HNSW ef_construction (default: 200)
 *   --quantize             Use int8 quantization (default: true)
 */

const fs = require('fs');

async function loadWASM() {
    const path = require('path');
    const Pancake = require('../../dist/engine.js');
    const wasmBuf = fs.readFileSync(path.join(__dirname, '..', '..', 'dist', 'engine.wasm'));
    return await Pancake({ wasmBinary: wasmBuf.buffer.slice(wasmBuf.byteOffset, wasmBuf.byteOffset + wasmBuf.byteLength) });
}

function readEmbeddings(path) {
    const buffer = fs.readFileSync(path);
    const dims = buffer.readUInt32LE(0);
    const count = buffer.readUInt32LE(4);

    const embeddings = new Float32Array(count * dims);
    let offset = 16; // Skip header

    for (let i = 0; i < count * dims; i++) {
        embeddings[i] = buffer.readFloatLE(offset);
        offset += 4;
    }

    return { embeddings, dims, count };
}

async function main() {
    const args = process.argv.slice(2);

    const getArg = (flag, defaultValue) => {
        const idx = args.indexOf(flag);
        return idx !== -1 ? args[idx + 1] : defaultValue;
    };

    const hasFlag = (flag) => args.includes(flag);

    const embeddingsPath = getArg('--embeddings', null);
    const outputPath = getArg('--output', 'index.bin');
    const M = parseInt(getArg('--m', '16'));
    const efConstruction = parseInt(getArg('--ef-construction', '200'));
    const quantize = !hasFlag('--no-quantize');

    if (!embeddingsPath) {
        console.log('Usage: node build_index.js --embeddings <file> [options]');
        console.log('');
        console.log('Options:');
        console.log('  --embeddings <path>    Input embeddings (required)');
        console.log('  --output <path>        Output index (default: index.bin)');
        console.log('  --m <number>           HNSW M (default: 16)');
        console.log('  --ef-construction <n>  Build quality (default: 200)');
        console.log('  --no-quantize          Disable int8 quantization');
        process.exit(1);
    }

    console.log('='.repeat(70));
    console.log('Pancake Index Builder');
    console.log('='.repeat(70));
    console.log('');

    console.log(`Reading embeddings from ${embeddingsPath}...`);
    const { embeddings, dims, count } = readEmbeddings(embeddingsPath);
    console.log(`  ${count} vectors, ${dims}D`);
    console.log('');

    console.log('Initializing WASM engine...');
    const engine = await loadWASM();
    console.log('');

    const indexType = quantize ? 'Int8 HNSW' : 'Float32 HNSW';
    console.log(`Building ${indexType} index (M=${M}, ef_construction=${efConstruction})...`);

    const startTime = Date.now();

    if (quantize) {
        const status = engine.ccall('i8_init', 'number',
            ['number', 'number', 'number', 'number', 'number', 'number'],
            [dims, count, 1, M, efConstruction, 100]);

        if (status !== 0) throw new Error('Failed to initialize int8 index');

        // Bulk insert
        const vecPtr = engine._emsc_malloc(dims * 4);
        for (let i = 0; i < count; i++) {
            engine.HEAPF32.set(embeddings.subarray(i * dims, (i + 1) * dims), vecPtr >> 2);
            engine.ccall('i8_add', 'number', ['number'], [vecPtr]);

            if ((i + 1) % 1000 === 0) {
                const elapsed = (Date.now() - startTime) / 1000;
                const rate = (i + 1) / elapsed;
                const eta = (count - i - 1) / rate;
                process.stdout.write(`  Progress: ${i + 1}/${count} (${rate.toFixed(0)} vec/sec, ETA: ${eta.toFixed(0)}s)\r`);
            }
        }
        engine._emsc_free(vecPtr);

    } else {
        const status = engine.ccall('float_init', 'number',
            ['number', 'number', 'number', 'number'],
            [dims, count, M, efConstruction]);

        if (status !== 0) throw new Error('Failed to initialize float index');

        const vecPtr = engine._emsc_malloc(dims * 4);
        for (let i = 0; i < count; i++) {
            engine.HEAPF32.set(embeddings.subarray(i * dims, (i + 1) * dims), vecPtr >> 2);
            engine.ccall('float_add', 'number', ['number', 'number'], [vecPtr, i]);

            if ((i + 1) % 1000 === 0) {
                const elapsed = (Date.now() - startTime) / 1000;
                const rate = (i + 1) / elapsed;
                const eta = (count - i - 1) / rate;
                process.stdout.write(`  Progress: ${i + 1}/${count} (${rate.toFixed(0)} vec/sec, ETA: ${eta.toFixed(0)}s)\r`);
            }
        }
        engine._emsc_free(vecPtr);
    }

    const buildTime = (Date.now() - startTime) / 1000;
    console.log(`  Built ${count} vectors in ${buildTime.toFixed(1)}s (${(count / buildTime).toFixed(0)} vec/sec)`);
    console.log('');

    // Export index
    console.log(`Exporting index to ${outputPath}...`);
    const sizePtr = engine._emsc_malloc(4);
    const exportFn = quantize ? 'i8_export_index' : 'float_export_index';
    const dataPtr = engine.ccall(exportFn, 'number', ['number'], [sizePtr]);
    const size = engine.HEAP32[sizePtr >> 2];

    const indexData = new Uint8Array(engine.HEAPU8.buffer, dataPtr, size);
    fs.writeFileSync(outputPath, indexData);

    engine._emsc_free(sizePtr);
    engine._emsc_free(dataPtr);

    const sizeKB = (size / 1024).toFixed(1);
    const sizeMB = (size / 1024 / 1024).toFixed(2);
    console.log(`  Written ${sizeKB} KB (${sizeMB} MB)`);
    console.log('');

    const memoryFn = quantize ? 'i8_memory' : 'float_memory';
    const memory = engine.ccall(memoryFn, 'number', [], []);
    console.log(`Index memory usage: ${(memory / 1024 / 1024).toFixed(2)} MB`);
    console.log('');

    console.log('Done! Index ready for search.');
    console.log(`  node search_index.js --index ${outputPath} --query "your search text"`);
}

main().catch(err => {
    console.error('Error:', err.message);
    process.exit(1);
});
