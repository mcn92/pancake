#!/usr/bin/env node
'use strict';

/**
 * Build a Pancake index from a binary embeddings file.
 *
 * Input format:
 *   uint32_le dims
 *   uint32_le count
 *   8 bytes reserved/padding
 *   count * dims float32_le coordinates
 *
 * Usage:
 *   node examples/cli/build_index.js --embeddings embeddings.bin --output index.pnck
 */

const fs = require('fs');
const { pathToFileURL } = require('url');
const path = require('path');

function readEmbeddings(filePath) {
    const buffer = fs.readFileSync(filePath);
    if (buffer.byteLength < 16) {
        throw new Error('Embeddings file is too small; expected a 16-byte header');
    }

    const dims = buffer.readUInt32LE(0);
    const count = buffer.readUInt32LE(4);
    if (!Number.isInteger(dims) || dims < 1) throw new Error(`Invalid dims: ${dims}`);
    if (!Number.isInteger(count) || count < 1) throw new Error(`Invalid count: ${count}`);

    const expectedBytes = 16 + count * dims * 4;
    if (buffer.byteLength < expectedBytes) {
        throw new Error(`Embeddings file is truncated (${buffer.byteLength} < ${expectedBytes})`);
    }

    const embeddings = new Array(count);
    let offset = 16;
    for (let i = 0; i < count; i++) {
        const vec = new Float32Array(dims);
        for (let j = 0; j < dims; j++) {
            vec[j] = buffer.readFloatLE(offset);
            offset += 4;
        }
        embeddings[i] = vec;
    }

    return { embeddings, dims, count };
}

function parsePositiveInt(value, name) {
    const parsed = Number.parseInt(value, 10);
    if (!Number.isInteger(parsed) || parsed < 1) {
        throw new Error(`${name} must be a positive integer`);
    }
    return parsed;
}

async function loadPancake() {
    const url = pathToFileURL(path.join(__dirname, '..', '..', 'pancake.node.mjs')).href;
    const mod = await import(url);
    return mod.default;
}

async function main() {
    const args = process.argv.slice(2);
    const getArg = (flag, defaultValue) => {
        const idx = args.indexOf(flag);
        return idx !== -1 ? args[idx + 1] : defaultValue;
    };
    const hasFlag = (flag) => args.includes(flag);

    const embeddingsPath = getArg('--embeddings', null);
    const outputPath = getArg('--output', 'index.pnck');
    const M = parsePositiveInt(getArg('--m', '16'), '--m');
    const efConstruction = parsePositiveInt(getArg('--ef-construction', '200'), '--ef-construction');
    const efSearch = parsePositiveInt(getArg('--ef-search', '100'), '--ef-search');
    const quantized = !hasFlag('--no-quantize');

    if (!embeddingsPath) {
        console.log('Usage: node examples/cli/build_index.js --embeddings <file> [options]');
        console.log('');
        console.log('Options:');
        console.log('  --embeddings <path>    Input embeddings (required)');
        console.log('  --output <path>        Output Pancake snapshot (default: index.pnck)');
        console.log('  --m <number>           HNSW M (default: 16)');
        console.log('  --ef-construction <n>  Build quality (default: 200)');
        console.log('  --ef-search <n>        Search quality for restored index (default: 100)');
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

    const Pancake = await loadPancake();
    const startTime = Date.now();
    const index = await Pancake.create({
        dim: dims,
        maxElements: count,
        metric: 'cosine',
        quantized,
        M,
        efConstruction,
        efSearch
    });

    try {
        console.log(`Building ${quantized ? 'int8' : 'float32'} index (M=${M}, efConstruction=${efConstruction})...`);
        const ids = index.addBatch(embeddings);
        const buildTime = (Date.now() - startTime) / 1000;
        console.log(`  Built ${ids.length} vectors in ${buildTime.toFixed(1)}s (${(ids.length / buildTime).toFixed(0)} vec/sec)`);
        console.log('');

        console.log(`Exporting Pancake snapshot to ${outputPath}...`);
        const snapshot = index.export();
        fs.writeFileSync(outputPath, snapshot);
        console.log(`  Written ${(snapshot.byteLength / 1024 / 1024).toFixed(2)} MB`);
        console.log(`  Index memory usage: ${(index.memory / 1024 / 1024).toFixed(2)} MB`);
    } finally {
        index.dispose();
    }
}

main().catch(err => {
    console.error('Error:', err && err.message ? err.message : String(err));
    process.exit(1);
});
