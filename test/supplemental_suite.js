'use strict';

const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const REPO_ROOT = path.resolve(__dirname, '..');
const TMP_ROOT = path.join(REPO_ROOT, '.tmp-test-work');
const _wasmBuf = fs.readFileSync(path.join(REPO_ROOT, 'dist', 'engine.wasm'));
const _wasmBinary = _wasmBuf.buffer.slice(_wasmBuf.byteOffset, _wasmBuf.byteOffset + _wasmBuf.byteLength);

function writeAscii(engine, text, ptr) {
    for (let i = 0; i < text.length; i++) {
        engine.HEAPU8[ptr + i] = text.charCodeAt(i) & 0x7F;
    }
    engine.HEAPU8[ptr + text.length] = 0;
}

function makeTempDir(prefix) {
    if (!fs.existsSync(TMP_ROOT)) {
        fs.mkdirSync(TMP_ROOT, { recursive: true });
    }
    const base = path.join(TMP_ROOT, prefix || 'pancake-supp-');
    return fs.mkdtempSync(base);
}

function runNodeScript(scriptRelPath, args = [], options = {}) {
    const scriptPath = path.join(REPO_ROOT, scriptRelPath);
    const execArgs = [scriptPath, ...args];
    return execFileSync(process.execPath, execArgs, {
        cwd: REPO_ROOT,
        stdio: 'inherit',
        ...options,
    });
}

function readEmbeddingFile(binPath) {
    const buffer = fs.readFileSync(binPath);
    const dim = buffer.readUInt32LE(0);
    const count = buffer.readUInt32LE(4);
    const embeddings = [];
    let offset = 16; // header
    for (let i = 0; i < count; i++) {
        const vec = new Float32Array(dim);
        for (let d = 0; d < dim; d++) {
            vec[d] = buffer.readFloatLE(offset);
            offset += 4;
        }
        embeddings.push(vec);
    }
    return { dim, count, embeddings };
}

module.exports = function registerSupplementalSuites(harness) {
    const { assert, assertNear, section } = harness;

    async function testEmbedDocumentsCli() {
        section('Embedding CLI integration');
        const tmpDir = makeTempDir('pancake-embed-');
        const docsPath = path.join(tmpDir, 'docs.json');
        const outputPath = path.join(tmpDir, 'embeddings.bin');

        const docs = [
            { id: 1, text: 'alpha beta gamma' },
            { id: 2, text: 'alpha beta gamma' }, // duplicate for determinism check
            { id: 3, text: 'kappa lambda mu' },
        ];
        fs.writeFileSync(docsPath, JSON.stringify(docs), 'utf8');

        runNodeScript('examples/cli/embed_documents.js', ['--input', docsPath, '--output', outputPath]);

        assert(fs.existsSync(outputPath), 'embed_documents writes embeddings.bin');
        const metadataPath = outputPath.replace(/\.bin$/, '_metadata.json');
        assert(fs.existsSync(metadataPath), 'embed_documents writes metadata JSON');

        const { dim, count, embeddings } = readEmbeddingFile(outputPath);
        assert(dim === 384, `embedding dimension recorded as 384 (got ${dim})`);
        assert(count === docs.length, `embedding count matches input (${count})`);

        let identical = true;
        for (let i = 0; i < dim; i++) {
            if (Math.abs(embeddings[0][i] - embeddings[1][i]) > 1e-6) {
                identical = false;
                break;
            }
        }
        assert(identical, 'duplicate documents yield identical embeddings');

        const metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
        assert(metadata.length === docs.length, 'metadata length matches docs');
        assert(metadata[0].text.includes('alpha'), 'metadata captures source text');

        fs.rmSync(tmpDir, { recursive: true, force: true });
    }

    async function testEmbeddingApiRoundTrip() {
        section('Embedding API round-trip');
        const loadEngine = require(path.join(REPO_ROOT, 'dist', 'engine.js'));
        const engine = await loadEngine({ wasmBinary: _wasmBinary.slice(0) });

        const initIdx = engine.ccall('pi', 'number', ['number'], [128]);
        assert(initIdx === 0, 'pi() initializes quantized 384D index');

        const embInit = engine.ccall('emb_init', 'number', [], []);
        assert(embInit === 0, 'emb_init() succeeds');

        function addDoc(text) {
            const ptr = engine._emsc_malloc(text.length + 1);
            writeAscii(engine, text, ptr);
            const id = engine.ccall('emb_add', 'number', ['number'], [ptr]);
            engine._emsc_free(ptr);
            return id;
        }

        const docTexts = [
            'searchable contract clause about force majeure obligations',
            'culinary narrative about apples and bananas',
            'yet another force majeure clause example',
        ];
        const ids = docTexts.map(addDoc);
        assert(ids.every(id => id !== 0xFFFFFFFF), 'emb_add returns valid ids');

        const k = 2;
        const idsPtr = engine._emsc_malloc(k * 8);
        const distsPtr = engine._emsc_malloc(k * 4);
        const queryPtr = engine._emsc_malloc(docTexts[0].length + 1);
        writeAscii(engine, docTexts[0], queryPtr);

        const found = engine.ccall('emb_search', 'number',
            ['number', 'number', 'number', 'number'],
            [queryPtr, k, idsPtr, distsPtr]);
        assert(found === k, `emb_search returns requested k results (got ${found})`);

        const dv = new DataView(engine.HEAPU8.buffer);
        const lo = dv.getUint32(idsPtr, true);
        const hi = dv.getUint32(idsPtr + 4, true);
        const topId = hi * 0x100000000 + lo;
        assert(topId === ids[0], 'emb_search returns the matching document id first');

        engine._emsc_free(queryPtr);
        engine._emsc_free(idsPtr);
        engine._emsc_free(distsPtr);
    }

    function embedTextWithEngine(engine, text) {
        const textPtr = engine._emsc_malloc(text.length + 1);
        const sizePtr = engine._emsc_malloc(4);
        writeAscii(engine, text, textPtr);
        const resultPtr = engine.ccall('emb_encode', 'number',
            ['number', 'number'],
            [textPtr, sizePtr]);
        if (resultPtr === 0) {
            engine._emsc_free(textPtr);
            engine._emsc_free(sizePtr);
            throw new Error('emb_encode returned null');
        }
        const dim = engine.ccall('emb_dimension', 'number', [], []);
        const embedding = new Float32Array(dim);
        for (let i = 0; i < dim; i++) {
            embedding[i] = engine.HEAPF32[resultPtr / 4 + i];
        }
        engine._emsc_free(textPtr);
        engine._emsc_free(sizePtr);
        return embedding;
    }

    async function testCliBuildAndSearchPipeline() {
        section('CLI build + search pipeline');
        const tmpDir = makeTempDir('pancake-pipeline-');
        const docsPath = path.join(tmpDir, 'docs.json');
        const embedsPath = path.join(tmpDir, 'embeddings.bin');
        const indexPath = path.join(tmpDir, 'index.bin');
        const docs = [
            { id: 0, text: 'apple banana carrot' },
            { id: 1, text: 'banana carrot date' },
            { id: 2, text: 'dragonfruit elderberry fig' },
        ];
        fs.writeFileSync(docsPath, JSON.stringify(docs), 'utf8');

        runNodeScript('examples/cli/embed_documents.js', ['--input', docsPath, '--output', embedsPath]);
        runNodeScript('examples/cli/build_index.js', ['--embeddings', embedsPath, '--output', indexPath]);

        const { dim } = readEmbeddingFile(embedsPath);
        const engine = await require(path.join(REPO_ROOT, 'dist', 'engine.js'))({ wasmBinary: _wasmBinary.slice(0) });

        const indexData = fs.readFileSync(indexPath);
        const dataPtr = engine._emsc_malloc(indexData.length);
        engine.HEAPU8.set(indexData, dataPtr);
        const importStatus = engine.ccall('i8_import_index', 'number',
            ['number', 'number', 'number'],
            [dataPtr, indexData.length, dim]);
        engine._emsc_free(dataPtr);
        assert(importStatus === 0, 'i8_import_index succeeds');

        const embInit = engine.ccall('emb_init', 'number', [], []);
        assert(embInit === 0, 'emb_init succeeds before custom search');

        const queryVec = embedTextWithEngine(engine, 'apple banana');
        const vecPtr = engine._emsc_malloc(queryVec.length * 4);
        engine.HEAPF32.set(queryVec, vecPtr >> 2);
        const idsPtr = engine._emsc_malloc(16);
        const distsPtr = engine._emsc_malloc(8);

        engine.ccall('i8_set_ef', null, ['number'], [50]);
        const i8count = engine.ccall('i8_count', 'number', [], []);
        const k = Math.min(2, i8count);
        const found = engine.ccall('i8_query', 'number',
            ['number', 'number', 'number', 'number'],
            [vecPtr, k, idsPtr, distsPtr]);
        assert(found >= 1, 'i8_query returns requested neighbors');

        const dv = new DataView(engine.HEAPU8.buffer);
        const topLo = dv.getUint32(idsPtr, true);
        const topHi = dv.getUint32(idsPtr + 4, true);
        const topId = topHi * 0x100000000 + topLo;
        assert(topId < i8count, 'search returns a valid document id');

        engine._emsc_free(vecPtr);
        engine._emsc_free(idsPtr);
        engine._emsc_free(distsPtr);

        fs.rmSync(tmpDir, { recursive: true, force: true });
    }

    return [
        testEmbedDocumentsCli,
        testEmbeddingApiRoundTrip,
        testCliBuildAndSearchPipeline,
    ];
};
