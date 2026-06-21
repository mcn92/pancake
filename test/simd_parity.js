'use strict';

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawnSync } = require('child_process');
const createPancakeApi = require('../pancake-core.js');

const ROOT = path.resolve(__dirname, '..');
const DIST = path.join(ROOT, 'dist');
const SCALAR_BASENAME = 'engine.scalar';

function makeRng(seed) {
    let state = seed >>> 0;
    return () => {
        state = (1664525 * state + 1013904223) >>> 0;
        return state / 0x100000000;
    };
}

function randomFloatArray(length, rng) {
    const out = new Float32Array(length);
    for (let i = 0; i < length; i++) out[i] = (rng() * 2 - 1) * 3;
    return out;
}

function randomIntArray(length, maxExclusive, rng) {
    const out = new Int32Array(length);
    for (let i = 0; i < length; i++) out[i] = Math.floor(rng() * maxExclusive);
    return out;
}

function denseMatmulRef(matrix, vec, bias, rows, cols) {
    const out = new Float32Array(rows);
    for (let i = 0; i < rows; i++) {
        let sum = bias[i];
        const rowOff = i * cols;
        for (let j = 0; j < cols; j++) sum += matrix[rowOff + j] * vec[j];
        out[i] = sum;
    }
    return out;
}

function sparseMatmulRef(matrix, indices, values, bias, rows, cols) {
    const out = new Float32Array(bias);
    for (let k = 0; k < indices.length; k++) {
        const j = indices[k];
        const val = values[k];
        for (let i = 0; i < rows; i++) out[i] += matrix[i * cols + j] * val;
    }
    return out;
}

function normalizeRef(vec) {
    let normSq = 0;
    for (let i = 0; i < vec.length; i++) normSq += vec[i] * vec[i];
    const out = new Float32Array(vec);
    const inv = normSq > 0 ? 1 / Math.sqrt(normSq) : 0;
    for (let i = 0; i < out.length; i++) out[i] *= inv;
    return out;
}

function randomUnitArray(length, rng) {
    const out = randomFloatArray(length, rng);
    let normSq = 0;
    for (let i = 0; i < out.length; i++) normSq += out[i] * out[i];
    const inv = normSq > 0 ? 1 / Math.sqrt(normSq) : 0;
    for (let i = 0; i < out.length; i++) out[i] *= inv;
    return out;
}

function assertFloatArraysNear(actual, expected, tolerance, label) {
    assert.strictEqual(actual.length, expected.length, `${label}: length mismatch`);
    for (let i = 0; i < actual.length; i++) {
        const delta = Math.abs(actual[i] - expected[i]);
        assert.ok(
            delta <= tolerance,
            `${label}: mismatch at ${i}, got ${actual[i]}, expected ${expected[i]}, delta ${delta}`
        );
    }
}

function assertSearchResultsNear(actual, expected, tolerance, label) {
    assert.strictEqual(actual.length, expected.length, `${label}: length mismatch`);
    for (let i = 0; i < actual.length; i++) {
        assert.strictEqual(actual[i].id, expected[i].id, `${label}: id mismatch at ${i}`);
        const delta = Math.abs(actual[i].distance - expected[i].distance);
        assert.ok(
            delta <= tolerance,
            `${label}: distance mismatch at ${i}, got ${actual[i].distance}, expected ${expected[i].distance}, delta ${delta}`
        );
    }
}

function newestMtimeMs(paths) {
    let newest = 0;
    for (const file of paths) {
        const stat = fs.statSync(file);
        if (stat.mtimeMs > newest) newest = stat.mtimeMs;
    }
    return newest;
}

function ensureScalarBuild() {
    const scalarJs = path.join(DIST, `${SCALAR_BASENAME}.js`);
    const scalarWasm = path.join(DIST, `${SCALAR_BASENAME}.wasm`);
    const buildInputs = [
        path.join(ROOT, 'build.sh'),
        path.join(ROOT, 'src', 'engine.cpp'),
        path.join(ROOT, 'src', 'float_hnsw.hpp'),
        path.join(ROOT, 'src', 'int8_float_hnsw.hpp'),
    ];
    if (fs.existsSync(scalarJs) && fs.existsSync(scalarWasm)) {
        const scalarMtime = Math.min(fs.statSync(scalarJs).mtimeMs, fs.statSync(scalarWasm).mtimeMs);
        if (scalarMtime >= newestMtimeMs(buildInputs)) return;
    }

    const result = spawnSync('bash', ['build.sh'], {
        cwd: ROOT,
        env: {
            ...process.env,
            OUT_BASENAME: SCALAR_BASENAME,
            WASM_SIMD: '0',
            PATCH_ENGINE_JS: '0',
            EM_CACHE: process.env.EM_CACHE || path.join(os.tmpdir(), 'pancake-emscripten-cache'),
        },
        encoding: 'utf8',
    });

    if (result.status !== 0) {
        process.stderr.write(result.stdout || '');
        process.stderr.write(result.stderr || '');
        throw new Error('Failed to build scalar WASM artifact for SIMD parity test');
    }
}

async function loadModule(basename) {
    const factory = require(path.join(DIST, `${basename}.js`));
    const wasmBinary = fs.readFileSync(path.join(DIST, `${basename}.wasm`));
    return factory({
        wasmBinary,
        locateFile(file) {
            return path.join(DIST, file);
        },
    });
}

function makePancakeApi(basename) {
    return createPancakeApi(() => loadModule(basename));
}

function allocF32(mod, values) {
    const ptr = mod._emsc_malloc(values.byteLength);
    mod.HEAPF32.set(values, ptr >> 2);
    return ptr;
}

function allocI32(mod, values) {
    const ptr = mod._emsc_malloc(values.byteLength);
    mod.HEAP32.set(values, ptr >> 2);
    return ptr;
}

function readF32(mod, ptr, length) {
    return new Float32Array(mod.HEAPF32.buffer.slice(ptr, ptr + length * 4));
}

function freePtrs(mod, ptrs) {
    for (const ptr of ptrs) mod._emsc_free(ptr);
}

function runDenseMatmul(mod, matrix, vec, bias, rows, cols) {
    const matrixPtr = allocF32(mod, matrix);
    const vecPtr = allocF32(mod, vec);
    const biasPtr = allocF32(mod, bias);
    const outPtr = mod._emsc_malloc(rows * 4);
    try {
        mod._dense_matmul(matrixPtr, vecPtr, biasPtr, outPtr, rows, cols);
        return readF32(mod, outPtr, rows);
    } finally {
        freePtrs(mod, [matrixPtr, vecPtr, biasPtr, outPtr]);
    }
}

function runSparseMatmul(mod, matrix, indices, values, bias, rows, cols) {
    const matrixPtr = allocF32(mod, matrix);
    const indicesPtr = allocI32(mod, indices);
    const valuesPtr = allocF32(mod, values);
    const biasPtr = allocF32(mod, bias);
    const outPtr = mod._emsc_malloc(rows * 4);
    try {
        mod._sparse_matmul(matrixPtr, indicesPtr, valuesPtr, indices.length, biasPtr, outPtr, rows, cols);
        return readF32(mod, outPtr, rows);
    } finally {
        freePtrs(mod, [matrixPtr, indicesPtr, valuesPtr, biasPtr, outPtr]);
    }
}

function runNormalize(mod, vec) {
    const ptr = allocF32(mod, vec);
    try {
        mod._normalize(ptr, vec.length);
        return readF32(mod, ptr, vec.length);
    } finally {
        freePtrs(mod, [ptr]);
    }
}

async function main() {
    ensureScalarBuild();

    const simd = await loadModule('engine');
    const scalar = await loadModule(SCALAR_BASENAME);
    const PancakeSimd = makePancakeApi('engine');
    const PancakeScalar = makePancakeApi(SCALAR_BASENAME);
    const rng = makeRng(0xC0FFEE);

    const denseCases = [
        { rows: 5, cols: 7 },
        { rows: 9, cols: 16 },
        { rows: 11, cols: 19 },
    ];
    for (const { rows, cols } of denseCases) {
        const matrix = randomFloatArray(rows * cols, rng);
        const vec = randomFloatArray(cols, rng);
        const bias = randomFloatArray(rows, rng);
        const ref = denseMatmulRef(matrix, vec, bias, rows, cols);
        const simdOut = runDenseMatmul(simd, matrix, vec, bias, rows, cols);
        const scalarOut = runDenseMatmul(scalar, matrix, vec, bias, rows, cols);
        assertFloatArraysNear(simdOut, ref, 1e-5, `dense ref ${rows}x${cols}`);
        assertFloatArraysNear(scalarOut, ref, 1e-5, `dense scalar ${rows}x${cols}`);
        assertFloatArraysNear(simdOut, scalarOut, 1e-5, `dense parity ${rows}x${cols}`);
    }

    const sparseCases = [
        { rows: 6, cols: 13, nnz: 4 },
        { rows: 10, cols: 17, nnz: 7 },
    ];
    for (const { rows, cols, nnz } of sparseCases) {
        const matrix = randomFloatArray(rows * cols, rng);
        const indices = randomIntArray(nnz, cols, rng);
        const values = randomFloatArray(nnz, rng);
        const bias = randomFloatArray(rows, rng);
        const ref = sparseMatmulRef(matrix, indices, values, bias, rows, cols);
        const simdOut = runSparseMatmul(simd, matrix, indices, values, bias, rows, cols);
        const scalarOut = runSparseMatmul(scalar, matrix, indices, values, bias, rows, cols);
        assertFloatArraysNear(simdOut, ref, 1e-5, `sparse ref ${rows}x${cols}`);
        assertFloatArraysNear(scalarOut, ref, 1e-5, `sparse scalar ${rows}x${cols}`);
        assertFloatArraysNear(simdOut, scalarOut, 1e-5, `sparse parity ${rows}x${cols}`);
    }

    const normCases = [3, 4, 15, 32];
    for (const dim of normCases) {
        const vec = randomFloatArray(dim, rng);
        const ref = normalizeRef(vec);
        const simdOut = runNormalize(simd, vec);
        const scalarOut = runNormalize(scalar, vec);
        assertFloatArraysNear(simdOut, ref, 1e-5, `normalize ref dim=${dim}`);
        assertFloatArraysNear(scalarOut, ref, 1e-5, `normalize scalar dim=${dim}`);
        assertFloatArraysNear(simdOut, scalarOut, 1e-5, `normalize parity dim=${dim}`);
    }

    const annScenarios = [
        { label: 'float-cosine', dim: 64, metric: 'cosine', quantized: false, count: 120, k: 8 },
        { label: 'float-l2', dim: 48, metric: 'l2', quantized: false, count: 100, k: 6 },
        { label: 'int8-cosine', dim: 96, metric: 'cosine', quantized: true, count: 140, k: 8 },
        { label: 'int8-l2', dim: 72, metric: 'l2', quantized: true, count: 110, k: 6 },
    ];

    for (const scenario of annScenarios) {
        const opts = {
            dim: scenario.dim,
            metric: scenario.metric,
            quantized: scenario.quantized,
            maxElements: scenario.count + 32,
            M: 16,
            efConstruction: 200,
            efSearch: 80,
        };
        const idxSimd = await PancakeSimd.create(opts);
        const idxScalar = await PancakeScalar.create(opts);
        try {
            const vectors = Array.from({ length: scenario.count }, () =>
                scenario.metric === 'cosine'
                    ? randomUnitArray(scenario.dim, rng)
                    : randomFloatArray(scenario.dim, rng)
            );

            const idsSimd = idxSimd.addBatch(vectors);
            const idsScalar = idxScalar.addBatch(vectors);
            assert.deepStrictEqual(idsSimd, idsScalar, `${scenario.label}: addBatch IDs match`);

            const deleteIds = [idsSimd[5], idsSimd[17], idsSimd[29]];
            for (const id of deleteIds) {
                idxSimd.delete(id);
                idxScalar.delete(id);
            }

            const queries = [
                vectors[0],
                vectors[Math.floor(scenario.count / 3)],
                vectors[scenario.count - 1],
                scenario.metric === 'cosine'
                    ? randomUnitArray(scenario.dim, rng)
                    : randomFloatArray(scenario.dim, rng),
            ];

            for (let i = 0; i < queries.length; i++) {
                const simdResults = idxSimd.search(queries[i], scenario.k);
                const scalarResults = idxScalar.search(queries[i], scenario.k);
                assertSearchResultsNear(simdResults, scalarResults, 1e-4, `${scenario.label}: search parity before compact q${i}`);
            }

            idxSimd.compact();
            idxScalar.compact();
            assert.strictEqual(idxSimd.count, idxScalar.count, `${scenario.label}: count parity after compact`);

            for (let i = 0; i < queries.length; i++) {
                const simdResults = idxSimd.search(queries[i], scenario.k);
                const scalarResults = idxScalar.search(queries[i], scenario.k);
                assertSearchResultsNear(simdResults, scalarResults, 1e-4, `${scenario.label}: search parity after compact q${i}`);
            }

            const nextVec = scenario.metric === 'cosine'
                ? randomUnitArray(scenario.dim, rng)
                : randomFloatArray(scenario.dim, rng);
            const nextSimdId = idxSimd.add(nextVec);
            const nextScalarId = idxScalar.add(nextVec);
            assert.strictEqual(nextSimdId, nextScalarId, `${scenario.label}: add parity after compact`);

            const exported = idxSimd.export();
            const restoredScalar = await PancakeScalar.create(opts);
            try {
                restoredScalar.import(exported);
                for (let i = 0; i < queries.length; i++) {
                    const simdResults = idxSimd.search(queries[i], scenario.k);
                    const scalarResults = restoredScalar.search(queries[i], scenario.k);
                    assertSearchResultsNear(simdResults, scalarResults, 1e-4, `${scenario.label}: simd export -> scalar import parity q${i}`);
                }
            } finally {
                restoredScalar.dispose();
            }
        } finally {
            idxSimd.dispose();
            idxScalar.dispose();
        }
    }

    console.log('SIMD parity checks passed.');
}

main().catch((err) => {
    console.error(err.stack || err);
    process.exit(1);
});
