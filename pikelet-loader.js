'use strict';

function toCompileSource(source) {
    if (source instanceof WebAssembly.Module) return source;
    if (source instanceof ArrayBuffer) return source;
    if (typeof SharedArrayBuffer !== 'undefined' && source instanceof SharedArrayBuffer) {
        return source;
    }
    if (ArrayBuffer.isView(source)) {
        return source.buffer.slice(source.byteOffset, source.byteOffset + source.byteLength);
    }
    throw new TypeError(`Unsupported WASM source type: ${typeof source}`);
}

function createCachedModuleLoader(loadSource) {
    const sourcePromises = new Map();
    const modulePromises = new Map();

    function getSource(variant) {
        let promise = sourcePromises.get(variant);
        if (!promise) {
            promise = Promise.resolve().then(() => loadSource(variant));
            sourcePromises.set(variant, promise);
        }
        return promise;
    }

    function getModule(variant) {
        let promise = modulePromises.get(variant);
        if (!promise) {
            promise = getSource(variant).then((source) => {
                const compileSource = toCompileSource(source);
                return compileSource instanceof WebAssembly.Module
                    ? compileSource
                    : WebAssembly.compile(compileSource);
            });
            modulePromises.set(variant, promise);
        }
        return promise;
    }

    async function supports(variant) {
        const source = toCompileSource(await getSource(variant));
        if (source instanceof WebAssembly.Module) return true;
        try {
            return WebAssembly.validate(source);
        } catch {
            return false;
        }
    }

    async function instantiate(factory, variant) {
        const compiled = await getModule(variant);
        return factory({
            instantiateWasm(imports, successCallback) {
                const instance = new WebAssembly.Instance(compiled, imports);
                successCallback(instance, compiled);
                return instance.exports;
            },
        });
    }

    return Object.freeze({ getModule, instantiate, supports });
}

module.exports = { createCachedModuleLoader };
