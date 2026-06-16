/**
 * pancake_napi.cpp — Node.js N-API binding for Pancake HNSW engine.
 *
 * Wraps the same C++ engine (engine.cpp) that the WASM build uses, but
 * compiled natively with SSE2 SIMD. Provides an identical API surface so
 * the JS benchmark harness can swap between WASM and native transparently.
 *
 * Exposed JS API:
 *   pancake_init(dim, maxElem, quantized, metric, M, efC, efS) -> handle
 *   pancake_add(handle, Float32Array) -> internalId
 *   pancake_bulk_insert(handle, Float32Array, n) -> count
 *   pancake_query(handle, Float32Array, k) -> { ids: Uint32Array, distances: Float32Array, count }
 *   pancake_set_ef(handle, ef)
 *   pancake_delete(handle, id)
 *   pancake_compact(handle)
 *   pancake_count(handle) -> number
 *   pancake_ghost_count(handle) -> number
 *   pancake_ghost_ratio(handle) -> number
 *   pancake_memory(handle) -> number
 *   pancake_dimension(handle) -> number
 *   pancake_export(handle) -> Buffer
 *   pancake_import(handle, Buffer) -> 0 on success
 *   pancake_dispose(handle)
 */

#include <napi.h>

// Pull in the engine directly — it's header-only behind the wrappers.
// We define away the Emscripten-specific bits.
#include "float_hnsw.hpp"
#include "int8_float_hnsw.hpp"

#include <vector>
#include <memory>
#include <cstring>
#include <algorithm>

using namespace pancake::wasm;

namespace {

size_t serialized_index_count_hint(const uint8_t* data, size_t size, uint32_t versioned_magic) {
    if (!data || size < 12) return 100000;

    uint32_t magic = 0;
    std::memcpy(&magic, data, sizeof(magic));

    size_t count_offset = (magic == versioned_magic) ? 12 : 8;
    if (size < count_offset + 4) return 100000;

    uint32_t count = 0;
    std::memcpy(&count, data + count_offset, sizeof(count));
    return static_cast<size_t>(count);
}

} // namespace

// ============================================================================
// Handle table (same structure as engine.cpp)
// ============================================================================

class IndexWrapper {
public:
    virtual ~IndexWrapper() = default;
    virtual uint32_t insert(const float* vec) = 0;
    virtual int bulk_insert(const float* vecs, int n) = 0;
    virtual std::vector<std::pair<uint32_t, float>> search(const float* query, size_t k) = 0;
    virtual std::vector<std::pair<uint32_t, float>> search_filtered(const float* query, size_t k, const uint8_t* bitset, size_t bitset_len) = 0;
    virtual void mark_delete(uint32_t id) = 0;
    virtual void compact() = 0;
    virtual void compact(std::vector<uint32_t>& out_map) = 0;
    virtual size_t count() const = 0;
    virtual size_t ghost_count() const = 0;
    virtual float ghost_ratio() const = 0;
    virtual size_t memory_bytes() const = 0;
    virtual std::vector<uint8_t> serialize() const = 0;
    virtual bool deserialize(const uint8_t* data, size_t size) = 0;
    virtual void set_ef_search(size_t ef) = 0;
    virtual size_t dimension() const = 0;
};

class FloatHNSWWrapper : public IndexWrapper {
    std::unique_ptr<FloatHNSW> impl_;
    FloatHNSWConfig cfg_;
    size_t dims_;
public:
    FloatHNSWWrapper(size_t dims, const FloatHNSWConfig& cfg)
        : impl_(std::make_unique<FloatHNSW>(dims, cfg)), cfg_(cfg), dims_(dims) {}
    uint32_t insert(const float* vec) override { return impl_->insert(vec); }
    int bulk_insert(const float* vecs, int n) override {
        int inserted = 0;
        for (int i = 0; i < n; i++) {
            uint32_t id = impl_->insert(vecs + i * dims_);
            if (id != 0xFFFFFFFF) inserted++;
        }
        return inserted;
    }
    std::vector<std::pair<uint32_t, float>> search(const float* query, size_t k) override { return impl_->search(query, k); }
    std::vector<std::pair<uint32_t, float>> search_filtered(const float* query, size_t k, const uint8_t* bitset, size_t bitset_len) override { return impl_->search_filtered(query, k, bitset, bitset_len); }
    void mark_delete(uint32_t id) override { impl_->mark_delete(id); }
    void compact() override { impl_->compact(); }
    void compact(std::vector<uint32_t>& out_map) override { impl_->compact(out_map); }
    size_t count() const override { return impl_->count(); }
    size_t ghost_count() const override { return impl_->ghost_count(); }
    float ghost_ratio() const override { return impl_->ghost_ratio(); }
    size_t memory_bytes() const override { return impl_->memory_bytes(); }
    std::vector<uint8_t> serialize() const override { return impl_->serialize(); }
    bool deserialize(const uint8_t* data, size_t size) override {
        size_t cnt = serialized_index_count_hint(data, size, 0x464C4831);
        FloatHNSWConfig cfg = cfg_;
        cfg.max_elements = std::max(static_cast<size_t>(cnt * 1.2), static_cast<size_t>(100000));
        impl_ = std::make_unique<FloatHNSW>(dims_, cfg);
        return impl_->deserialize(data, size);
    }
    void set_ef_search(size_t ef) override { impl_->set_ef(ef); }
    size_t dimension() const override { return dims_; }
};

class Int8FloatHNSWWrapper : public IndexWrapper {
    std::unique_ptr<Int8FloatHNSW> impl_;
    Int8FloatHNSWConfig cfg_;
    size_t dims_;
public:
    Int8FloatHNSWWrapper(size_t dims, const Int8FloatHNSWConfig& cfg)
        : impl_(std::make_unique<Int8FloatHNSW>(dims, cfg)), cfg_(cfg), dims_(dims) {}
    uint32_t insert(const float* vec) override { return impl_->insert(vec); }
    int bulk_insert(const float* vecs, int n) override { return impl_->bulk_insert(vecs, n); }
    std::vector<std::pair<uint32_t, float>> search(const float* query, size_t k) override { return impl_->search(query, k); }
    std::vector<std::pair<uint32_t, float>> search_filtered(const float* query, size_t k, const uint8_t* bitset, size_t bitset_len) override { return impl_->search_filtered(query, k, bitset, bitset_len); }
    void mark_delete(uint32_t id) override { impl_->mark_delete(id); }
    void compact() override { impl_->compact(); }
    void compact(std::vector<uint32_t>& out_map) override { impl_->compact(out_map); }
    size_t count() const override { return impl_->count(); }
    size_t ghost_count() const override { return impl_->ghost_count(); }
    float ghost_ratio() const override { return impl_->ghost_ratio(); }
    size_t memory_bytes() const override { return impl_->memory_bytes(); }
    std::vector<uint8_t> serialize() const override { return impl_->serialize(); }
    bool deserialize(const uint8_t* data, size_t size) override {
        size_t cnt = serialized_index_count_hint(data, size, 0x49384831);
        Int8FloatHNSWConfig cfg = cfg_;
        cfg.max_elements = std::max(static_cast<size_t>(cnt * 1.2), static_cast<size_t>(100000));
        impl_ = std::make_unique<Int8FloatHNSW>(dims_, cfg);
        return impl_->deserialize(data, size);
    }
    void set_ef_search(size_t ef) override { impl_->set_ef_search(ef); }
    size_t dimension() const override { return dims_; }
};

constexpr uint32_t MAX_HANDLES = 64;
constexpr uint32_t INVALID_HANDLE = 0xFFFFFFFF;

static IndexWrapper* g_handles[MAX_HANDLES] = {};

static uint32_t alloc_handle() {
    for (uint32_t i = 0; i < MAX_HANDLES; i++) {
        if (!g_handles[i]) return i;
    }
    return INVALID_HANDLE;
}

static void free_handle(uint32_t h) {
    if (h < MAX_HANDLES && g_handles[h]) {
        delete g_handles[h];
        g_handles[h] = nullptr;
    }
}

// ============================================================================
// N-API wrappers
// ============================================================================

Napi::Value Init(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    int dim     = info[0].As<Napi::Number>().Int32Value();
    int maxElem = info[1].As<Napi::Number>().Int32Value();
    int quant   = info[2].As<Napi::Number>().Int32Value();
    int metric  = info[3].As<Napi::Number>().Int32Value();
    int M       = info[4].As<Napi::Number>().Int32Value();
    int efC     = info[5].As<Napi::Number>().Int32Value();
    int efS     = info[6].As<Napi::Number>().Int32Value();

    if (dim <= 0 || maxElem <= 0)
        return Napi::Number::New(env, INVALID_HANDLE);

    uint32_t h = alloc_handle();
    if (h == INVALID_HANDLE)
        return Napi::Number::New(env, INVALID_HANDLE);

    bool use_cosine = (metric == 1);

    if (quant) {
        Int8FloatHNSWConfig cfg;
        cfg.max_elements = maxElem;
        cfg.M = (M > 0) ? M : 32;
        cfg.ef_construction = (efC > 0) ? efC : 200;
        cfg.ef_search = (efS > 0) ? efS : 128;
        cfg.metric = use_cosine ? DistanceMetric::Cosine : DistanceMetric::L2;
        cfg.use_heuristic = true;
        g_handles[h] = new Int8FloatHNSWWrapper(dim, cfg);
    } else {
        FloatHNSWConfig cfg;
        cfg.max_elements = maxElem;
        cfg.M = (M > 0) ? M : 32;
        cfg.ef_construction = (efC > 0) ? efC : 200;
        cfg.ef_search = (efS > 0) ? efS : 128;
        cfg.metric = use_cosine ? DistanceMetric::Cosine : DistanceMetric::L2;
        g_handles[h] = new FloatHNSWWrapper(dim, cfg);
    }

    return Napi::Number::New(env, h);
}

Napi::Value Add(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    Napi::Float32Array vec = info[1].As<Napi::TypedArray>().As<Napi::Float32Array>();
    if (h >= MAX_HANDLES || !g_handles[h])
        return Napi::Number::New(env, INVALID_HANDLE);
    uint32_t id = g_handles[h]->insert(vec.Data());
    return Napi::Number::New(env, id);
}

Napi::Value BulkInsert(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    Napi::Float32Array vecs = info[1].As<Napi::TypedArray>().As<Napi::Float32Array>();
    int n = info[2].As<Napi::Number>().Int32Value();
    if (h >= MAX_HANDLES || !g_handles[h])
        return Napi::Number::New(env, 0);
    int count = g_handles[h]->bulk_insert(vecs.Data(), n);
    return Napi::Number::New(env, count);
}

Napi::Value Query(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    Napi::Float32Array qv = info[1].As<Napi::TypedArray>().As<Napi::Float32Array>();
    int k = info[2].As<Napi::Number>().Int32Value();

    if (h >= MAX_HANDLES || !g_handles[h]) {
        Napi::Object result = Napi::Object::New(env);
        result.Set("count", 0);
        return result;
    }

    auto res = g_handles[h]->search(qv.Data(), k);
    int count = static_cast<int>(res.size());

    Napi::Uint32Array ids = Napi::Uint32Array::New(env, count);
    Napi::Float32Array dists = Napi::Float32Array::New(env, count);
    for (int i = 0; i < count; i++) {
        ids[i] = res[i].first;
        dists[i] = res[i].second;
    }

    Napi::Object result = Napi::Object::New(env);
    result.Set("ids", ids);
    result.Set("distances", dists);
    result.Set("count", count);
    return result;
}

Napi::Value SetEf(const Napi::CallbackInfo& info) {
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    int ef = info[1].As<Napi::Number>().Int32Value();
    if (h < MAX_HANDLES && g_handles[h] && ef > 0)
        g_handles[h]->set_ef_search(ef);
    return info.Env().Undefined();
}

Napi::Value Delete(const Napi::CallbackInfo& info) {
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    uint32_t id = info[1].As<Napi::Number>().Uint32Value();
    if (h < MAX_HANDLES && g_handles[h])
        g_handles[h]->mark_delete(id);
    return info.Env().Undefined();
}

Napi::Value Compact(const Napi::CallbackInfo& info) {
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    if (h < MAX_HANDLES && g_handles[h])
        g_handles[h]->compact();
    return info.Env().Undefined();
}

Napi::Value Count(const Napi::CallbackInfo& info) {
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    if (h >= MAX_HANDLES || !g_handles[h])
        return Napi::Number::New(info.Env(), 0);
    return Napi::Number::New(info.Env(), static_cast<double>(g_handles[h]->count()));
}

Napi::Value GhostCount(const Napi::CallbackInfo& info) {
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    if (h >= MAX_HANDLES || !g_handles[h])
        return Napi::Number::New(info.Env(), 0);
    return Napi::Number::New(info.Env(), static_cast<double>(g_handles[h]->ghost_count()));
}

Napi::Value GhostRatio(const Napi::CallbackInfo& info) {
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    if (h >= MAX_HANDLES || !g_handles[h])
        return Napi::Number::New(info.Env(), 0);
    return Napi::Number::New(info.Env(), g_handles[h]->ghost_ratio());
}

Napi::Value Memory(const Napi::CallbackInfo& info) {
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    if (h >= MAX_HANDLES || !g_handles[h])
        return Napi::Number::New(info.Env(), 0);
    return Napi::Number::New(info.Env(), static_cast<double>(g_handles[h]->memory_bytes()));
}

Napi::Value Dimension(const Napi::CallbackInfo& info) {
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    if (h >= MAX_HANDLES || !g_handles[h])
        return Napi::Number::New(info.Env(), 0);
    return Napi::Number::New(info.Env(), static_cast<double>(g_handles[h]->dimension()));
}

Napi::Value Export(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    if (h >= MAX_HANDLES || !g_handles[h])
        return env.Null();
    auto data = g_handles[h]->serialize();
    auto buf = Napi::Buffer<uint8_t>::Copy(env, data.data(), data.size());
    return buf;
}

Napi::Value Import(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    Napi::Buffer<uint8_t> buf = info[1].As<Napi::Buffer<uint8_t>>();
    if (h >= MAX_HANDLES || !g_handles[h])
        return Napi::Number::New(env, -1);
    bool ok = g_handles[h]->deserialize(buf.Data(), buf.Length());
    return Napi::Number::New(env, ok ? 0 : -1);
}

Napi::Value Dispose(const Napi::CallbackInfo& info) {
    uint32_t h = info[0].As<Napi::Number>().Uint32Value();
    free_handle(h);
    return info.Env().Undefined();
}

// ============================================================================
// Module init
// ============================================================================

Napi::Object InitModule(Napi::Env env, Napi::Object exports) {
    exports.Set("pancake_init",         Napi::Function::New(env, Init));
    exports.Set("pancake_add",          Napi::Function::New(env, Add));
    exports.Set("pancake_bulk_insert",  Napi::Function::New(env, BulkInsert));
    exports.Set("pancake_query",        Napi::Function::New(env, Query));
    exports.Set("pancake_set_ef",       Napi::Function::New(env, SetEf));
    exports.Set("pancake_delete",       Napi::Function::New(env, Delete));
    exports.Set("pancake_compact",      Napi::Function::New(env, Compact));
    exports.Set("pancake_count",        Napi::Function::New(env, Count));
    exports.Set("pancake_ghost_count",  Napi::Function::New(env, GhostCount));
    exports.Set("pancake_ghost_ratio",  Napi::Function::New(env, GhostRatio));
    exports.Set("pancake_memory",       Napi::Function::New(env, Memory));
    exports.Set("pancake_dimension",    Napi::Function::New(env, Dimension));
    exports.Set("pancake_export",       Napi::Function::New(env, Export));
    exports.Set("pancake_import",       Napi::Function::New(env, Import));
    exports.Set("pancake_dispose",      Napi::Function::New(env, Dispose));
    return exports;
}

NODE_API_MODULE(pancake_native, InitModule)
