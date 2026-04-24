/**
 * Pancake C API Implementation
 *
 * Wraps SegmentedDualIndex in a C-compatible interface.
 */

#include "pancake.h"
#include "../core/streaming/segmented_dual_index.hpp"
#include "../core/streaming/streaming_config.hpp"

#include <new>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

using namespace pancake::streaming;

/* ============================================================================
 * Internal Structure
 * ============================================================================ */

struct pancake_index {
    SegmentedDualIndex* impl;
    uint32_t dim;
    size_t search_budget;
};

/* ============================================================================
 * Version
 * ============================================================================ */

const char* pancake_version(void) {
    return "0.1.0";
}

/* ============================================================================
 * Error Handling
 * ============================================================================ */

const char* pancake_error_string(pancake_error_t err) {
    switch (err) {
        case PANCAKE_OK:               return "Success";
        case PANCAKE_ERR_NULL_PARAM:   return "Null parameter";
        case PANCAKE_ERR_INVALID_DIM:  return "Invalid dimension";
        case PANCAKE_ERR_INVALID_CONFIG: return "Invalid configuration";
        case PANCAKE_ERR_OUT_OF_MEMORY: return "Out of memory";
        case PANCAKE_ERR_NOT_FOUND:    return "Not found";
        case PANCAKE_ERR_MAX_CAPACITY: return "Maximum capacity reached";
        case PANCAKE_ERR_INTERNAL:     return "Internal error";
        default:                       return "Unknown error";
    }
}

/* ============================================================================
 * Configuration
 * ============================================================================ */

pancake_config_t pancake_default_config(void) {
    pancake_config_t cfg;
    cfg.segment_capacity = 100000;
    cfg.initial_segments = 8;
    cfg.max_segments = 128;
    cfg.search_budget = 5;
    cfg.hnsw_m = 16;
    cfg.hnsw_ef_construction = 200;
    cfg.hnsw_ef_search = 100;
    cfg.metric = PANCAKE_METRIC_L2;
    cfg.num_threads = 0;
    cfg.compaction_threshold = 0.15f;
    return cfg;
}

/* ============================================================================
 * Lifecycle
 * ============================================================================ */

pancake_error_t pancake_create(
    pancake_index_t** out_index,
    uint32_t dim,
    const pancake_config_t* config
) {
    if (!out_index) return PANCAKE_ERR_NULL_PARAM;
    if (dim == 0) return PANCAKE_ERR_INVALID_DIM;

    *out_index = nullptr;

    // Use default config if not provided
    pancake_config_t cfg = config ? *config : pancake_default_config();

    // Validate config
    if (cfg.segment_capacity == 0 || cfg.initial_segments == 0) {
        return PANCAKE_ERR_INVALID_CONFIG;
    }

    try {
        // Allocate wrapper
        pancake_index* idx = new (std::nothrow) pancake_index;
        if (!idx) return PANCAKE_ERR_OUT_OF_MEMORY;

        // Convert to internal config
        SegmentedConfig internal_cfg;
        internal_cfg.segment_capacity = cfg.segment_capacity;
        internal_cfg.initial_segments = cfg.initial_segments;
        internal_cfg.max_segments = cfg.max_segments;
        internal_cfg.search_budget = cfg.search_budget;
        internal_cfg.hnsw_M = cfg.hnsw_m;
        internal_cfg.hnsw_ef_construction = cfg.hnsw_ef_construction;
        internal_cfg.hnsw_ef_search = cfg.hnsw_ef_search;
        internal_cfg.metric = (cfg.metric == PANCAKE_METRIC_INNER_PRODUCT)
            ? Metric::InnerProduct
            : Metric::L2;
        internal_cfg.search_threads = cfg.num_threads;
        internal_cfg.compaction_threshold = cfg.compaction_threshold;

        // Create implementation
        idx->impl = new (std::nothrow) SegmentedDualIndex(dim, internal_cfg);
        if (!idx->impl) {
            delete idx;
            return PANCAKE_ERR_OUT_OF_MEMORY;
        }

        idx->dim = dim;
        idx->search_budget = cfg.search_budget;

        *out_index = idx;
        return PANCAKE_OK;

    } catch (const std::bad_alloc&) {
        return PANCAKE_ERR_OUT_OF_MEMORY;
    } catch (...) {
        return PANCAKE_ERR_INTERNAL;
    }
}

void pancake_free(pancake_index_t* index) {
    if (index) {
        delete index->impl;
        delete index;
    }
}

/* ============================================================================
 * Core Operations
 * ============================================================================ */

pancake_error_t pancake_add(
    pancake_index_t* index,
    uint64_t id,
    const float* vec
) {
    if (!index || !vec) return PANCAKE_ERR_NULL_PARAM;

    try {
        index->impl->addPoint(id, vec);
        return PANCAKE_OK;
    } catch (const std::runtime_error&) {
        return PANCAKE_ERR_MAX_CAPACITY;
    } catch (...) {
        return PANCAKE_ERR_INTERNAL;
    }
}

pancake_error_t pancake_delete(
    pancake_index_t* index,
    uint64_t id
) {
    if (!index) return PANCAKE_ERR_NULL_PARAM;

    try {
        bool deleted = index->impl->deletePoint(id);
        return deleted ? PANCAKE_OK : PANCAKE_ERR_NOT_FOUND;
    } catch (...) {
        return PANCAKE_ERR_INTERNAL;
    }
}

pancake_error_t pancake_search(
    pancake_index_t* index,
    const float* query,
    size_t k,
    uint64_t* out_ids,
    float* out_dists,
    size_t* out_count
) {
    if (!index || !query || !out_ids || !out_count) {
        return PANCAKE_ERR_NULL_PARAM;
    }

    try {
        auto results = index->impl->search(query, k);

        *out_count = results.size();

        for (size_t i = 0; i < results.size(); ++i) {
            out_ids[i] = results[i].first;
            if (out_dists) {
                out_dists[i] = results[i].second;
            }
        }

        return PANCAKE_OK;

    } catch (...) {
        return PANCAKE_ERR_INTERNAL;
    }
}

/* ============================================================================
 * Batch Operations
 * ============================================================================ */

pancake_error_t pancake_add_batch(
    pancake_index_t* index,
    const uint64_t* ids,
    const float* vecs,
    size_t n
) {
    if (!index || !ids || !vecs) return PANCAKE_ERR_NULL_PARAM;

    try {
        for (size_t i = 0; i < n; ++i) {
            index->impl->addPoint(ids[i], vecs + i * index->dim);
        }
        return PANCAKE_OK;
    } catch (const std::runtime_error&) {
        return PANCAKE_ERR_MAX_CAPACITY;
    } catch (...) {
        return PANCAKE_ERR_INTERNAL;
    }
}

pancake_error_t pancake_search_batch(
    pancake_index_t* index,
    const float* queries,
    size_t n,
    size_t k,
    uint64_t* out_ids,
    float* out_dists,
    size_t* out_counts
) {
    if (!index || !queries || !out_ids || !out_counts) {
        return PANCAKE_ERR_NULL_PARAM;
    }

    try {
        for (size_t q = 0; q < n; ++q) {
            const float* query = queries + q * index->dim;
            uint64_t* ids = out_ids + q * k;
            float* dists = out_dists ? (out_dists + q * k) : nullptr;

            auto results = index->impl->search(query, k);

            out_counts[q] = results.size();

            for (size_t i = 0; i < results.size(); ++i) {
                ids[i] = results[i].first;
                if (dists) {
                    dists[i] = results[i].second;
                }
            }

            // Zero-fill unused slots
            for (size_t i = results.size(); i < k; ++i) {
                ids[i] = 0;
                if (dists) dists[i] = 0.0f;
            }
        }

        return PANCAKE_OK;

    } catch (...) {
        return PANCAKE_ERR_INTERNAL;
    }
}

/* ============================================================================
 * Maintenance
 * ============================================================================ */

int pancake_needs_compaction(pancake_index_t* index) {
    if (!index) return 0;
    return index->impl->needs_compaction() ? 1 : 0;
}

pancake_error_t pancake_compact(pancake_index_t* index) {
    if (!index) return PANCAKE_ERR_NULL_PARAM;

    try {
        index->impl->compact();
        return PANCAKE_OK;
    } catch (...) {
        return PANCAKE_ERR_INTERNAL;
    }
}

/* ============================================================================
 * Statistics
 * ============================================================================ */

static size_t get_memory_bytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(),
                             (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
    return 0;
#else
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss * 1024;
#endif
}

pancake_error_t pancake_stats(
    pancake_index_t* index,
    pancake_stats_t* out_stats
) {
    if (!index || !out_stats) return PANCAKE_ERR_NULL_PARAM;

    try {
        out_stats->live_count = index->impl->live_count();
        out_stats->ghost_count = index->impl->ghost_count();
        out_stats->segment_count = index->impl->segment_count();
        out_stats->ghost_ratio = index->impl->ghost_ratio();
        out_stats->dim = index->dim;
        out_stats->memory_bytes = get_memory_bytes();

        return PANCAKE_OK;

    } catch (...) {
        return PANCAKE_ERR_INTERNAL;
    }
}

/* ============================================================================
 * Advanced / Tuning
 * ============================================================================ */

pancake_error_t pancake_set_ef_search(
    pancake_index_t* index,
    size_t ef
) {
    if (!index) return PANCAKE_ERR_NULL_PARAM;
    // Would need to add this to SegmentedDualIndex
    // For now, return OK as no-op
    (void)ef;
    return PANCAKE_OK;
}

pancake_error_t pancake_set_search_budget(
    pancake_index_t* index,
    size_t budget
) {
    if (!index) return PANCAKE_ERR_NULL_PARAM;
    // Would need to add this to SegmentedDualIndex
    index->search_budget = budget;
    return PANCAKE_OK;
}

pancake_error_t pancake_set_compaction_threshold(
    pancake_index_t* index,
    float threshold
) {
    if (!index) return PANCAKE_ERR_NULL_PARAM;

    try {
        index->impl->set_compaction_threshold(threshold);
        return PANCAKE_OK;
    } catch (...) {
        return PANCAKE_ERR_INTERNAL;
    }
}

/* ============================================================================
 * Serialization (stubs - would need HNSW serialization support)
 * ============================================================================ */

pancake_error_t pancake_serialized_size(
    pancake_index_t* index,
    size_t* out_size
) {
    if (!index || !out_size) return PANCAKE_ERR_NULL_PARAM;
    // TODO: Implement when serialization is added
    *out_size = 0;
    return PANCAKE_ERR_INTERNAL;
}

pancake_error_t pancake_serialize(
    pancake_index_t* index,
    void* buffer,
    size_t size
) {
    (void)index;
    (void)buffer;
    (void)size;
    // TODO: Implement when serialization is added
    return PANCAKE_ERR_INTERNAL;
}

pancake_error_t pancake_deserialize(
    pancake_index_t** out_index,
    const void* buffer,
    size_t size
) {
    (void)out_index;
    (void)buffer;
    (void)size;
    // TODO: Implement when serialization is added
    return PANCAKE_ERR_INTERNAL;
}
