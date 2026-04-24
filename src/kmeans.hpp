#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

// iostream included for optional verbose output (gated by verbose flag).
// Dead code elimination removes unused code when verbose=false (default).
#include <iostream>

#include "math_utils.hpp"

namespace pancake {
namespace clustering {

enum class DistanceMetric {
    L2,      // Euclidean distance (for L2/dot product datasets)
    COSINE   // Cosine distance (for angular/cosine datasets)
};

/**
 * K-Means clustering implementation with k-means++ initialization
 *
 * This implementation is optimized for speed and determinism:
 * - K-means++ initialization for better convergence (WHY: avoids pathological
 *   clustered initial centroids that trap Lloyd's algorithm in local minima)
 * - Lloyd's algorithm for iteration (standard expectation-maximization)
 * - Early stopping when assignments don't change (convergence detection)
 * - Fixed random seed for reproducibility (critical for testing/debugging)
 * - Supports both Euclidean and spherical k-means (spherical for cosine metric)
 */
class KMeans {
public:
    /**
     * Perform k-means clustering on vectors
     *
     * @param vectors Pointer to NxD matrix (row-major, float32)
     * @param n Number of vectors
     * @param dim Dimensionality
     * @param k Number of clusters
     * @param max_iters Maximum iterations (default: 100)
     * @param seed Random seed for initialization (default: 42)
     * @param verbose Print progress (default: false)
     * @param metric Distance metric (default: L2)
     * @return Vector of cluster assignments (length n, values 0..k-1)
     */
    static std::vector<int> fit(
        const float* vectors,
        size_t n,
        uint32_t dim,
        size_t k,
        size_t max_iters = 10,  // 10 iterations sufficient for segment routing quality
        uint32_t seed = 42,
        bool verbose = false,
        DistanceMetric metric = DistanceMetric::L2
    ) {
#ifndef PANCAKE_WASM_BUILD
        if (n == 0 || dim == 0 || k == 0) {
            throw std::invalid_argument("kmeans: n, dim, k must be > 0");
        }
        if (k > n) {
            throw std::invalid_argument("kmeans: k > n (more clusters than points)");
        }
#else
        // In WASM build, return empty assignments on error
        if (n == 0 || dim == 0 || k == 0) {
            return std::vector<int>();
        }
        if (k > n) {
            return std::vector<int>();
        }
#endif

        if (verbose && metric == DistanceMetric::COSINE) {
            std::cout << "  Using spherical k-means (cosine distance)" << std::endl;
        }

        // For spherical k-means, normalize all input vectors to unit sphere
        std::vector<float> normalized_vectors;
        const float* vectors_to_use = vectors;

        if (metric == DistanceMetric::COSINE) {
            normalized_vectors.resize(n * dim);
            for (size_t i = 0; i < n; ++i) {
                // Copy vector
                for (uint32_t d = 0; d < dim; ++d) {
                    normalized_vectors[i * dim + d] = vectors[i * dim + d];
                }
                // Normalize in-place
                wasm::math::normalize_vector(&normalized_vectors[i * dim], dim);
            }
            vectors_to_use = normalized_vectors.data();

            if (verbose) {
                std::cout << "  Normalized " << n << " vectors to unit sphere" << std::endl;
            }
        }

        // Initialize centroids using k-means++
        std::vector<std::vector<float>> centroids = initialize_centroids(
            vectors_to_use, n, dim, k, seed, verbose, metric
        );

        // Initialize assignments
        std::vector<int> assignments(n, -1);
        std::vector<int> prev_assignments(n, -1);

        // Lloyd's algorithm
        for (size_t iter = 0; iter < max_iters; ++iter) {
            // Assignment step: assign each point to nearest centroid
            bool changed = false;
            for (size_t i = 0; i < n; ++i) {
                int best_cluster = find_nearest_centroid(
                    vectors_to_use + i * dim, centroids, dim, metric
                );
                if (assignments[i] != best_cluster) {
                    changed = true;
                    assignments[i] = best_cluster;
                }
            }

            if (verbose) {
                std::cout << "  K-means iteration " << (iter + 1)
                          << "/" << max_iters;
                if (!changed) {
                    std::cout << " (converged)";
                }
                std::cout << std::endl;
            }

            // Early stopping if no assignments changed
            if (!changed) {
                if (verbose) {
                    std::cout << "  Converged after " << (iter + 1)
                              << " iterations" << std::endl;
                }
                break;
            }

            // Update step: recompute centroids
            update_centroids(vectors_to_use, n, dim, assignments, centroids, metric);
        }

        // Validate: ensure no empty clusters
        std::vector<size_t> cluster_sizes(k, 0);
        for (int c : assignments) {
            cluster_sizes[c]++;
        }

        if (verbose) {
            std::cout << "  Cluster sizes: ";
            for (size_t i = 0; i < k; ++i) {
                std::cout << cluster_sizes[i];
                if (i + 1 < k) std::cout << ", ";
            }
            std::cout << std::endl;
        }

        // Handle empty clusters by splitting the largest cluster
        for (size_t empty_idx = 0; empty_idx < k; ++empty_idx) {
            if (cluster_sizes[empty_idx] == 0) {
                // Find largest cluster
                size_t largest_idx = 0;
                for (size_t j = 1; j < k; ++j) {
                    if (cluster_sizes[j] > cluster_sizes[largest_idx]) {
                        largest_idx = j;
                    }
                }

                if (cluster_sizes[largest_idx] < 2) {
                    // Can't split further - accept degenerate clustering
                    if (verbose) {
                        std::cout << "  WARNING: Empty cluster " << empty_idx
                                  << " but no splittable clusters remain" << std::endl;
                    }
                    continue;
                }

                // Find point in largest cluster that's farthest from its centroid
                uint32_t farthest_point = 0;
                float max_dist = -1.0f;

                for (uint32_t i = 0; i < n; ++i) {
                    if (assignments[i] != static_cast<int>(largest_idx)) continue;

                    float dist = 0.0f;
                    for (uint32_t d = 0; d < dim; ++d) {
                        float diff = vectors_to_use[i * dim + d] - centroids[largest_idx][d];
                        dist += diff * diff;
                    }

                    if (dist > max_dist) {
                        max_dist = dist;
                        farthest_point = i;
                    }
                }

                // Move farthest point to empty cluster as new centroid
                for (uint32_t d = 0; d < dim; ++d) {
                    centroids[empty_idx][d] = vectors_to_use[farthest_point * dim + d];
                }
                assignments[farthest_point] = static_cast<int>(empty_idx);

                // Update counts
                cluster_sizes[largest_idx]--;
                cluster_sizes[empty_idx] = 1;

                if (verbose) {
                    std::cout << "  Recovered empty cluster " << empty_idx
                              << " by splitting cluster " << largest_idx << std::endl;
                }
            }
        }

        return assignments;
    }

private:
    /**
     * Initialize centroids using k-means++ algorithm
     * This provides better initialization than random selection
     */
    static std::vector<std::vector<float>> initialize_centroids(
        const float* vectors,
        size_t n,
        uint32_t dim,
        size_t k,
        uint32_t seed,
        bool verbose,
        DistanceMetric metric
    ) {
        std::mt19937 rng(seed);
        std::vector<std::vector<float>> centroids;
        centroids.reserve(k);

        // Choose first centroid uniformly at random
        std::uniform_int_distribution<size_t> uniform_dist(0, n - 1);
        size_t first_idx = uniform_dist(rng);
        centroids.push_back(copy_vector(vectors + first_idx * dim, dim));

        if (verbose) {
            std::cout << "  Initializing centroids with k-means++..." << std::endl;
        }

        // Choose remaining centroids with probability proportional to D(x)^2
        std::vector<float> min_distances(n, std::numeric_limits<float>::max());

        for (size_t c = 1; c < k; ++c) {
            // Update minimum distances to nearest centroid
            const auto& latest_centroid = centroids.back();
            for (size_t i = 0; i < n; ++i) {
                float dist = compute_distance(
                    vectors + i * dim, latest_centroid.data(), dim, metric
                );
                min_distances[i] = std::min(min_distances[i], dist);
            }

            // Compute cumulative distribution for weighted sampling
            std::vector<float> cumulative(n);
            cumulative[0] = min_distances[0];
            for (size_t i = 1; i < n; ++i) {
                cumulative[i] = cumulative[i - 1] + min_distances[i];
            }

            // Sample proportional to squared distance
            std::uniform_real_distribution<float> real_dist(0.0f, cumulative.back());
            float threshold = real_dist(rng);

            // Binary search for threshold in cumulative array
            auto it = std::lower_bound(cumulative.begin(), cumulative.end(), threshold);
            size_t idx = std::distance(cumulative.begin(), it);
            if (idx >= n) idx = n - 1;

            centroids.push_back(copy_vector(vectors + idx * dim, dim));
        }

        return centroids;
    }

    /**
     * Find nearest centroid to a given vector
     */
    static int find_nearest_centroid(
        const float* vec,
        const std::vector<std::vector<float>>& centroids,
        uint32_t dim,
        DistanceMetric metric
    ) {
        int best_cluster = 0;
        float best_dist = compute_distance(vec, centroids[0].data(), dim, metric);

        for (size_t c = 1; c < centroids.size(); ++c) {
            float dist = compute_distance(vec, centroids[c].data(), dim, metric);
            if (dist < best_dist) {
                best_dist = dist;
                best_cluster = static_cast<int>(c);
            }
        }

        return best_cluster;
    }

    /**
     * Update centroids based on current assignments
     */
    static void update_centroids(
        const float* vectors,
        size_t n,
        uint32_t dim,
        const std::vector<int>& assignments,
        std::vector<std::vector<float>>& centroids,
        DistanceMetric metric
    ) {
        size_t k = centroids.size();

        // Reset centroids to zero
        for (auto& centroid : centroids) {
            std::fill(centroid.begin(), centroid.end(), 0.0f);
        }

        // Accumulate sums
        std::vector<size_t> counts(k, 0);
        for (size_t i = 0; i < n; ++i) {
            int cluster = assignments[i];
            counts[cluster]++;

            for (uint32_t d = 0; d < dim; ++d) {
                centroids[cluster][d] += vectors[i * dim + d];
            }
        }

        // Divide by counts to get means
        for (size_t c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                float inv_count = 1.0f / counts[c];
                for (uint32_t d = 0; d < dim; ++d) {
                    centroids[c][d] *= inv_count;
                }

                // For spherical k-means, normalize centroids to unit sphere
                if (metric == DistanceMetric::COSINE) {
                    wasm::math::normalize_vector(centroids[c].data(), dim);
                }
            }
        }
    }

    /**
     * Compute distance based on metric
     */
    static float compute_distance(
        const float* a,
        const float* b,
        uint32_t dim,
        DistanceMetric metric
    ) {
        if (metric == DistanceMetric::COSINE) {
            return cosine_distance(a, b, dim);
        } else {
            return l2_distance_squared(a, b, dim);
        }
    }

    /**
     * Compute squared L2 distance between two vectors
     */
    static float l2_distance_squared(const float* a, const float* b, uint32_t dim) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return sum;
    }

    /**
     * Compute cosine distance between two vectors (1 - dot product)
     * Assumes vectors are normalized to unit length
     */
    static float cosine_distance(const float* a, const float* b, uint32_t dim) {
        float dot = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) {
            dot += a[i] * b[i];
        }
        return 1.0f - dot;
    }

    /**
     * Copy a vector to a new std::vector
     */
    static std::vector<float> copy_vector(const float* vec, uint32_t dim) {
        return std::vector<float>(vec, vec + dim);
    }
};

} // namespace clustering
} // namespace pancake
