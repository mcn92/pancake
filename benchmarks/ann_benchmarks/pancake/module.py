import numpy as np

import pancake_py

from ..base.module import BaseANN


class Pancake(BaseANN):
    """ANN-Benchmarks adapter for Pancake's native C++ engine."""

    def __init__(self, metric, quantized, method_param):
        if metric not in ("angular", "euclidean"):
            raise ValueError(f"Pancake does not support ANN-Benchmarks metric {metric!r}")
        self.metric = "cosine" if metric == "angular" else "l2"
        self.quantized = bool(quantized)
        self.method_param = dict(method_param)
        self.index = None
        self.ef_search = None
        self.original_ids = None
        self.zero_ids = np.empty(0, dtype=np.int64)
        self.filtered_base_vectors = 0
        self.retained_zero_vectors = 0
        self.name = self._display_name()

    def _display_name(self):
        storage = "uint8" if self.quantized else "fp32"
        suffix = f", efSearch={self.ef_search}" if self.ef_search is not None else ""
        return (
            f"Pancake {storage} "
            f"(M={self.method_param['M']}, efConstruction={self.method_param['efConstruction']}{suffix})"
        )

    def fit(self, X):
        data = np.asarray(X, dtype=np.float32, order="C")
        if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError(f"Expected a non-empty 2D float matrix, received {data.shape}")
        if self.metric == "cosine":
            finite = np.isfinite(data).all(axis=1)
            if not np.all(finite):
                raise ValueError("Pancake does not support non-finite angular vectors")
            norm_sq = np.einsum("ij,ij->i", data, data)
            if not np.isfinite(norm_sq).all():
                raise ValueError("Pancake does not support angular vectors with non-finite norms")
            keep = norm_sq > 0.0
            self.zero_ids = np.flatnonzero(~keep).astype(np.int64)
            self.retained_zero_vectors = int(self.zero_ids.size)
            self.filtered_base_vectors = 0
            if self.retained_zero_vectors:
                print(
                    f"Pancake adapter: retaining {self.retained_zero_vectors} "
                    "zero-norm angular base vector(s) outside the HNSW graph"
                )
                data = np.ascontiguousarray(data[keep], dtype=np.float32)
                self.original_ids = np.flatnonzero(keep).astype(np.int64)
            else:
                self.original_ids = None
        else:
            self.original_ids = None
            self.zero_ids = np.empty(0, dtype=np.int64)
            self.filtered_base_vectors = 0
            self.retained_zero_vectors = 0
        if data.shape[0] == 0:
            self.index = None
            return
        else:
            self.index = pancake_py.PancakeIndex(
                data.shape[1],
                data.shape[0],
                self.quantized,
                self.metric,
                int(self.method_param["M"]),
                int(self.method_param["efConstruction"]),
            )
            self.index.fit(data)

    def set_query_arguments(self, ef_search):
        self.ef_search = int(ef_search)
        if self.index is not None:
            self.index.set_ef(self.ef_search)
        self.name = self._display_name()

    def query(self, vector, count):
        query = np.asarray(vector, dtype=np.float32, order="C")
        count = int(count)
        if self.metric == "cosine":
            norm_sq = float(np.dot(query, query))
            if not np.isfinite(query).all() or not np.isfinite(norm_sq):
                return np.empty(0, dtype=np.int64)
            if not (norm_sq > 0.0):
                zero = self.zero_ids[:count]
                if zero.size >= count or self.index is None:
                    return zero.copy()
                fill_count = count - int(zero.size)
                fill = self.original_ids[:fill_count]
                return np.concatenate((zero, fill)).astype(np.int64)
        if self.index is None:
            return self.zero_ids[:count].copy()
        result = self.index.query(query, count)
        if self.original_ids is not None:
            result = self.original_ids[np.asarray(result, dtype=np.int64)]
        if self.metric == "cosine" and result.size < count and self.zero_ids.size:
            need = count - int(result.size)
            result = np.concatenate((np.asarray(result, dtype=np.int64), self.zero_ids[:need]))
        return result

    def get_additional(self):
        return {
            "engine_memory_bytes": self.index.memory_bytes() if self.index is not None else 0,
            "storage": "uint8" if self.quantized else "fp32",
            "M": int(self.method_param["M"]),
            "efConstruction": int(self.method_param["efConstruction"]),
            "efSearch": self.ef_search,
            "filtered_base_vectors": self.filtered_base_vectors,
            "retained_zero_vectors": self.retained_zero_vectors,
        }

    def done(self):
        self.index = None

    def __str__(self):
        return self.name
