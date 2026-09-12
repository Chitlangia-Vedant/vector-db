"""Flat (brute-force) index. Exact nearest neighbors, O(n) per query.

Used as the ground truth for recall measurement: it always returns the true
k nearest neighbors, so HNSW's recall is (overlap with this) / k.
"""

from __future__ import annotations

import numpy as np

from .distance import Metric, distance_batch, prepare


class FlatIndex:
    def __init__(self, dim: int, metric: Metric = Metric.L2):
        self.dim = dim
        self.metric = metric if isinstance(metric, Metric) else Metric(metric)
        self._matrix = np.empty((0, dim), dtype=np.float32)
        self._ids: list[int] = []

    def add(self, vector: np.ndarray, node_id: int) -> None:
        vec = prepare(np.asarray(vector, dtype=np.float32).reshape(1, -1), self.metric)
        self._matrix = np.vstack([self._matrix, vec]) if self._matrix.size else vec
        self._ids.append(node_id)

    def add_batch(self, vectors: np.ndarray, ids: list[int]) -> None:
        vecs = prepare(np.asarray(vectors, dtype=np.float32), self.metric)
        self._matrix = np.vstack([self._matrix, vecs]) if self._matrix.size else vecs
        self._ids.extend(ids)

    def search(
        self, query: np.ndarray, k: int, allowed: set[int] | None = None
    ) -> list[tuple[int, float]]:
        if self._matrix.shape[0] == 0:
            return []
        q = prepare(np.asarray(query, dtype=np.float32).reshape(-1), self.metric)
        dists = distance_batch(q, self._matrix, self.metric)
        order = np.argsort(dists)
        out: list[tuple[int, float]] = []
        for idx in order:
            node_id = self._ids[idx]
            if allowed is not None and node_id not in allowed:
                continue
            out.append((node_id, float(dists[idx])))
            if len(out) >= k:
                break
        return out

    @property
    def size(self) -> int:
        return self._matrix.shape[0]
