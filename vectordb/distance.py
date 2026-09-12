"""Distance metrics for the vector index.

Every metric is expressed as a *distance* where a smaller value means "more
similar". The index only ever compares distances, so as long as the ordering is
correct the absolute value does not matter.

  - L2      : squared Euclidean distance. Squared (not the square root) because
              the square root is monotonic and we only need ordering; skipping
              it saves a sqrt per comparison.
  - COSINE  : implemented as inner product on L2-normalized vectors, returned as
              (1 - dot) so identical vectors give 0. Vectors are normalized once
              on insert (see Collection / HNSW), so at query time cosine reduces
              to the same code path as inner product.
  - IP      : negative inner product. Larger dot product means more similar, so
              we negate to keep "smaller is closer".
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class Metric(str, Enum):
    L2 = "l2"
    COSINE = "cosine"
    IP = "ip"


def normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize each row. Zero vectors are left as-is (norm clamped to 1)."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        norm = np.linalg.norm(vectors)
        if norm == 0.0:
            return vectors
        return vectors / norm
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return vectors / norms


def prepare(vectors: np.ndarray, metric: Metric) -> np.ndarray:
    """Transform stored vectors so the query-time distance is a plain formula.

    For cosine we store normalized vectors; then cosine distance is 1 - dot,
    which uses the same dot-product machinery as inner product.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if metric is Metric.COSINE:
        return normalize(vectors)
    return vectors


def distance_batch(query: np.ndarray, matrix: np.ndarray, metric: Metric) -> np.ndarray:
    """Distance from one query vector to every row of ``matrix``.

    ``query`` must already be prepared with the same metric (normalized for
    cosine). Returns a 1-D float32 array, one distance per row.
    """
    if matrix.shape[0] == 0:
        return np.empty(0, dtype=np.float32)
    if metric is Metric.L2:
        diff = matrix - query
        return np.einsum("ij,ij->i", diff, diff).astype(np.float32)
    if metric is Metric.COSINE:
        return (1.0 - matrix @ query).astype(np.float32)
    # inner product
    return (-(matrix @ query)).astype(np.float32)
