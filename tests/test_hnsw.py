"""HNSW correctness: recall against brute force, and layer-distribution sanity."""

import numpy as np
import pytest

from vectordb.config import HNSWConfig
from vectordb.distance import Metric
from vectordb.flat import FlatIndex
from vectordb.hnsw import HNSW


def _recall(index, flat, queries, k, ef):
    hits = 0
    for q in queries:
        truth = {n for n, _ in flat.search(q, k)}
        got = {n for n, _ in index.search(q, k, ef_search=ef)}
        hits += len(truth & got)
    return hits / (len(queries) * k)


@pytest.mark.parametrize("metric", [Metric.L2, Metric.COSINE, Metric.IP])
def test_recall_above_threshold(dataset, metric):
    data, queries = dataset
    dim = data.shape[1]
    flat = FlatIndex(dim, metric)
    flat.add_batch(data, list(range(len(data))))

    index = HNSW(HNSWConfig(dim=dim, metric=metric, M=16, ef_construction=200, seed=7))
    for row in data:
        index.add(row)

    recall = _recall(index, flat, queries, k=10, ef=100)
    assert recall >= 0.90, f"{metric} recall too low: {recall}"


def test_recall_increases_with_ef(dataset):
    data, queries = dataset
    dim = data.shape[1]
    flat = FlatIndex(dim, Metric.L2)
    flat.add_batch(data, list(range(len(data))))
    index = HNSW(HNSWConfig(dim=dim, metric=Metric.L2, M=16, ef_construction=200, seed=7))
    for row in data:
        index.add(row)

    low = _recall(index, flat, queries, k=10, ef=10)
    high = _recall(index, flat, queries, k=10, ef=200)
    assert high >= low  # larger ef never hurts recall


def test_layer_distribution_is_geometric(dataset):
    """Layer occupancy should decay geometrically by roughly 1/M per level."""
    data, _ = dataset
    dim = data.shape[1]
    M = 16
    index = HNSW(HNSWConfig(dim=dim, metric=Metric.L2, M=M, seed=7))
    for row in data:
        index.add(row)

    # level_histogram buckets nodes by their TOP level. Every node lives on
    # layer 0, but only ~1/M of them have their top level above 0.
    hist = index.level_histogram()
    assert 0 in hist
    assert sum(hist.values()) == len(data)  # every node counted once
    assert hist[0] == max(hist.values())  # layer-0-only nodes are the majority
    # Higher top levels must be rarer (monotonic decay).
    levels = sorted(hist)
    for lo, hi in zip(levels, levels[1:]):
        assert hist[hi] <= hist[lo]
    # Fraction with top level >= 1 should be near 1/M (paper: p = 1 - exp(-1/mL)).
    frac_upper = sum(v for lv, v in hist.items() if lv >= 1) / len(data)
    expected = 1.0 / M
    assert abs(frac_upper - expected) < 0.04


def test_empty_and_single(dataset):
    dim = dataset[0].shape[1]
    index = HNSW(HNSWConfig(dim=dim))
    assert index.search(np.zeros(dim, dtype=np.float32), 5) == []
    index.add(np.ones(dim, dtype=np.float32))
    res = index.search(np.ones(dim, dtype=np.float32), 5)
    assert len(res) == 1 and res[0][0] == 0


def test_search_returns_sorted(dataset):
    data, queries = dataset
    dim = data.shape[1]
    index = HNSW(HNSWConfig(dim=dim, metric=Metric.L2, seed=3))
    for row in data:
        index.add(row)
    res = index.search(queries[0], 10, ef_search=100)
    dists = [d for _, d in res]
    assert dists == sorted(dists)
