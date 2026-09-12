"""Recall@10 vs latency/QPS benchmark.

Compares three things on the same data at matched parameters:

  1. This project's from-scratch HNSW, swept over several efSearch values.
  2. A brute-force exact index (our FlatIndex) for ground truth and as the
     exact-search latency baseline.
  3. FAISS IndexHNSWFlat (the C reference HNSW), swept over the same efSearch
     values, if faiss is installed. FAISS is used ONLY here, for comparison.

Vectors are L2-normalized up front, so L2 distance ranks identically to cosine
similarity; that keeps our engine and FAISS on the exact same metric.

Raw numbers are written to benchmarks/results/results.json.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from vectordb.config import HNSWConfig
from vectordb.distance import Metric
from vectordb.flat import FlatIndex
from vectordb.hnsw import HNSW
from vectordb.storage import save_hnsw

HERE = os.path.dirname(__file__)


def load_data(path: str, n_queries: int):
    npz = np.load(path, allow_pickle=True)
    emb = npz["embeddings"].astype(np.float32)
    # L2-normalize so L2 == cosine ordering.
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    rng = np.random.default_rng(0)
    idx = rng.permutation(emb.shape[0])
    q_idx = idx[:n_queries]
    b_idx = idx[n_queries:]
    return emb[b_idx], emb[q_idx]


def ground_truth(base: np.ndarray, queries: np.ndarray, k: int) -> list[set[int]]:
    flat = FlatIndex(base.shape[1], Metric.L2)
    flat.add_batch(base, list(range(base.shape[0])))
    truth = []
    for q in queries:
        truth.append({n for n, _ in flat.search(q, k)})
    return truth


def measure(search_fn, queries: np.ndarray, truth: list[set[int]], k: int) -> dict:
    latencies = []
    recall_hits = 0
    for q, gt in zip(queries, truth):
        t = time.perf_counter()
        got = search_fn(q)
        latencies.append((time.perf_counter() - t) * 1000.0)
        recall_hits += len(gt & set(got))
    lat = np.asarray(latencies)
    return {
        "recall@%d" % k: recall_hits / (len(queries) * k),
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "qps": float(1000.0 / lat.mean()),
    }


def bench_ours(base, queries, truth, k, M, efc, ef_values):
    cfg = HNSWConfig(dim=base.shape[1], metric=Metric.L2, M=M, ef_construction=efc, seed=1)
    index = HNSW(cfg)
    t = time.time()
    for row in base:
        index.add(row)
    build_s = time.time() - t

    tmp = os.path.join(HERE, "results", "_tmp_graph.bin")
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    save_hnsw(index, tmp)
    size_mb = os.path.getsize(tmp) / 1e6
    os.remove(tmp)

    rows = []
    for ef in ef_values:
        stats = measure(
            lambda q, ef=ef: [n for n, _ in index.search(q, k, ef_search=ef)],
            queries, truth, k,
        )
        stats.update({"ef_search": ef})
        rows.append(stats)
    return {"build_s": build_s, "index_mb": size_mb, "sweep": rows}


def bench_faiss(base, queries, truth, k, M, efc, ef_values):
    try:
        import faiss
    except ImportError:
        return None
    dim = base.shape[1]
    index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_L2)
    index.hnsw.efConstruction = efc
    t = time.time()
    index.add(base)
    build_s = time.time() - t

    # exact baseline latency via IndexFlatL2
    flat = faiss.IndexFlatL2(dim)
    flat.add(base)

    rows = []
    for ef in ef_values:
        index.hnsw.efSearch = ef

        def search_fn(q, index=index):
            _, ids = index.search(q.reshape(1, -1), k)
            return ids[0].tolist()

        stats = measure(search_fn, queries, truth, k)
        stats.update({"ef_search": ef})
        rows.append(stats)

    def flat_search(q):
        _, ids = flat.search(q.reshape(1, -1), k)
        return ids[0].tolist()

    flat_stats = measure(flat_search, queries, truth, k)
    return {"build_s": build_s, "sweep": rows, "flat_exact": flat_stats}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "data", "embeddings.npz"))
    ap.add_argument("--queries", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--M", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--ef-values", default="10,20,40,80,160,320")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(
            f"No embeddings at {args.data}. Run generate_embeddings.py first "
            "(see benchmarks/generate_embeddings.py)."
        )
    ef_values = [int(x) for x in args.ef_values.split(",")]

    base, queries = load_data(args.data, args.queries)
    print(f"base={base.shape[0]} queries={queries.shape[0]} dim={base.shape[1]}")

    print("computing exact ground truth (brute force) ...")
    t = time.time()
    truth = ground_truth(base, queries, args.k)
    gt_s = time.time() - t

    # brute-force latency for our FlatIndex
    flat = FlatIndex(base.shape[1], Metric.L2)
    flat.add_batch(base, list(range(base.shape[0])))
    flat_stats = measure(
        lambda q: [n for n, _ in flat.search(q, args.k)], queries, truth, args.k
    )

    print(f"building our HNSW (M={args.M}, efc={args.ef_construction}) ...")
    ours = bench_ours(base, queries, truth, args.k, args.M, args.ef_construction, ef_values)

    print("benchmarking FAISS (if available) ...")
    faiss_res = bench_faiss(base, queries, truth, args.k, args.M, args.ef_construction, ef_values)

    result = {
        "dataset": {
            "base": int(base.shape[0]),
            "queries": int(queries.shape[0]),
            "dim": int(base.shape[1]),
            "k": args.k,
            "M": args.M,
            "ef_construction": args.ef_construction,
        },
        "ground_truth_build_s": gt_s,
        "brute_force": flat_stats,
        "ours": ours,
        "faiss": faiss_res,
    }
    out = os.path.join(HERE, "results", "results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)

    _print_table(result)
    print(f"\nraw results -> {out}")


def _print_table(r: dict) -> None:
    d = r["dataset"]
    print("\n=== Benchmark ===")
    print(f"base={d['base']} dim={d['dim']} k={d['k']} M={d['M']} efc={d['ef_construction']}")
    bf = r["brute_force"]
    print(f"\nBrute force (exact, ours): recall=1.000 p50={bf['p50_ms']:.2f}ms "
          f"p95={bf['p95_ms']:.2f}ms qps={bf['qps']:.0f}")
    print(f"\nOurs HNSW  build={r['ours']['build_s']:.1f}s  index={r['ours']['index_mb']:.1f}MB")
    print(f"{'efSearch':>8} {'recall@10':>10} {'p50_ms':>8} {'p95_ms':>8} {'qps':>8}")
    for row in r["ours"]["sweep"]:
        print(f"{row['ef_search']:>8} {row['recall@10']:>10.4f} {row['p50_ms']:>8.3f} "
              f"{row['p95_ms']:>8.3f} {row['qps']:>8.0f}")
    if r["faiss"]:
        print(f"\nFAISS HNSW build={r['faiss']['build_s']:.2f}s")
        print(f"{'efSearch':>8} {'recall@10':>10} {'p50_ms':>8} {'p95_ms':>8} {'qps':>8}")
        for row in r["faiss"]["sweep"]:
            print(f"{row['ef_search']:>8} {row['recall@10']:>10.4f} {row['p50_ms']:>8.3f} "
                  f"{row['p95_ms']:>8.3f} {row['qps']:>8.0f}")
        fe = r["faiss"]["flat_exact"]
        print(f"FAISS flat exact: p50={fe['p50_ms']:.3f}ms qps={fe['qps']:.0f}")


if __name__ == "__main__":
    main()
