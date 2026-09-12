# Benchmarks

Honest recall and latency measurements for the from-scratch HNSW implementation. The goal is to measure the speed/accuracy tradeoff against exact brute-force search and compare the implementation with FAISS as an external reference.

## Setup

* **Data**: 50,500 real AG News texts embedded with `sentence-transformers/all-mpnet-base-v2`.
* **Embeddings**: 768-dimensional, L2-normalized vectors.
* **Indexed vectors**: 50,000.
* **Queries**: 500 held-out vectors.
* **Ground truth**: The exact brute-force index implemented in `vectordb/flat.py`. Recall@10 is the fraction of the true 10 nearest neighbours returned by HNSW, averaged over all 500 queries.
* **Parameters**: `M = 16`, `efConstruction = 200`, `k = 10`.
* **efSearch**: 10, 20, 40, 80, 160, 320.
* **Comparison**: FAISS `IndexHNSWFlat` with the same `M` and `efConstruction`, plus FAISS `IndexFlatL2` as an exact reference.
* **FAISS role**: FAISS is used only for external benchmarking. It is not used by the vector database implementation.

The embeddings were generated once and saved to:

```text
benchmarks/data/embeddings.npz
```

The benchmark reuses this file, so running the benchmark does not regenerate the embeddings.

Reproduce the benchmark with:

```bash
python benchmarks/run_benchmark.py --ef-values 10,20,40,80,160,320
```

Raw measurements are saved to:

```text
benchmarks/results/results.json
```

## Build Time and Index Size

The from-scratch implementation builds the graph in Python. Its insertion process performs graph traversal and neighbour selection for every vector, which makes construction much slower than FAISS's optimized native implementation.

The measured index size for our HNSW includes the stored float32 vectors and graph data.

| Engine | Build time | Index size |
| :--- | ---: | ---: |
| Ours HNSW | 3693.7 s | 160.5 MB |
| FAISS HNSW | 29.48 s | n/a |

The build-time difference is expected because the custom implementation prioritizes transparency of the algorithm over native-code performance.

## Exact Search Baseline

The exact baseline scans every indexed vector for every query. It is used as the ground truth for recall.

Because `FlatIndex` performs exhaustive search, its nearest-neighbour results define the reference set used to calculate recall.

| Engine | Recall@10 | p50 ms | p95 ms | QPS |
| :--- | ---: | ---: | ---: | ---: |
| Ours FlatIndex | 1.0000 | 118.42 | 133.37 | 8 |
| FAISS Flat | 1.0000 | 42.707 | n/a | 23 |

## Approximate Search Performance

### Our HNSW

| efSearch | Recall@10 | p50 ms | p95 ms | QPS |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.9576 | 2.264 | 3.639 | 419 |
| 20 | 0.9854 | 2.378 | 3.702 | 395 |
| 40 | 0.9942 | 3.990 | 5.918 | 239 |
| 80 | 0.9968 | 7.518 | 22.521 | 111 |
| 160 | 0.9984 | 12.713 | 18.502 | 76 |
| 320 | 0.9994 | 22.959 | 37.887 | 40 |

### FAISS HNSW Reference

| efSearch | Recall@10 | p50 ms | p95 ms | QPS |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 0.9368 | 0.130 | 0.195 | 6499 |
| 20 | 0.9770 | 0.186 | 0.270 | 5190 |
| 40 | 0.9922 | 0.322 | 0.465 | 3008 |
| 80 | 0.9968 | 0.550 | 0.773 | 1775 |
| 160 | 0.9982 | 1.003 | 1.447 | 982 |
| 320 | 0.9998 | 1.796 | 2.550 | 544 |

## Recall vs Speed

The benchmark demonstrates the speed/accuracy tradeoff controlled by `efSearch`.

Increasing `efSearch` expands the candidate search performed by HNSW. This generally improves recall while increasing query latency and reducing throughput.

The measured results therefore provide a tunable speed/accuracy curve instead of a single ANN accuracy number.

![HNSW recall vs QPS](benchmarks/results/recall_vs_qps.png)

![HNSW recall vs efSearch](benchmarks/results/recall_vs_efsearch.png)

## Interpretation

### Correctness

The from-scratch HNSW reaches 0.9994 recall@10 at `efSearch=320`.

Recall is measured against the exact results produced by the project's `FlatIndex`. A recall of 0.9994 means that, averaged across the 500 queries, 99.94% of the true top-10 neighbours were recovered.

At matched settings, our implementation has higher recall than FAISS at `efSearch` values 10 through 160 in this benchmark. At `efSearch=320`, FAISS reaches 0.9998 recall compared with 0.9994 for our implementation.

The important result is that increasing `efSearch` moves the custom HNSW closer to the exact-search result.

### Recall and Latency Tradeoff

The `efSearch` parameter acts as the main accuracy/speed knob.

* At `efSearch=40`, the implementation reaches 0.9942 recall@10 at 239 QPS.
* At `efSearch=80`, recall increases to 0.9968 while throughput decreases to 111 QPS.
* At `efSearch=320`, recall reaches 0.9994 while throughput decreases to 40 QPS.

This demonstrates the expected ANN tradeoff: more graph exploration produces higher recall at the cost of additional query work.

### ANN vs Exact Search

Our exact `FlatIndex` processes every stored vector for every query and achieves 1.0000 recall at approximately 8 QPS.

At `efSearch=40`, our HNSW reaches 0.9942 recall at approximately 239 QPS.

That is about 30x the measured query throughput of our exact implementation, while recovering 99.42% of the true top-10 neighbours on average.

At `efSearch=320`, HNSW reaches 0.9994 recall at approximately 40 QPS. This shows that the custom implementation can approach exact-search recall while avoiding an exhaustive scan of all 50,000 vectors.

### Comparison with FAISS

FAISS is substantially faster than the Python implementation because its HNSW implementation is optimized native code with optimized low-level vector operations.

At `efSearch=320`:

| Engine | Recall@10 | QPS |
| :--- | ---: | ---: |
| Ours HNSW | 0.9994 | 40 |
| FAISS HNSW | 0.9998 | 544 |

FAISS also builds the graph much faster:

| Engine | Build time |
| :--- | ---: |
| Ours HNSW | 3693.7 s |
| FAISS HNSW | 29.48 s |

This comparison is useful for establishing the performance gap between a transparent Python implementation and an optimized production-oriented implementation.

The purpose of this project is not to outperform FAISS. The goal is to implement the core vector-search algorithm from scratch and make its behaviour measurable and understandable.

## Key Result

The benchmark shows that the from-scratch HNSW implementation provides a continuous speed/accuracy tradeoff controlled by `efSearch`.

Across the tested settings, recall increases from **0.9576 to 0.9994** as `efSearch` increases from **10 to 320**, while throughput decreases from **419 QPS to 40 QPS**.

The exact `FlatIndex` provides the ground truth at **1.0000 recall**, allowing the approximate results to be evaluated against exhaustive nearest-neighbour search.

The benchmark therefore demonstrates both required properties of the approximate index:

1. It provides substantially higher throughput than the project's exhaustive search at lower `efSearch` settings.
2. It provides a measurable accuracy knob, allowing recall to approach the exact-search result as more candidates are explored.