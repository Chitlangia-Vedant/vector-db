# vectordb

A vector database engine written from scratch in Python, built around an HNSW
index implemented directly from the paper by Malkov and Yashunin rather than
wrapping an existing vector-search library.

The project implements approximate nearest-neighbor search, exact brute-force
search for ground truth, metadata and filtered kNN, soft deletion, persistence,
and a configurable HNSW search parameter for measuring the speed/accuracy
tradeoff.

The point of this project is understanding, not beating C. The graph
construction, greedy search, neighbor-selection heuristic, level generation,
and search logic are implemented from scratch. NumPy is used for vector
arithmetic and distance calculations.

FAISS is used only as an external benchmark reference. It is not used to
implement the vector database.

## Why it exists

I wanted to understand how systems such as FAISS, hnswlib, Pinecone, and Qdrant
perform nearest-neighbor search in high-dimensional space, so I built the core
of one.

HNSW is the interesting part: a layered graph that can be navigated greedily
from sparse long-range connections toward dense local connections.

Reading the paper is one thing. Implementing the graph, verifying it against
exact search, supporting deletion and persistence, and measuring the
speed/accuracy tradeoff makes the algorithm much easier to understand.

## Features

* **HNSW index from scratch** (`vectordb/hnsw.py`)

  * Multi-layer graph
  * Probabilistic level assignment
  * Greedy search through upper layers
  * Configurable `M`
  * Configurable `efConstruction`
  * Configurable `efSearch`
  * Neighbor-selection heuristic based on Algorithm 4 from the HNSW paper
  * Optional pruned-candidate connections
  * L2, cosine, and inner-product distance metrics

* **Exact brute-force index** (`vectordb/flat.py`)

  * Scans every vector
  * Used as the ground truth for ANN recall measurements

* **Soft deletion**

  * Deleted nodes are marked with tombstones
  * Deleted nodes are excluded from search results
  * Tombstoned nodes remain in the graph so their connections can still help
    traversal

* **Rebuilding**

  * `HNSW.rebuild()` creates a fresh graph containing the live vectors
  * Useful for removing the cost of accumulated tombstones

* **Persistence** (`vectordb/storage.py`)

  * Binary snapshots
  * Write-ahead log
  * CRC-protected WAL records
  * Durable writes using `fsync`
  * Recovery after process interruption

* **Collection layer** (`vectordb/collection.py`)

  * User-facing string IDs
  * Metadata
  * Filtered nearest-neighbor queries
  * Persistence and deletion handling

* **Benchmarking** (`benchmarks/`)

  * Exact ground truth
  * Recall@10
  * p50 latency
  * p95 latency
  * Queries per second
  * `efSearch` sweep
  * FAISS comparison

* **End-to-end demo** (`scripts/demo.py`)

  * Insert vectors
  * Search
  * Metadata filtering
  * Delete
  * Save and reload

## Architecture

```text
vectordb/
    distance.py
    config.py
    hnsw.py
    flat.py
    storage.py
    collection.py

benchmarks/
    generate_embeddings.py
    run_benchmark.py
    data/
    results/

scripts/
    demo.py

tests/
    conftest.py
    test_hnsw.py

docs/
    ARCHITECTURE.md
    BENCHMARKS.md
```

The `Collection` layer maps user-facing IDs to internal HNSW node IDs, stores
metadata, and coordinates persistence.

The HNSW index stores vectors in a growing float32 matrix. Each node has a
randomly assigned maximum layer and a neighbor list for every layer in which
it exists.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the detailed
implementation explanation.

## Setup

Create a virtual environment and install the project:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install the core package:

```bash
pip install -e .
```

For development and tests:

```bash
pip install pytest
```

For reproducing the embedding benchmark:

```bash
pip install sentence-transformers
```

FAISS is optional and is required only for the external benchmark comparison.

## Usage

### Library

```python
import numpy as np

from vectordb import Collection, HNSWConfig, Metric

coll = Collection.open(
    "data/docs",
    HNSWConfig(dim=768, metric=Metric.COSINE),
)

vectors = np.random.randn(3, 768).astype(np.float32)

coll.insert(
    vectors,
    ids=["a", "b", "c"],
    metadatas=[
        {"cat": "news"},
        {"cat": "sports"},
        {"cat": "tech"},
    ],
)

hits = coll.query(vectors[0], k=2)

hits = coll.query(
    vectors[0],
    k=2,
    flt={"cat": "news"},
)

coll.delete("a")

coll.save("data/docs")
```

### Demo

Run the complete insert/search/filter/delete/persistence demonstration:

```bash
python scripts/demo.py
```

The demo verifies that vectors can be inserted, queried, filtered, deleted,
saved, and loaded again.

## Benchmarks

The benchmark uses **50,500 real AG News texts** embedded with
`sentence-transformers/all-mpnet-base-v2`.

The dataset is split into:

* **50,000 base vectors**
* **500 query vectors**
* **768 dimensions**
* **k = 10**

The exact answers are calculated using the project's `FlatIndex`. These exact
results are the ground truth used to calculate recall@10.

The HNSW benchmark keeps `M = 16` and `efConstruction = 200` fixed while
sweeping `efSearch` across:

```text
10, 20, 40, 80, 160, 320
```

For every setting the benchmark measures:

* Recall@10
* p50 query latency
* p95 query latency
* Queries per second

### Results

#### Exact baseline

| Engine         | Recall@10 | p50 ms | p95 ms | QPS |
| -------------- | --------: | -----: | -----: | --: |
| Ours FlatIndex |    1.0000 | 118.42 | 133.37 |   8 |
| FAISS Flat     |    1.0000 | 42.707 |    n/a |  23 |

#### Our HNSW

| efSearch | Recall@10 | p50 ms | p95 ms | QPS |
| -------: | --------: | -----: | -----: | --: |
|       10 |    0.9576 |  2.264 |  3.639 | 419 |
|       20 |    0.9854 |  2.378 |  3.702 | 395 |
|       40 |    0.9942 |  3.990 |  5.918 | 239 |
|       80 |    0.9968 |  7.518 | 22.521 | 111 |
|      160 |    0.9984 | 12.713 | 18.502 |  76 |
|      320 |    0.9994 | 22.959 | 37.887 |  40 |

#### FAISS HNSW

| efSearch | Recall@10 | p50 ms | p95 ms |  QPS |
| -------: | --------: | -----: | -----: | ---: |
|       10 |    0.9368 |  0.130 |  0.195 | 6499 |
|       20 |    0.9770 |  0.186 |  0.270 | 5190 |
|       40 |    0.9922 |  0.322 |  0.465 | 3008 |
|       80 |    0.9968 |  0.550 |  0.773 | 1775 |
|      160 |    0.9982 |  1.003 |  1.447 |  982 |
|      320 |    0.9998 |  1.796 |  2.550 |  544 |

Our HNSW therefore provides a clear speed/accuracy curve:

```text
efSearch   recall@10   QPS
10         0.9576      419
20         0.9854      395
40         0.9942      239
80         0.9968      111
160        0.9984       76
320        0.9994       40
```

At `efSearch=40`, the implementation reaches 0.9942 recall@10 at 239 QPS.
Increasing `efSearch` to 320 raises recall to 0.9994 while reducing throughput
to 40 QPS.

The exact `FlatIndex` provides 1.0000 recall and approximately 8 QPS on the
same benchmark.

FAISS is much faster because it is an optimized native implementation. At
`efSearch=320`, FAISS reaches 0.9998 recall at 544 QPS compared with 0.9994
recall at 40 QPS for this implementation.

The purpose of the comparison is not to beat FAISS. It demonstrates that the
from-scratch implementation produces high-recall nearest-neighbor results
while exposing the algorithm and its tradeoffs directly in Python.

Full benchmark methodology and measurements are documented in
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md).

Raw results are stored in:

```text
benchmarks/results/results.json
```

## Reproducing the benchmark

The embedding generation step downloads the AG News dataset and generates
embeddings using `sentence-transformers/all-mpnet-base-v2`.

```bash
python benchmarks/generate_embeddings.py --limit 50500
```

This produces:

```text
benchmarks/data/embeddings.npz
```

The benchmark can then be run without regenerating the embeddings:

```bash
python benchmarks/run_benchmark.py \
    --ef-values 10,20,40,80,160,320
```

On Windows PowerShell:

```powershell
python benchmarks/run_benchmark.py --ef-values 10,20,40,80,160,320
```

Embedding generation is intentionally not part of the benchmark timing.
The generated embeddings are reused for subsequent benchmark runs.

## Tests

Run the test suite with:

```bash
pytest -q
```

The current test suite focuses on the HNSW implementation and verifies:

* Recall against exact brute-force search
* L2, cosine, and inner-product metrics
* Recall behaviour as `efSearch` increases
* Random HNSW layer distribution
* Empty and single-node searches
* Sorted search results

The benchmark provides the large-scale correctness and performance
measurement, while the tests provide smaller deterministic correctness checks.

## Design decisions

### Exact search as ground truth

The project contains its own brute-force `FlatIndex` instead of relying on
FAISS to determine correctness.

For every query, the exact index scans all 50,000 base vectors and returns the
true top-k results. HNSW results are compared against these results to
calculate recall.

This makes the benchmark independent of the external ANN implementation.

### `efSearch` as the accuracy knob

`efSearch` controls how many candidates HNSW explores during a query.

Higher values generally improve recall while increasing query work.

This parameter can be changed after building the index, which makes it useful
for measuring the recall/latency tradeoff.

### `efConstruction` as the build-time knob

`efConstruction` controls how much work is performed while constructing the
graph.

Higher values can produce a better graph, but increase build time.

Unlike `efSearch`, changing `efConstruction` requires rebuilding the index.

### Soft deletion

Deletion uses tombstones rather than immediately modifying the graph.

A deleted node remains available to graph traversal but is excluded from
returned search results.

This makes deletion cheap and preserves graph connectivity. The tradeoff is
that deleted nodes continue to consume memory until the graph is rebuilt.

### NumPy usage

NumPy is used for vector storage and arithmetic, including distance
calculations.

The HNSW graph traversal, candidate management, neighbor selection, and graph
updates are implemented directly in Python.

## Challenges

### Neighbor selection

A simple implementation that keeps only the closest `M` candidates can create
clusters of similar links.

The HNSW neighbor-selection heuristic instead considers whether a candidate
provides a useful direction relative to already selected neighbors.

Implementing this correctly was important for producing a navigable graph and
high recall.

### Level generation

Each node receives a random maximum layer using the HNSW level-generation
formula:

```text
level = floor(-ln(u) * mL)
mL = 1 / ln(M)
```

Most nodes therefore exist only at layer 0, while progressively fewer nodes
appear at higher layers.

### Python insertion speed

The main performance limitation is graph construction in Python.

Each insertion performs graph traversal, distance calculations, candidate
management, and neighbor updates. Although NumPy accelerates vector
arithmetic, the graph algorithm still performs many small operations from
Python.

The benchmark makes this limitation visible rather than hiding it.

### Persistence and recovery

The storage layer separates durable logging from in-memory graph state.

The WAL uses framed records and checksums so incomplete records at the end of a
file can be detected during recovery.

Snapshots provide a durable representation of the current collection while
the WAL records subsequent operations.

## What I learned

* HNSW uses sparse upper layers for long-range navigation and a dense bottom
  layer for local search.
* The neighbor-selection heuristic is important because graph connectivity is
  not determined by distance alone.
* `efSearch` provides a practical speed/accuracy control after the graph has
  been built.
* Exact brute-force search provides a simple and reliable ground truth for
  measuring ANN recall.
* Soft deletion preserves graph connectivity but leaves tombstones that may
  eventually require rebuilding.
* Persistence requires careful ordering and recovery logic, not just writing
  data to disk.
* Optimized native ANN libraries are much faster than a straightforward Python
  implementation, but implementing the algorithm directly makes its behaviour
  easier to inspect and understand.

## What I would do differently

* Move the graph traversal and insertion hot path into a compiled extension
  such as Cython, Rust, or C.
* Replace Python adjacency lists with more compact contiguous structures.
* Make filtered search more deeply integrated with graph traversal rather than
  relying on exact scanning for selective filters.
* Add vector quantization to reduce memory usage.
* Add sharding and replication for a distributed deployment.
* Improve checkpointing so recovery requires less WAL replay.

