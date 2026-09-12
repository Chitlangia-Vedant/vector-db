# Architecture

This document describes the main data structures and algorithms used by the vector database.

## Project Layout

```text
vectordb/
    distance.py      Distance metrics
    config.py        HNSW configuration
    hnsw.py          HNSW graph and search
    flat.py          Exact brute-force index
    storage.py       Persistence
    collection.py    IDs, metadata, and queries

benchmarks/
    generate_embeddings.py
    run_benchmark.py

tests/
    conftest.py
    test_hnsw.py

scripts/
    demo.py
```

## HNSW

The approximate index is a Hierarchical Navigable Small World graph.

Layer 0 contains every node. Higher layers contain progressively fewer nodes and provide long-range routing.

```text
Layer 2:                  [A]
                          |
Layer 1:          [B] --- [A] --- [C]
                   |       |       |
Layer 0:    [D]---[B]---[E]---[A]---[C]---[F]
             \     |     |     |     /
              [G]--+-----+-----+---[H]
```

A node with maximum level `L` exists on every layer from `0` through `L`.

### Level Assignment

Levels follow an exponentially decreasing distribution:

$$
level = \lfloor-\ln(U) \times m_L\rfloor
$$

where:

$$
m_L = \frac{1}{\ln(M)}
$$

`M` is the maximum number of connections on upper layers.

For `M = 16`, approximately 1 in 16 nodes reaches level 1 and 1 in 256 reaches level 2.

## Data Structures

The HNSW index maintains:

```text
_matrix        float32 vector matrix
_levels        maximum level for each node
_links         neighbor lists per node and layer
_deleted       tombstoned node IDs
_entry_point   starting node for search
_max_level     highest graph layer
```

Node IDs correspond to rows in `_matrix`.

NumPy handles vector storage and numerical operations. Graph traversal and heap management use Python data structures.

## Distance Metrics

All metrics are represented as distances where smaller values are better.

**Squared L2**

$$
d(u,v)=\|u-v\|^2
$$

The square root is omitted because it does not change distance ordering.

**Cosine**

For L2-normalized vectors:

$$
d(u,v)=1-(u\cdot v)
$$

The query is normalized before cosine search.

**Inner Product**

$$
d(u,v)=-(u\cdot v)
$$

This converts similarity maximization into distance minimization.

## Search

Search begins at `_entry_point` on the highest layer.

Upper layers use greedy routing:

1. Examine the current node's neighbors.
2. Move to a closer neighbor when one exists.
3. Stop at the local minimum.
4. Continue from that node on the next layer.

Layer 0 performs exploratory search using:

* a min-heap of candidates,
* a bounded max-heap of results,
* a visited-node set.

The search depth is:

$$
ef=\max(efSearch,k)
$$

Higher `efSearch` values examine more candidates and generally improve recall while reducing query throughput.

Deleted nodes are excluded from returned results but can still participate in graph traversal.

## Insertion

Insertion:

1. Samples the node's maximum level.
2. Adds its vector to `_matrix`.
3. Descends from the current entry point to the node's target level.
4. Searches each relevant layer using `efConstruction`.
5. Selects neighboring nodes.
6. Creates bidirectional links.
7. Prunes connections exceeding the layer limit.
8. Updates the entry point if the new node reaches a higher level.

The layer 0 connection limit is larger than the upper-layer limit.

## Neighbor Selection

Candidates are selected using a diversity-aware heuristic rather than simply taking the closest nodes.

The heuristic prefers connections that provide useful graph coverage relative to already selected neighbors. This helps prevent connections from concentrating in one local direction.

The implementation is contained in `vectordb/hnsw.py`.

## Deletion

Deletion uses tombstones.

```text
mark_deleted(node)
        |
        v
add node ID to _deleted
        |
        v
node remains in graph
        |
        v
excluded from final search results
```

Deleted nodes remain connected so their existing edges can still assist graph traversal.

This avoids expensive graph rewiring during every deletion, but deleted nodes continue to consume memory and may add traversal overhead.

### Rebuild

`HNSW.rebuild()` removes tombstones by creating a new graph and reinserting only live nodes.

Surviving node IDs are preserved.

## Persistence

`vectordb/storage.py` stores the state required to reload an index, including:

* vectors
* node levels
* graph links
* entry point
* maximum level
* deleted nodes
* index configuration

Snapshots are written to a temporary file and atomically replaced with `os.replace()`.

## Collection Layer

`Collection` provides the application-level interface around the index.

It handles:

* string document IDs
* vector insertion
* metadata
* nearest-neighbor queries
* deletion
* saving and loading

External IDs are mapped to internal integer node IDs:

```text
"doc_001" -> 0
"doc_002" -> 1
"doc_003" -> 2
```

Metadata filtering is handled by the collection layer rather than the HNSW graph.

## Exact Search

`FlatIndex` performs exhaustive search over every stored vector.

It provides exact nearest-neighbor results and is therefore used as the ground truth for evaluating HNSW recall.

## Benchmark Architecture

The benchmark uses:

```text
Dataset:          AG News
Embedding model:  all-mpnet-base-v2
Dimensions:       768
Base vectors:     50,000
Query vectors:    500
k:                10
M:                16
efConstruction:   200
```

The benchmark evaluates:

```text
efSearch = 10, 20, 40, 80, 160, 320
```

For each setting it records:

* Recall@10
* p50 latency
* p95 latency
* Queries per second

`FlatIndex` provides the exact ground truth. FAISS is used only as an external comparison and is not part of the custom search implementation.

## Design Tradeoffs

**FlatIndex vs HNSW**

Flat search guarantees exact results but scans every vector. HNSW reduces the search space and trades some recall for higher throughput.

**Tombstones vs graph rewiring**

Tombstones make deletion cheap. The cost is additional memory and traversal work until `rebuild()` is performed.

**Python and NumPy**

The HNSW graph logic is implemented in Python, while NumPy handles vector storage and numerical calculations. This keeps the implementation transparent while avoiding high-level vector-search libraries.

**efSearch**

Lower values favor speed. Higher values favor recall. The benchmark measures this tradeoff across multiple settings rather than using a single accuracy value.
