# Architecture

This document explains how the engine works, in enough detail that the owner
can study it and explain HNSW from memory. It is written in my own words from
the paper (Malkov & Yashunin, arXiv:1603.09320), not copied from a library.

## Layout

```
vectordb/
  distance.py    metrics (L2, cosine, inner product) as "smaller = closer"
  config.py      dataclass configs (HNSWConfig)
  hnsw.py        the HNSW index: graph, insert, search, heuristic, delete
  flat.py        brute-force exact index (ground truth for recall)
  storage.py     binary snapshot format + write-ahead log
  collection.py  ids + metadata + filtered search + durability, over an HNSW
benchmarks/      embedding generation + recall/latency benchmark
tests/
    conftest.py      deterministic test dataset fixture
    test_hnsw.py     HNSW correctness tests
scripts/
    demo.py          end-to-end insert/search/filter/delete/persistence demo
```

## The problem

Exact nearest-neighbor search means scanning every stored vector for each query:
O(n) distance computations. At a few thousand vectors that is fine (`flat.py`
does exactly this, and it is the ground truth we measure recall against). At
millions it is too slow. Approximate nearest neighbor (ANN) trades a little
recall for a large speedup. HNSW is one of the strongest ANN methods on CPU.

## HNSW in one paragraph

Build a graph where each vector is a node connected to a handful of nearby
vectors. To search, start somewhere and repeatedly walk to whichever neighbor is
closer to the query (greedy routing). A single flat graph gets stuck in local
minima and needs long-range links to cross the space quickly. HNSW solves that
with **layers**: a stack of graphs where the top layers are sparse (long hops
across the whole space) and the bottom layer contains every node (fine-grained
local search). You descend layer by layer, using each layer's result as the
entry point for the next, so the search zooms in like a skip list for geometry.

## The layered graph

```
 layer 2        (A)                      few nodes, long-range links
                 |  \
 layer 1     (A)-(D)---(G)               more nodes
              |    |     |
 layer 0  (A)(B)(C)(D)(E)(F)(G)(H)...     every node, dense local links
```

A node assigned top level `L` exists on layers `0..L`. Most nodes only live on
layer 0. The number of layers a node gets is random:

```
level = floor( -ln(uniform(0,1)) * mL )     with mL = 1 / ln(M)
```

This is sampling from a geometric distribution: the probability of reaching
level `l` decays by a factor of `M` per level, so if `M = 16` roughly 1 node in
16 reaches level 1, 1 in 256 reaches level 2, and so on. That decay is what
gives the expected O(log n) number of layers and O(log n) search. `test_hnsw.py`
checks that the empirical fraction of nodes above layer 0 is close to `1/M`.

Storage: all vectors live in one growing float32 matrix; a node's id is its row
index. `self._levels[node]` is its top layer. `self._links[node][layer]` is the
list of neighbor ids at that layer.

## Distance

The index only ever compares distances, so every metric is expressed as
"smaller means closer":

- **L2**: squared Euclidean distance (we skip the square root, it is monotonic).
- **cosine**: vectors are L2-normalized on insert, then cosine distance is
  `1 - dot`. After normalization it is the same code path as inner product.
- **inner product**: `-dot`, negated so larger similarity sorts first.

Distances are computed in batches with numpy: for a query and a set of neighbor
rows we do one vectorized operation instead of a Python loop. numpy is used only
for this arithmetic; the graph logic is pure Python.

## Search (Algorithm 2 and 5)

Two pieces.

**Greedy descent (ef = 1).** From the entry point on a layer, look at the
current node's neighbors, jump to the closest one to the query, repeat until no
neighbor is closer. This finds a local minimum on that layer fast. Used on every
layer above 0 during a query (`_greedy_descend`).

**Search-layer (ef > 1).** On layer 0 (and during construction) we need more
than one candidate, so we keep a dynamic list of the `ef` best nodes found so
far. Two heaps drive it:

- a min-heap of *candidates to explore* (nearest first),
- a max-heap of *results kept* (farthest first), capped at `ef`.

Pop the nearest candidate; if it is farther than the worst kept result and we
already have `ef` results, stop (nothing closer can be reached). Otherwise expand
its unvisited neighbors, and for each one that is closer than the current worst
(or while we still have room), push it into both heaps and evict the worst if we
exceed `ef`. Larger `ef` explores more of the graph: higher recall, more work.
That is the recall/latency knob the benchmark sweeps.

**Full query.** Greedy-descend from the top layer down to layer 1 to get a good
entry point, then run search-layer with `ef = max(efSearch, k)` on layer 0, sort,
and return the top `k` live results.

## Insert (Algorithm 1)

1. Draw a random top level `L` for the new node (formula above).
2. Greedy-descend from the current entry point through the layers above `L`,
   using `ef = 1`, to arrive near the new node's neighborhood cheaply.
3. From `min(topLevel, L)` down to 0, on each layer:
   - run search-layer with `efConstruction` to gather candidate neighbors,
   - pick up to `M` of them with the selection heuristic (below),
   - add links both ways (new node <-> chosen neighbors),
   - if a neighbor now exceeds its max degree (`M` above layer 0, `M0 = 2M` on
     layer 0), re-run the heuristic on its link list to prune it back. Keeping
     layer 0 twice as dense is the paper's recommendation and matters for recall.
4. If `L` is above the current top, the new node becomes the entry point.

`efConstruction` controls graph quality: a larger candidate list during build
produces a better-connected graph and higher query recall, at the cost of slower
inserts.

## The neighbor-selection heuristic (Algorithm 4)

The naive choice is "keep the M nearest candidates". That clusters all of a
node's links in one direction and leaves whole regions unreachable, so greedy
search gets stuck. The heuristic instead keeps a candidate **only if it is closer
to the base node than to any already-selected neighbor**:

```
for each candidate e, nearest first:
    if dist(e, base) < min over selected r of dist(e, r):
        select e
    else:
        set e aside (pruned)
```

Intuitively: if `e` is closer to something we already picked than to the base,
that region is already covered, so `e` is redundant. This spreads links across
directions and is what makes the small world navigable.

Two options from the paper are implemented:

- **keep_pruned_connections** (on by default): if the diversity filter selects
  fewer than `M`, backfill from the set-aside candidates so nodes stay
  well-connected.
- **extend_candidates** (off by default): also consider the neighbors of the
  candidates. Rarely needed; matches common implementations leaving it off.

## Deletion

Deletes are soft. `mark_deleted()` adds the internal node id to a tombstone
set. Deleted nodes remain in the graph so their links can still provide
connectivity during traversal, but `search()` excludes tombstoned nodes from
the returned results.

This makes deletion cheap and avoids immediately modifying the graph structure.
The tradeoff is that deleted nodes continue to consume memory and may still be
visited during search.

`HNSW.rebuild()` creates a fresh graph containing only live vectors while
preserving the existing internal node ids of surviving vectors. This is useful
for removing tombstones after a deletion-heavy workload.

The public `Collection.delete()` operation uses this deletion mechanism and
updates the collection's id mapping and persistence state accordingly.

Soft deletion is therefore the normal fast path, while rebuilding is the
compaction mechanism when tombstones become costly.

## Persistence and durability

Two mechanisms in `storage.py`, described honestly.

**Snapshot** (`save_hnsw` / `load_hnsw`): a compact little-endian binary file
holding the header (dim, metric, M, M0, efConstruction, size, entry point, max
level), then per node its level and neighbor lists per layer, then the tombstone
set, then the raw float32 vector matrix. The exact byte layout is documented at
the top of `storage.py`. It is written to a temp file and `os.replace`d into
place, so a snapshot is atomic: you never see a half-written graph.

**Write-ahead log** (`WriteAheadLog`): an append-only log. Every insert and
delete is framed (`magic | length | payload | crc32`), written, and **fsync'd
before** the operation is applied to the in-memory index. So an operation is
durable the moment it is acknowledged. On load we read the snapshot, then replay
whatever remains in the WAL (everything since the last snapshot). A crash
mid-write leaves a torn tail whose length or crc will not check out; replay stops
at that record and discards it, which is exactly the right recovery boundary.
`save()` folds the WAL into a fresh snapshot and truncates it.

What this guarantees, stated plainly: any insert/delete that returned to the
caller survives a process kill, because it was fsync'd to the WAL first. What it
does **not** do: it is not a distributed log, there is no group commit or
batched fsync (so per-record fsync makes inserts slower but simpler to reason
about), and recovery replay re-inserts WAL records into a fresh graph rather than
restoring the exact original graph. The recovered index is correct and
searchable; it is not bit-identical to the pre-crash graph, which is fine for an
ANN index.

## Collection layer

`Collection` maps user string ids to internal node ids, stores per-vector
metadata dicts, and owns the WAL. Filtered kNN has two implementations:

- **prefilter**: gather node ids whose metadata matches, then brute-force scan
  just those vectors. Exact by construction; cost is O(matches). Best when the
  filter is selective.
- **postfilter**: run the ANN search with an enlarged candidate list, then drop
  non-matching hits. Cheap when the filter passes most vectors, but it can return
  fewer than `k` when the filter is very selective, because the graph traversal
  is filter-blind.

Both are shipped so the tradeoff is a per-query choice, not a baked-in decision.
