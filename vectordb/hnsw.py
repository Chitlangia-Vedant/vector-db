"""Hierarchical Navigable Small World (HNSW) index, implemented from the paper.

Reference:
    Yu. A. Malkov, D. A. Yashunin, "Efficient and robust approximate nearest
    neighbor search using Hierarchical Navigable Small World graphs", IEEE
    Transactions on Pattern Analysis and Machine Intelligence, 2018.
    arXiv:1603.09320

The algorithm numbering in the comments (Algorithm 1..5) refers to that paper.
This is a from-scratch implementation: the graph, the greedy search, the level
generation and the neighbor-selection heuristic are all written here. numpy is
used only for the vector arithmetic (batched distance computation).

Layout of the graph
-------------------
Vectors live in a single growing float32 matrix, ``self._matrix``; a node's
internal id is its row index. ``self._levels[node]`` is the node's top layer.
``self._links[node]`` is a list indexed by layer; ``self._links[node][lc]`` is
the list of neighbor node ids at layer ``lc``. Every node exists on layers
0..levels[node].

Deletion is a soft delete: the node id is added to ``self._deleted`` and skipped
in query *results*, but is kept in the graph so it can still be traversed (its
links keep the small world connected). ``rebuild()`` compacts the index by
re-inserting only the live vectors into a fresh graph.
"""

from __future__ import annotations

import heapq
import math
from typing import Iterable

import numpy as np

from .config import HNSWConfig
from .distance import Metric, distance_batch, normalize


class HNSW:
    def __init__(self, config: HNSWConfig):
        self.config = config
        self.metric: Metric = config.metric
        self._rng = np.random.default_rng(config.seed)

        # Vector storage: amortized-doubling float32 matrix.
        self._capacity = 0
        self._size = 0
        self._matrix = np.empty((0, config.dim), dtype=np.float32)

        # Graph.
        self._levels: list[int] = []          # node -> top layer
        self._links: list[list[list[int]]] = []  # node -> layer -> neighbor ids
        self._entry_point: int | None = None
        self._max_level = -1

        self._deleted: set[int] = set()

    # ------------------------------------------------------------------ utils
    @property
    def size(self) -> int:
        """Number of live (non-deleted) vectors."""
        return self._size - len(self._deleted)

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph, including tombstoned ones."""
        return self._size

    def _ensure_capacity(self, needed: int) -> None:
        if needed <= self._capacity:
            return
        new_cap = max(16, self._capacity * 2, needed)
        grown = np.empty((new_cap, self.config.dim), dtype=np.float32)
        grown[: self._size] = self._matrix[: self._size]
        self._matrix = grown
        self._capacity = new_cap

    def _random_level(self) -> int:
        # Algorithm 1: l = floor(-ln(unif(0,1)) * mL)
        u = self._rng.random()
        return int(math.floor(-math.log(u + 1e-30) * self.config.mL))

    def _distances(self, query: np.ndarray, nodes: list[int]) -> np.ndarray:
        rows = self._matrix[nodes]
        return distance_batch(query, rows, self.metric)

    # --------------------------------------------------------------- insertion
    def add(self, vector: np.ndarray, node_id: int | None = None) -> int:
        """Insert one vector, returning its internal node id."""
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.config.dim:
            raise ValueError(f"expected dim {self.config.dim}, got {vector.shape[0]}")
        if self.metric is Metric.COSINE:
            vector = normalize(vector)

        node = self._size if node_id is None else node_id
        self._ensure_capacity(node + 1)
        self._matrix[node] = vector
        # Extend bookkeeping lists to cover this node id.
        while len(self._levels) <= node:
            self._levels.append(0)
            self._links.append([])
        self._size = max(self._size, node + 1)

        level = self._random_level()
        self._levels[node] = level
        self._links[node] = [[] for _ in range(level + 1)]

        if self._entry_point is None:
            self._entry_point = node
            self._max_level = level
            return node

        ep = self._entry_point
        top = self._max_level

        # Phase 1 (Algorithm 1): descend greedily from the top down to level+1.
        for lc in range(top, level, -1):
            ep = self._greedy_descend(vector, ep, lc)

        M = self.config.M
        M0 = self.config.M0

        # Phase 2: connect from min(top, level) down to layer 0.
        entry_points = [ep]
        for lc in range(min(top, level), -1, -1):
            candidates = self._search_layer(vector, entry_points, self.config.ef_construction, lc)
            m = M0 if lc == 0 else M
            neighbors = self._select_neighbors_heuristic(vector, candidates, m, lc)

            self._links[node][lc] = list(neighbors)
            # Add reverse links and shrink over-full neighbors.
            for neigh in neighbors:
                self._links[neigh][lc].append(node)
                m_max = M0 if lc == 0 else M
                if len(self._links[neigh][lc]) > m_max:
                    neigh_vec = self._matrix[neigh]
                    existing = self._links[neigh][lc]
                    dists = self._distances(neigh_vec, existing)
                    cand = list(zip(dists.tolist(), existing))
                    pruned = self._select_neighbors_heuristic(neigh_vec, cand, m_max, lc)
                    self._links[neigh][lc] = list(pruned)

            # candidates become entry points for the next lower layer.
            entry_points = [n for _, n in candidates]

        if level > top:
            self._entry_point = node
            self._max_level = level
        return node

    def _greedy_descend(self, query: np.ndarray, entry: int, layer: int) -> int:
        """Greedy search with ef=1: walk to the local nearest node on ``layer``."""
        best = entry
        best_dist = float(self._distances(query, [entry])[0])
        improved = True
        while improved:
            improved = False
            neighbors = self._links[best][layer] if layer < len(self._links[best]) else []
            if not neighbors:
                break
            dists = self._distances(query, neighbors)
            idx = int(np.argmin(dists))
            if dists[idx] < best_dist:
                best_dist = float(dists[idx])
                best = neighbors[idx]
                improved = True
        return best

    def _search_layer(
        self, query: np.ndarray, entry_points: list[int], ef: int, layer: int
    ) -> list[tuple[float, int]]:
        """Algorithm 2: greedy best-first search on one layer.

        Returns up to ``ef`` (distance, node) pairs, the closest found. The list
        is not sorted; callers that need ordering sort it themselves.
        """
        visited: set[int] = set(entry_points)
        init_d = self._distances(query, entry_points)

        # candidates: min-heap by distance (nearest first).
        candidates: list[tuple[float, int]] = [
            (float(d), n) for d, n in zip(init_d.tolist(), entry_points)
        ]
        heapq.heapify(candidates)
        # results: max-heap by distance (store negative distance).
        results: list[tuple[float, int]] = [
            (-float(d), n) for d, n in zip(init_d.tolist(), entry_points)
        ]
        heapq.heapify(results)
        while len(results) > ef:
            heapq.heappop(results)

        while candidates:
            c_dist, c = heapq.heappop(candidates)
            worst = -results[0][0]  # largest distance currently kept
            if c_dist > worst and len(results) >= ef:
                break

            neighbors = self._links[c][layer] if layer < len(self._links[c]) else []
            fresh = [n for n in neighbors if n not in visited]
            if not fresh:
                continue
            visited.update(fresh)
            dists = self._distances(query, fresh)
            for d, n in zip(dists.tolist(), fresh):
                worst = -results[0][0]
                if len(results) < ef or d < worst:
                    heapq.heappush(candidates, (d, n))
                    heapq.heappush(results, (-d, n))
                    if len(results) > ef:
                        heapq.heappop(results)

        return [(-nd, n) for nd, n in results]

    def _select_neighbors_heuristic(
        self,
        base: np.ndarray,
        candidates: list[tuple[float, int]],
        M: int,
        layer: int,
    ) -> list[int]:
        """Algorithm 4: neighbor selection with the diversity heuristic.

        A candidate ``e`` is kept only if it is closer to ``base`` than to any
        already-selected neighbor. This spreads links across directions instead
        of clustering them all toward the nearest blob, which is what makes the
        graph navigable. ``keep_pruned_connections`` backfills from the rejected
        set when fewer than M survive the filter.
        """
        cfg = self.config
        # working queue: nearest-first min-heap
        working: list[tuple[float, int]] = list(candidates)
        heapq.heapify(working)

        if cfg.extend_candidates:
            seen = {n for _, n in candidates}
            extra: list[tuple[float, int]] = []
            for _, e in candidates:
                nbrs = self._links[e][layer] if layer < len(self._links[e]) else []
                for adj in nbrs:
                    if adj not in seen:
                        seen.add(adj)
                        d = float(self._distances(base, [adj])[0])
                        extra.append((d, adj))
            for item in extra:
                heapq.heappush(working, item)

        selected: list[int] = []
        discarded: list[tuple[float, int]] = []

        while working and len(selected) < M:
            cand_dist, e = heapq.heappop(working)
            if not selected:
                selected.append(e)
                continue
            # Keep e only if it is closer to base than to any selected neighbor.
            e_vec = self._matrix[e]
            to_selected = distance_batch(e_vec, self._matrix[selected], self.metric)
            if cand_dist < float(to_selected.min()):
                selected.append(e)
            else:
                discarded.append((cand_dist, e))

        if cfg.keep_pruned_connections:
            heapq.heapify(discarded)
            while discarded and len(selected) < M:
                _, e = heapq.heappop(discarded)
                selected.append(e)

        return selected

    # ------------------------------------------------------------------ search
    def search(
        self, query: np.ndarray, k: int, ef_search: int | None = None
    ) -> list[tuple[int, float]]:
        """Return up to ``k`` (node_id, distance) pairs, nearest first.

        Tombstoned nodes are traversed for connectivity but excluded from the
        returned results.
        """
        if self._entry_point is None:
            return []
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if self.metric is Metric.COSINE:
            query = normalize(query)
        ef = ef_search if ef_search is not None else self.config.ef_search
        ef = max(ef, k)

        ep = self._entry_point
        for lc in range(self._max_level, 0, -1):
            ep = self._greedy_descend(query, ep, lc)

        found = self._search_layer(query, [ep], ef, 0)
        found.sort(key=lambda x: x[0])
        out: list[tuple[int, float]] = []
        for dist, node in found:
            if node in self._deleted:
                continue
            out.append((node, dist))
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------------------ delete
    def mark_deleted(self, node: int) -> None:
        if 0 <= node < self._size:
            self._deleted.add(node)

    def is_deleted(self, node: int) -> bool:
        return node in self._deleted

    def rebuild(self) -> "HNSW":
        """Return a fresh compacted index containing only live vectors.

        The returned index reuses the *same* node ids for surviving nodes so the
        owning collection's id maps stay valid. Deleted node ids simply do not
        appear in the new graph (their rows are kept but unreachable, matching
        how the collection remaps).
        """
        new = HNSW(self.config)
        for node in range(self._size):
            if node in self._deleted:
                continue
            new.add(self._matrix[node].copy(), node_id=node)
        return new

    # ------------------------------------------------------------------ vector
    def get_vector(self, node: int) -> np.ndarray:
        return self._matrix[node].copy()

    # ---------------------------------------------------------------- introspection
    def level_histogram(self) -> dict[int, int]:
        hist: dict[int, int] = {}
        for node in range(self._size):
            if node in self._deleted:
                continue
            lvl = self._levels[node]
            hist[lvl] = hist.get(lvl, 0) + 1
        return hist

    def bulk_add(self, vectors: Iterable[np.ndarray]) -> list[int]:
        return [self.add(v) for v in vectors]
