"""A Collection ties an HNSW index together with user-facing ids, per-vector
metadata, filtered search and durable persistence.

Identity model
--------------
Users address vectors by an ``external_id`` (any string). Internally each vector
gets a monotonically increasing integer node id (the HNSW row index). Two maps
keep them in sync: ``_ext_to_node`` and ``_node_to_ext``. Deletes are soft: the
node is tombstoned in the HNSW graph and removed from the id maps, so it stops
appearing in results but still helps connect the graph until ``compact()``.

Durability
----------
Every insert and delete is appended to a write-ahead log and fsync'd *before*
the in-memory index is touched. ``save()`` writes a full snapshot (graph +
metadata) and truncates the WAL. On ``load()`` we read the snapshot, then replay
whatever is left in the WAL (everything written since the last snapshot). See
docs/ARCHITECTURE.md for the honest discussion of what this does and does not
guarantee.

Filtered search
---------------
Two strategies, both implemented, selectable per query:

  - ``prefilter``: collect the node ids whose metadata matches, then do an exact
    brute-force scan over just those vectors. Correct by construction. Cost is
    O(matches), so this wins when the filter is selective.
  - ``postfilter``: run the ANN search with an enlarged candidate list, then drop
    non-matching results. Cheap when the filter passes most vectors, but it can
    return fewer than k (or miss some) when the filter is very selective, since
    the ANN traversal is filter-blind.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .config import HNSWConfig
from .distance import Metric, distance_batch, prepare
from .hnsw import HNSW
from .storage import (
    OP_DELETE,
    OP_INSERT,
    WriteAheadLog,
    load_hnsw,
    save_hnsw,
)

Filter = dict | Callable[[dict], bool]


@dataclass
class QueryResult:
    id: str
    distance: float
    metadata: dict


def _matches(meta: dict, flt: Filter) -> bool:
    if flt is None:
        return True
    if callable(flt):
        return bool(flt(meta))
    # dict filter: every key must equal (or be "in" a provided list).
    for key, want in flt.items():
        have = meta.get(key)
        if isinstance(want, (list, tuple, set)):
            if have not in want:
                return False
        elif have != want:
            return False
    return True


class Collection:
    def __init__(self, name: str, config: HNSWConfig):
        self.name = name
        self.config = config
        self.index = HNSW(config)
        self._ext_to_node: dict[str, int] = {}
        self._node_to_ext: dict[int, str] = {}
        self._metadata: dict[int, dict] = {}
        self._wal: WriteAheadLog | None = None

    # ------------------------------------------------------------ persistence
    def _attach_wal(self, path: str) -> None:
        self._wal = WriteAheadLog(path)
        self._wal.open()

    # ------------------------------------------------------------------ insert
    def insert(
        self,
        vectors,
        ids: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """Insert a batch. Each record is WAL-durable before it enters the graph."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids length must match number of vectors")
        if metadatas is None:
            metadatas = [{} for _ in ids]

        for vec, ext_id, meta in zip(vectors, ids, metadatas):
            if self._wal is not None:
                self._wal.log_insert(ext_id, meta, vec)
            self._apply_insert(ext_id, meta, vec)

    def _apply_insert(self, ext_id: str, meta: dict, vec: np.ndarray) -> None:
        if ext_id in self._ext_to_node:
            # Overwrite: tombstone the old node, add a fresh one.
            self._apply_delete(ext_id)
        node = self.index.add(vec)
        self._ext_to_node[ext_id] = node
        self._node_to_ext[node] = ext_id
        self._metadata[node] = dict(meta)

    # ------------------------------------------------------------------ delete
    def delete(self, ext_id: str) -> bool:
        if ext_id not in self._ext_to_node:
            return False
        if self._wal is not None:
            self._wal.log_delete(ext_id)
        self._apply_delete(ext_id)
        return True

    def _apply_delete(self, ext_id: str) -> None:
        node = self._ext_to_node.pop(ext_id, None)
        if node is None:
            return
        self.index.mark_deleted(node)
        self._node_to_ext.pop(node, None)
        self._metadata.pop(node, None)

    # ------------------------------------------------------------------ query
    def query(
        self,
        vector,
        k: int = 10,
        flt: Filter = None,
        ef_search: int | None = None,
        filter_mode: str = "prefilter",
        overfetch: int = 8,
    ) -> list[QueryResult]:
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if flt is None:
            hits = self.index.search(vector, k, ef_search=ef_search)
            return self._to_results(hits)

        if filter_mode == "prefilter":
            return self._prefiltered(vector, k, flt)
        elif filter_mode == "postfilter":
            raw = self.index.search(vector, k * overfetch, ef_search=ef_search)
            filtered = [
                (n, d) for n, d in raw if _matches(self._metadata.get(n, {}), flt)
            ]
            return self._to_results(filtered[:k])
        raise ValueError(f"unknown filter_mode {filter_mode!r}")

    def _prefiltered(self, vector: np.ndarray, k: int, flt: Filter) -> list[QueryResult]:
        allowed = [n for n, m in self._metadata.items() if _matches(m, flt)]
        if not allowed:
            return []
        q = prepare(vector, self.config.metric)
        rows = np.stack([self.index.get_vector(n) for n in allowed])
        dists = distance_batch(q, rows, self.config.metric)
        order = np.argsort(dists)[:k]
        hits = [(allowed[i], float(dists[i])) for i in order]
        return self._to_results(hits)

    def _to_results(self, hits: list[tuple[int, float]]) -> list[QueryResult]:
        out: list[QueryResult] = []
        for node, dist in hits:
            ext = self._node_to_ext.get(node)
            if ext is None:
                continue
            out.append(QueryResult(id=ext, distance=dist, metadata=self._metadata.get(node, {})))
        return out

    # ---------------------------------------------------------------- lifecycle
    def compact(self) -> None:
        """Rebuild the graph without tombstones (see HNSW.rebuild)."""
        self.index = self.index.rebuild()

    def __len__(self) -> int:
        return len(self._ext_to_node)

    def stats(self) -> dict:
        return {
            "name": self.name,
            "count": len(self._ext_to_node),
            "nodes": self.index.num_nodes,
            "tombstones": len(self.index._deleted),
            "dim": self.config.dim,
            "metric": self.config.metric.value,
            "levels": self.index.level_histogram(),
        }

    # ------------------------------------------------------------------- disk
    # Files in a collection directory:
    #   config.json    written once at creation (durable). Holds dim/metric/M/...
    #   graph.bin      snapshot of the HNSW graph + vectors (written by save()).
    #   manifest.json  snapshot of the id maps + metadata      (written by save()).
    #   wal.log        append-only log of inserts/deletes since the last save().
    # config.json is separate from the snapshot so a crash with only a WAL (no
    # snapshot yet) can still be recovered: we know how to build the index.
    @staticmethod
    def _write_json_durable(path: str, obj: dict) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _config_dict(self) -> dict:
        return {
            "name": self.name,
            "dim": self.config.dim,
            "metric": self.config.metric.value,
            "M": self.config.M,
            "M0": self.config.M0,
            "ef_construction": self.config.ef_construction,
            "ef_search": self.config.ef_search,
            "seed": self.config.seed,
        }

    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self._write_json_durable(os.path.join(directory, "config.json"), self._config_dict())
        save_hnsw(self.index, os.path.join(directory, "graph.bin"))
        manifest = {
            "ext_to_node": self._ext_to_node,
            "metadata": {str(n): m for n, m in self._metadata.items()},
        }
        self._write_json_durable(os.path.join(directory, "manifest.json"), manifest)
        # Snapshot now contains everything: safe to clear the WAL.
        if self._wal is not None:
            self._wal.truncate()

    @staticmethod
    def _config_from_dict(c: dict) -> HNSWConfig:
        return HNSWConfig(
            dim=c["dim"],
            metric=Metric(c["metric"]),
            M=c["M"],
            M0=c["M0"],
            ef_construction=c["ef_construction"],
            ef_search=c["ef_search"],
            seed=c["seed"],
        )

    @classmethod
    def load(cls, directory: str, enable_wal: bool = True) -> "Collection":
        """Load a collection from disk: config + optional snapshot + WAL replay.

        Works whether or not a snapshot exists. With only config.json + wal.log
        (the state right after a crash) it rebuilds purely from the WAL.
        """
        with open(os.path.join(directory, "config.json")) as f:
            cfg_dict = json.load(f)
        config = cls._config_from_dict(cfg_dict)
        coll = cls(cfg_dict.get("name", os.path.basename(os.path.normpath(directory))), config)

        graph_path = os.path.join(directory, "graph.bin")
        manifest_path = os.path.join(directory, "manifest.json")
        if os.path.exists(graph_path) and os.path.exists(manifest_path):
            coll.index = load_hnsw(graph_path)
            with open(manifest_path) as f:
                manifest = json.load(f)
            coll._ext_to_node = {k: int(v) for k, v in manifest["ext_to_node"].items()}
            coll._node_to_ext = {v: k for k, v in coll._ext_to_node.items()}
            coll._metadata = {int(n): m for n, m in manifest["metadata"].items()}

        wal_path = os.path.join(directory, "wal.log")
        coll._replay_wal(wal_path)
        if enable_wal:
            coll._attach_wal(wal_path)
        return coll

    def _replay_wal(self, wal_path: str) -> None:
        for op, ext_id, meta, vec in WriteAheadLog.replay(wal_path, self.config.dim):
            if op == OP_INSERT:
                self._apply_insert(ext_id, meta, vec)
            elif op == OP_DELETE:
                self._apply_delete(ext_id)

    @classmethod
    def open(cls, directory: str, config: HNSWConfig | None = None) -> "Collection":
        """Open an existing collection dir, or create a new one if absent.

        A collection is considered to exist once config.json is present. Creating
        a new one requires ``config`` and writes config.json durably up front, so
        even a crash before the first save() is recoverable. The WAL is enabled.
        """
        os.makedirs(directory, exist_ok=True)
        if os.path.exists(os.path.join(directory, "config.json")):
            return cls.load(directory, enable_wal=True)
        if config is None:
            raise ValueError("config required to create a new collection")
        name = os.path.basename(os.path.normpath(directory))
        coll = cls(name, config)
        coll._write_json_durable(os.path.join(directory, "config.json"), coll._config_dict())
        wal_path = os.path.join(directory, "wal.log")
        coll._replay_wal(wal_path)  # recover an orphaned WAL if present
        coll._attach_wal(wal_path)
        return coll

    def close(self) -> None:
        if self._wal is not None:
            self._wal.close()
            self._wal = None
