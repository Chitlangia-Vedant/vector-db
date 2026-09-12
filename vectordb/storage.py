"""On-disk format for the HNSW graph and the write-ahead log (WAL).

Two things live here:

1. ``save_hnsw`` / ``load_hnsw``: a compact, documented binary snapshot of a
   whole HNSW index (graph + vectors + tombstones).

2. ``WriteAheadLog``: an append-only log that makes individual inserts and
   deletes durable *before* they touch the in-memory graph, so a crash in the
   middle of a batch loses nothing that was acknowledged.

Snapshot binary layout (all little-endian)
------------------------------------------
    magic   : 8 bytes  b"VDBHNSW1"
    dim      : uint32
    metric   : uint32   (0=l2, 1=cosine, 2=ip)
    M        : uint32
    M0       : uint32
    ef_con   : uint32
    size     : uint32   (number of node slots)
    entry_pt : int32    (-1 if empty)
    max_level: int32
    per node (size of them):
        level : uint16
        for layer in 0..level:
            deg   : uint32
            deg * uint32 neighbor ids
    deleted  : uint32 count, then count * uint32 ids
    vectors  : size * dim float32, row-major

WAL record layout
------------------
    magic : 4 bytes b"Wrec"
    length: uint32          (length of payload)
    payload: length bytes    (op byte + json header + optional vector)
    crc32 : uint32           (over payload)
A record is only valid if the magic, full payload and crc all read back. A torn
tail (partial write from a crash) fails one of those checks and is discarded,
which is exactly the desired recovery behavior.
"""

from __future__ import annotations

import json
import os
import struct
import zlib
from typing import BinaryIO

import numpy as np

from .config import HNSWConfig
from .distance import Metric
from .hnsw import HNSW

_MAGIC = b"VDBHNSW1"
_METRIC_CODE = {Metric.L2: 0, Metric.COSINE: 1, Metric.IP: 2}
_CODE_METRIC = {v: k for k, v in _METRIC_CODE.items()}


def save_hnsw(index: HNSW, path: str) -> None:
    """Write ``index`` to ``path`` atomically (write to tmp, then rename)."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        _write_hnsw(index, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_hnsw(index: HNSW, f: BinaryIO) -> None:
    cfg = index.config
    f.write(_MAGIC)
    f.write(
        struct.pack(
            "<IIIIIiii",
            cfg.dim,
            _METRIC_CODE[cfg.metric],
            cfg.M,
            cfg.M0,
            cfg.ef_construction,
            index.num_nodes,
            index._entry_point if index._entry_point is not None else -1,
            index._max_level,
        )
    )
    for node in range(index.num_nodes):
        level = index._levels[node]
        f.write(struct.pack("<H", level))
        links = index._links[node]
        for lc in range(level + 1):
            neigh = links[lc] if lc < len(links) else []
            f.write(struct.pack("<I", len(neigh)))
            if neigh:
                f.write(np.asarray(neigh, dtype="<u4").tobytes())
    deleted = sorted(index._deleted)
    f.write(struct.pack("<I", len(deleted)))
    if deleted:
        f.write(np.asarray(deleted, dtype="<u4").tobytes())
    f.write(index._matrix[: index.num_nodes].astype("<f4").tobytes())


def load_hnsw(path: str) -> HNSW:
    with open(path, "rb") as f:
        return _read_hnsw(f)


def _read_hnsw(f: BinaryIO) -> HNSW:
    magic = f.read(8)
    if magic != _MAGIC:
        raise ValueError("bad snapshot magic")
    dim, metric_code, M, M0, ef_con, size, entry, max_level = struct.unpack(
        "<IIIIIiii", f.read(32)
    )
    cfg = HNSWConfig(
        dim=dim,
        metric=_CODE_METRIC[metric_code],
        M=M,
        M0=M0,
        ef_construction=ef_con,
    )
    index = HNSW(cfg)
    levels: list[int] = []
    links: list[list[list[int]]] = []
    for _ in range(size):
        (level,) = struct.unpack("<H", f.read(2))
        levels.append(level)
        node_links: list[list[int]] = []
        for _lc in range(level + 1):
            (deg,) = struct.unpack("<I", f.read(4))
            if deg:
                ids = np.frombuffer(f.read(deg * 4), dtype="<u4").tolist()
            else:
                ids = []
            node_links.append(ids)
        links.append(node_links)
    (ndel,) = struct.unpack("<I", f.read(4))
    deleted: set[int] = set()
    if ndel:
        deleted = set(np.frombuffer(f.read(ndel * 4), dtype="<u4").tolist())
    matrix = np.frombuffer(f.read(size * dim * 4), dtype="<f4").reshape(size, dim).copy()

    index._levels = levels
    index._links = links
    index._deleted = deleted
    index._entry_point = None if entry < 0 else entry
    index._max_level = max_level
    index._size = size
    index._capacity = size
    index._matrix = matrix
    return index


# --------------------------------------------------------------------------- WAL
_WAL_MAGIC = b"Wrec"
OP_INSERT = 1
OP_DELETE = 2


class WriteAheadLog:
    """Append-only durability log. One log file per collection."""

    def __init__(self, path: str):
        self.path = path
        self._file: BinaryIO | None = None

    def open(self) -> None:
        self._file = open(self.path, "ab")

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def _append(self, payload: bytes) -> None:
        assert self._file is not None, "WAL not open"
        frame = _WAL_MAGIC + struct.pack("<I", len(payload)) + payload
        frame += struct.pack("<I", zlib.crc32(payload) & 0xFFFFFFFF)
        self._file.write(frame)
        self._file.flush()
        os.fsync(self._file.fileno())

    def log_insert(self, external_id: str, metadata: dict, vector: np.ndarray) -> None:
        header = json.dumps({"id": external_id, "meta": metadata}).encode("utf-8")
        vec = np.asarray(vector, dtype="<f4").tobytes()
        payload = bytes([OP_INSERT]) + struct.pack("<I", len(header)) + header + vec
        self._append(payload)

    def log_delete(self, external_id: str) -> None:
        header = json.dumps({"id": external_id}).encode("utf-8")
        payload = bytes([OP_DELETE]) + struct.pack("<I", len(header)) + header
        self._append(payload)

    def truncate(self) -> None:
        """Empty the log. Called after a snapshot folds the records into it."""
        self.close()
        open(self.path, "wb").close()
        self.open()

    @staticmethod
    def replay(path: str, dim: int):
        """Yield (op, external_id, metadata, vector) for every intact record.

        Stops at the first torn/corrupt record (a crash mid-write), which is the
        correct recovery boundary.
        """
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            data = f.read()
        pos = 0
        n = len(data)
        while pos + 8 <= n:
            if data[pos : pos + 4] != _WAL_MAGIC:
                break
            (length,) = struct.unpack("<I", data[pos + 4 : pos + 8])
            start = pos + 8
            end = start + length
            if end + 4 > n:
                break  # truncated payload
            payload = data[start:end]
            (crc,) = struct.unpack("<I", data[end : end + 4])
            if (zlib.crc32(payload) & 0xFFFFFFFF) != crc:
                break  # corrupt tail
            op = payload[0]
            (hlen,) = struct.unpack("<I", payload[1:5])
            header = json.loads(payload[5 : 5 + hlen].decode("utf-8"))
            if op == OP_INSERT:
                vec = np.frombuffer(payload[5 + hlen :], dtype="<f4").copy()
                yield OP_INSERT, header["id"], header.get("meta", {}), vec
            elif op == OP_DELETE:
                yield OP_DELETE, header["id"], None, None
            pos = end + 4
