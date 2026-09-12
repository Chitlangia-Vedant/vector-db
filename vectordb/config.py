"""Configuration dataclasses for the index types."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .distance import Metric


@dataclass
class HNSWConfig:
    """Tunables for the HNSW index.

    dim             : vector dimensionality.
    metric          : distance metric (see distance.Metric).
    M               : number of bidirectional links created per node on insert
                      at layers above 0. Layer 0 uses ``M0`` (defaults to 2*M).
    M0              : max links at layer 0. None -> 2*M (paper's recommendation).
    ef_construction : size of the dynamic candidate list while building. Larger
                      means a better graph but slower inserts.
    ef_search       : default candidate-list size at query time. Larger means
                      higher recall but slower queries. Can be overridden per
                      query.
    mL              : level-generation normalization factor. The paper recommends
                      1/ln(M); None computes that.
    seed            : RNG seed for the probabilistic level assignment.
    extend_candidates : heuristic option (Algorithm 4). Rarely needed; off by
                      default, matching common implementations.
    keep_pruned_connections : heuristic option (Algorithm 4). When the diversity
                      filter selects fewer than M neighbors, backfill from the
                      pruned set so nodes stay well-connected. On by default.
    """

    dim: int
    metric: Metric = Metric.L2
    M: int = 16
    M0: int | None = None
    ef_construction: int = 200
    ef_search: int = 50
    mL: float | None = None
    seed: int = 42
    extend_candidates: bool = False
    keep_pruned_connections: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.metric, str):
            self.metric = Metric(self.metric)
        if self.M0 is None:
            self.M0 = self.M * 2
        if self.mL is None:
            self.mL = 1.0 / math.log(self.M)
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.M < 2:
            raise ValueError("M must be >= 2")
