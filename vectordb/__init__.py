"""vectordb: a from-scratch vector database engine with an HNSW index."""

from .collection import Collection, QueryResult
from .config import HNSWConfig
from .distance import Metric
from .flat import FlatIndex
from .hnsw import HNSW

__all__ = [
    "Collection",
    "QueryResult",
    "HNSWConfig",
    "Metric",
    "FlatIndex",
    "HNSW",
]

__version__ = "0.1.0"
