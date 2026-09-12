import numpy as np
import pytest


@pytest.fixture
def dataset():
    rng = np.random.default_rng(42)

    # Small deterministic dataset for fast correctness tests.
    data = rng.normal(size=(1000, 32)).astype(np.float32)
    queries = rng.normal(size=(50, 32)).astype(np.float32)

    return data, queries