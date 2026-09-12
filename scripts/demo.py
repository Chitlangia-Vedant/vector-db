"""End-to-end demo: build a collection, query it, filter, delete, persist.

Run: python scripts/demo.py
"""

import tempfile

import numpy as np

from vectordb import Collection, HNSWConfig, Metric


def main() -> None:
    rng = np.random.default_rng(0)
    dim = 64
    data = rng.standard_normal((2000, dim)).astype(np.float32)
    cats = ["news", "sports", "tech"]

    with tempfile.TemporaryDirectory() as d:
        coll = Collection.open(d, HNSWConfig(dim=dim, metric=Metric.COSINE))
        ids = [f"doc{i}" for i in range(len(data))]
        metas = [{"cat": cats[i % 3]} for i in range(len(data))]
        coll.insert(data, ids, metas)

        assert len(coll) == 2000
        print("inserted", len(coll), "vectors")

        print("stats:", coll.stats())

        q = data[42]

        results = coll.query(q, k=5)
        assert results
        print("\nnearest to doc42:")
        for r in results:
            print(f"  {r.id:8s} dist={r.distance:.4f} {r.metadata}")

        filtered = coll.query(q, k=5, flt={"cat": "tech"})
        assert all(r.metadata["cat"] == "tech" for r in filtered)

        print("\nnearest to doc42 filtered to cat=tech:")
        for r in filtered:
            print(f"  {r.id:8s} dist={r.distance:.4f} {r.metadata}")

        coll.delete("doc42")

        after_delete = coll.query(q, k=5)
        assert all(r.id != "doc42" for r in after_delete)

        print(f"\nafter deleting doc42, nearest is now {after_delete[0].id}")

        coll.save(d)
        coll.close()

        reopened = Collection.load(d)

        assert len(reopened) == 1999
        assert all(r.id != "doc42" for r in reopened.query(q, k=5))

        print("reloaded from disk:", len(reopened), "vectors")
        reopened.close()


if __name__ == "__main__":
    main()
