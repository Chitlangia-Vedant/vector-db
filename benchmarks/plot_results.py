from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt


HERE = os.path.dirname(__file__)

RESULTS_FILE = os.path.join(
    HERE,
    "results",
    "results.json",
)

OUTPUT_DIR = os.path.join(
    HERE,
    "results",
)


def load_results():
    with open(RESULTS_FILE, "r") as f:
        return json.load(f)


def get_ef_results(results):
    """Get HNSW efSearch sweep results."""

    rows = results["ours"]["sweep"]

    return sorted(
        rows,
        key=lambda row: row["ef_search"],
    )


def plot_recall_vs_qps(results):
    """Plot Recall@10 against QPS."""

    rows = get_ef_results(results)

    ef_values = [
        row["ef_search"]
        for row in rows
    ]

    recall = [
        row["recall@10"] * 100
        for row in rows
    ]

    qps = [
        row["qps"]
        for row in rows
    ]

    plt.figure(figsize=(8, 6))

    plt.plot(
        qps,
        recall,
        marker="o",
    )

    for x, y, ef in zip(
        qps,
        recall,
        ef_values,
    ):
        plt.annotate(
            f"ef={ef}",
            (x, y),
            xytext=(6, 6),
            textcoords="offset points",
        )

    plt.xlabel("Queries per second")
    plt.ylabel("Recall@10 (%)")
    plt.title("HNSW Recall@10 vs QPS")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    output = os.path.join(
        OUTPUT_DIR,
        "recall_vs_qps.png",
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"saved: {output}")


def plot_recall_vs_ef(results):
    """Plot Recall@10 against efSearch."""

    rows = get_ef_results(results)

    ef_values = [
        row["ef_search"]
        for row in rows
    ]

    recall = [
        row["recall@10"] * 100
        for row in rows
    ]

    plt.figure(figsize=(8, 6))

    plt.plot(
        ef_values,
        recall,
        marker="o",
    )

    for x, y in zip(
        ef_values,
        recall,
    ):
        plt.annotate(
            f"{y:.2f}%",
            (x, y),
            xytext=(6, 6),
            textcoords="offset points",
        )

    plt.xlabel("efSearch")
    plt.ylabel("Recall@10 (%)")
    plt.title("HNSW Recall@10 vs efSearch")

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    output = os.path.join(
        OUTPUT_DIR,
        "recall_vs_efsearch.png",
    )

    plt.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(f"saved: {output}")


def main():
    if not os.path.exists(RESULTS_FILE):
        raise SystemExit(
            f"Results file not found: {RESULTS_FILE}\n"
            "Run run_benchmark.py first."
        )

    results = load_results()

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    rows = get_ef_results(results)

    print("efSearch results:")

    for row in rows:
        print(
            f"  efSearch={row['ef_search']:>3} "
            f"recall={row['recall@10'] * 100:.2f}% "
            f"qps={row['qps']:.1f}"
        )

    print()

    plot_recall_vs_qps(results)
    plot_recall_vs_ef(results)


if __name__ == "__main__":
    main()