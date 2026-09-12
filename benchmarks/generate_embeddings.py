"""Generate local text embeddings for the vector database benchmark.

Corpus:
    AG News dataset.

Data:
    50,500 short real-world news texts.

Embeddings:
    sentence-transformers/all-mpnet-base-v2
    768-dimensional embeddings.

Output:
    benchmarks/data/embeddings.npz


Usage:
    python benchmarks/generate_embeddings.py --limit 50500

Required packages:
    pip install sentence-transformers
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import urllib.request

import numpy as np


MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

TRAIN_URL = (
    "https://raw.githubusercontent.com/"
    "mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
)

TEST_URL = (
    "https://raw.githubusercontent.com/"
    "mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
)

MAX_CHARS = 1000


def download_file(url: str, output_path: str) -> None:
    """Download a file if it does not already exist."""

    if os.path.exists(output_path):
        print(f"already exists: {output_path}")
        return

    print(f"downloading: {url}")

    try:
        urllib.request.urlretrieve(url, output_path)
    except Exception:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise

    print(f"saved: {output_path}")


def read_csv_texts(path: str) -> tuple[list[str], list[str]]:
    """Read AG News CSV and return texts and labels.

    Each row has:
        class, title, description
    """

    texts = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            if len(row) < 3:
                continue

            try:
                label = int(row[0])
            except ValueError:
                continue

            title = row[1].strip()
            description = row[2].strip()

            text = f"{title}. {description}"
            text = text.replace("\n", " ").replace("\r", " ")

            if len(text) < 20:
                continue

            text = text[:MAX_CHARS]

            texts.append(text)
            labels.append(str(label))

    return texts, labels


def load_corpus(
    data_dir: str,
    limit: int,
) -> tuple[list[str], list[str]]:
    """Download and load enough AG News texts for the benchmark."""

    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    download_file(TRAIN_URL, train_path)
    download_file(TEST_URL, test_path)

    print("reading AG News CSV files ...")

    train_texts, train_labels = read_csv_texts(train_path)

    print(f"train rows: {len(train_texts)}")

    if len(train_texts) >= limit:
        return (
            train_texts[:limit],
            train_labels[:limit],
        )

    remaining = limit - len(train_texts)

    test_texts, test_labels = read_csv_texts(test_path)

    print(f"test rows: {len(test_texts)}")

    if len(test_texts) < remaining:
        raise RuntimeError(
            f"Not enough AG News rows for limit={limit}. "
            f"Found only {len(train_texts) + len(test_texts)} rows."
        )

    texts = train_texts + test_texts[:remaining]
    labels = train_labels + test_labels[:remaining]

    return texts, labels


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--limit",
        type=int,
        default=50500,
        help="Number of texts to embed.",
    )

    ap.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__),
            "data",
            "embeddings.npz",
        ),
        help="Output .npz file.",
    )

    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size.",
    )

    ap.add_argument(
        "--max-chars",
        type=int,
        default=1000,
        help="Maximum characters kept from each text.",
    )

    args = ap.parse_args()

    global MAX_CHARS
    MAX_CHARS = args.max_chars

    data_dir = os.path.join(
        os.path.dirname(__file__),
        "data",
        "ag_news",
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    print("=" * 60)
    print("AG News embedding generator")
    print("=" * 60)

    print(f"target texts : {args.limit}")
    print(f"model        : {MODEL_NAME}")
    print(f"output       : {args.out}")
    print()

    # ---------------------------------------------------------
    # 1. Download and load corpus
    # ---------------------------------------------------------

    print("loading corpus ...")

    texts, categories = load_corpus(
        data_dir=data_dir,
        limit=args.limit,
    )

    print(f"loaded {len(texts)} texts")

    if len(texts) != args.limit:
        raise RuntimeError(
            f"Expected {args.limit} texts, got {len(texts)}"
        )

    # ---------------------------------------------------------
    # 2. Load embedding model
    # ---------------------------------------------------------

    print()
    print(f"loading embedding model: {MODEL_NAME}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(MODEL_NAME)

    # ---------------------------------------------------------
    # 3. Generate embeddings
    # ---------------------------------------------------------

    print()
    print("generating embeddings ...")

    start = time.time()

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    embeddings = embeddings.astype(np.float32)

    elapsed = time.time() - start

    # ---------------------------------------------------------
    # 4. Validate dimensions
    # ---------------------------------------------------------

    expected_shape = (args.limit, 768)

    print()
    print(f"embedding shape: {embeddings.shape}")

    if embeddings.shape != expected_shape:
        raise RuntimeError(
            f"Expected embeddings with shape {expected_shape}, "
            f"got {embeddings.shape}"
        )

    # ---------------------------------------------------------
    # 5. Save
    # ---------------------------------------------------------

    categories = np.asarray(categories)

    np.savez_compressed(
        args.out,
        embeddings=embeddings,
        categories=categories,
    )

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"texts       : {len(texts)}")
    print(f"dimensions  : {embeddings.shape[1]}")
    print(f"embeddings  : {embeddings.shape}")
    print(f"generation  : {elapsed:.1f} seconds")
    print(f"saved to    : {args.out}")
    print("=" * 60)


if __name__ == "__main__":
    main()