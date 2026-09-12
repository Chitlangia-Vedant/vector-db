from sentence_transformers import SentenceTransformer

from vectordb import Collection, HNSWConfig, Metric


DOCUMENTS = [
    "The government announced new economic reforms to support small businesses.",
    "The stock market rose after investors responded positively to the latest earnings reports.",
    "Scientists discovered a new method for improving battery storage capacity.",
    "The football team won the championship after a dramatic final match.",
    "Researchers developed an artificial intelligence system for detecting diseases.",
    "Heavy rainfall caused flooding in several cities across the region.",
    "The company released a new smartphone with improved battery life and camera quality.",
    "The spacecraft successfully entered orbit after completing its journey.",
    "The cricket team defeated its opponent in the final match of the tournament.",
    "Engineers designed a faster processor for next-generation computers.",
    "The central bank announced a change in interest rates to control inflation.",
    "A new study found that regular exercise can improve cardiovascular health.",
    "The software company announced a major update to its cloud computing platform.",
    "The basketball team secured a place in the playoffs after winning its final game.",
    "Scientists are studying climate change and its effects on global temperatures.",
]


def main():
    print("Loading embedding model...")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    print("Creating embeddings...")
    embeddings = model.encode(
        DOCUMENTS,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    config = HNSWConfig(
        dim=embeddings.shape[1],
        metric=Metric.COSINE,
        M=16,
        ef_construction=200,
        ef_search=40,
    )

    coll = Collection.open("demo_collection", config)

    ids = [f"doc{i}" for i in range(len(DOCUMENTS))]
    metadata = [{"text": text} for text in DOCUMENTS]

    coll.insert(embeddings, ids, metadata)

    print(f"\nIndexed {len(DOCUMENTS)} statements.")
    print("Type a statement to find the closest matches.")
    print("Type 'exit' to quit.\n")

    while True:
        text = input("Query: ").strip()

        if text.lower() in {"exit", "quit", "q"}:
            break

        if not text:
            continue

        query_vector = model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        results = coll.query(query_vector, k=3)

        print("\nBest matches:")

        for rank, result in enumerate(results, start=1):
            print(f"{rank}. {result.metadata['text']}")
            print(f"   distance: {result.distance:.4f}")

        print()

    coll.close()
    print("Demo finished.")


if __name__ == "__main__":
    main()