import argparse

from chunking import CHUNKERS, RecursiveChunker
from model_provider import ModelProvider
from vector_store_provider import VectorStoreProvider


# Sample queries tied to the Acme demo corpus (samples/acme/) — the first
# should retrieve product-warranty chunks, the second is out-of-corpus and
# should rank low. Swap these for queries from your own corpus.
QUERIES = [
    "What's the warranty on the R-200?",
    "How do I bake bread?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--store",
        choices=list(CHUNKERS),
        default=RecursiveChunker.name,
        help="Which collection to query (default: recursive).",
    )
    args = parser.parse_args()

    collection = CHUNKERS[args.store].name
    embedder = ModelProvider().embeddings()
    vectorstore = VectorStoreProvider().load(embedder, collection)

    print(f"=== Store: {args.store} (collection: {collection}) ===")

    for query in QUERIES:
        print(f"\n=== Query: {query!r} ===")
        results = vectorstore.similarity_search_with_score(query, k=3)
        for rank, (doc, score) in enumerate(results, start=1):
            source = doc.metadata.get("source", "?")
            preview = doc.page_content.replace("\n", " ")[:200]
            print(f"\n[{rank}] score={score:.4f}  source={source}")
            print(f"    {preview}...")


if __name__ == "__main__":
    main()
