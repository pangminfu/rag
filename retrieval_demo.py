import os

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings


PERSIST_DIR = "./chroma_db"

QUERIES = [
    "What's the warranty on the R-200?",
    "How do I bake bread?",
]


def main():
    embedder = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=os.environ["OLLAMA_HOST"],
    )
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embedder,
    )

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
