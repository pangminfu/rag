import os

import numpy as np
from langchain_ollama import OllamaEmbeddings


SENTENCES = [
    "How many days of annual leave do I get?",
    "What is the vacation policy?",
    "How do I make pasta?",
]


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    base_url = os.environ["OLLAMA_HOST"]
    embedder = OllamaEmbeddings(model="nomic-embed-text", base_url=base_url)
    vectors = embedder.embed_documents(SENTENCES)

    print(f"Embedding dimension: {len(vectors[0])}\n")

    print("Pairwise cosine similarity:")
    for i in range(len(SENTENCES)):
        for j in range(i + 1, len(SENTENCES)):
            sim = cosine(vectors[i], vectors[j])
            print(f"  [{i}] vs [{j}]  sim={sim:.4f}")
            print(f"        {SENTENCES[i]!r}")
            print(f"        {SENTENCES[j]!r}")


if __name__ == "__main__":
    main()
