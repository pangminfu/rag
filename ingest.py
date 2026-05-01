import argparse

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma

from chunking import CHUNKERS, Chunker, RecursiveChunker, SemanticChunker
from model_provider import ModelProvider


DATA_DIR = "data"


def build_chunker(strategy: str, embedder) -> Chunker:
    if strategy == RecursiveChunker.name:
        return RecursiveChunker(chunk_size=500, chunk_overlap=50)
    if strategy == SemanticChunker.name:
        return SemanticChunker(embedder, breakpoint_threshold_type="percentile")
    raise ValueError(f"unknown strategy: {strategy}")


def main():
    parser = argparse.ArgumentParser(description="Chunk + embed corpus into Chroma.")
    parser.add_argument(
        "--strategy",
        choices=list(CHUNKERS),
        default=RecursiveChunker.name,
        help="Chunking strategy to use (default: recursive).",
    )
    args = parser.parse_args()

    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents from {DATA_DIR}/")

    embedder = ModelProvider().embeddings()

    chunker = build_chunker(args.strategy, embedder)
    chunks = chunker.split(docs)
    print(f"[{chunker.name}] Split into {len(chunks)} chunks.")

    Chroma.from_documents(
        documents=chunks,
        embedding=embedder,
        persist_directory=chunker.persist_dir,
    )

    print(f"[{chunker.name}] Indexed {len(chunks)} chunks into {chunker.persist_dir}.")


if __name__ == "__main__":
    main()
