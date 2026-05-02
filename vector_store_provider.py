"""Vector store backend abstraction.

Reads backend choice from env and exposes uniform write/load/dump operations
over the chosen vector index. Mirrors model_provider.py — add a new backend
by writing a `_VectorStore` subclass and registering it in `_VECTOR_STORES`.

Required env vars:
    VECTOR_STORE   chroma | qdrant

Per-backend config (only the one you actually use):
    Chroma:  no extra vars — collections persist under ./vectorstore/chroma/<name>
    Qdrant:  QDRANT_URL, optional QDRANT_API_KEY
"""

import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


def _env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


class _VectorStore:
    def from_documents(
        self, docs: list[Document], embedder: Embeddings, collection: str
    ) -> None:
        raise NotImplementedError

    def load(self, embedder: Embeddings, collection: str) -> VectorStore:
        raise NotImplementedError

    def dump_documents(self, store: VectorStore) -> list[Document]:
        raise NotImplementedError


class _ChromaVectorStore(_VectorStore):
    ROOT = Path("./vectorstore/chroma")

    def from_documents(self, docs, embedder, collection):
        from langchain_community.vectorstores import Chroma
        persist_dir = self.ROOT / collection
        persist_dir.mkdir(parents=True, exist_ok=True)
        Chroma.from_documents(
            documents=docs,
            embedding=embedder,
            persist_directory=str(persist_dir),
        )

    def load(self, embedder, collection):
        from langchain_community.vectorstores import Chroma
        return Chroma(
            persist_directory=str(self.ROOT / collection),
            embedding_function=embedder,
        )

    def dump_documents(self, store):
        raw = store.get(include=["documents", "metadatas"])
        return [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(raw["documents"], raw["metadatas"])
        ]


class _QdrantVectorStore(_VectorStore):
    def from_documents(self, docs, embedder, collection):
        from langchain_qdrant import QdrantVectorStore
        QdrantVectorStore.from_documents(
            docs,
            embedder,
            url=_env("QDRANT_URL"),
            api_key=os.environ.get("QDRANT_API_KEY"),
            collection_name=collection,
        )

    def load(self, embedder, collection):
        from langchain_qdrant import QdrantVectorStore
        return QdrantVectorStore.from_existing_collection(
            embedding=embedder,
            url=_env("QDRANT_URL"),
            api_key=os.environ.get("QDRANT_API_KEY"),
            collection_name=collection,
        )

    def dump_documents(self, store):
        docs: list[Document] = []
        offset = None
        while True:
            points, offset = store.client.scroll(
                collection_name=store.collection_name,
                limit=256,
                with_payload=True,
                with_vectors=False,
                offset=offset,
            )
            for p in points:
                payload = p.payload or {}
                docs.append(Document(
                    page_content=payload.get("page_content", ""),
                    metadata=payload.get("metadata", {}),
                ))
            if offset is None:
                break
        return docs


_VECTOR_STORES: dict[str, type[_VectorStore]] = {
    "chroma": _ChromaVectorStore,
    "qdrant": _QdrantVectorStore,
}


class VectorStoreProvider:
    def __init__(self):
        name = _env("VECTOR_STORE").lower()
        try:
            self._vector_store = _VECTOR_STORES[name]()
        except KeyError:
            raise ValueError(
                f"Unknown VECTOR_STORE: {name!r}. "
                f"Expected {' | '.join(_VECTOR_STORES)}."
            )

    def from_documents(
        self,
        docs: list[Document],
        embedder: Embeddings,
        collection: str,
    ) -> None:
        """Ingest chunks into the given collection."""
        self._vector_store.from_documents(docs, embedder, collection)

    def load(self, embedder: Embeddings, collection: str) -> VectorStore:
        """Open an existing collection for retrieval."""
        return self._vector_store.load(embedder, collection)

    def dump_documents(self, store: VectorStore) -> list[Document]:
        """Return every chunk in the collection.

        Used by the agent's BM25 arm so the lexical retriever sees the same
        corpus as the dense one. Cheap for local stores, expensive for
        cloud-hosted ones — if you add one, prefer persisting BM25 at ingest
        time instead of relying on this.
        """
        return self._vector_store.dump_documents(store)
