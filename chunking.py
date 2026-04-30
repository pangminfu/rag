from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker as _LCSemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker(ABC):
    """Pluggable chunking strategy. Each strategy persists to its own Chroma dir
    so different strategies can be compared side-by-side without re-indexing."""

    name: str
    persist_dir: str

    @abstractmethod
    def split(self, docs: list[Document]) -> list[Document]: ...


class RecursiveChunker(Chunker):
    name = "recursive"
    persist_dir = "./chroma_db"

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def split(self, docs: list[Document]) -> list[Document]:
        return self._splitter.split_documents(docs)


class SemanticChunker(Chunker):
    name = "semantic"
    persist_dir = "./chroma_db_semantic"

    def __init__(
        self,
        embedder: Embeddings,
        breakpoint_threshold_type: str = "percentile",
    ):
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self._splitter = _LCSemanticChunker(
            embedder,
            breakpoint_threshold_type=breakpoint_threshold_type,
        )

    def split(self, docs: list[Document]) -> list[Document]:
        return self._splitter.split_documents(docs)


CHUNKERS: dict[str, type[Chunker]] = {
    RecursiveChunker.name: RecursiveChunker,
    SemanticChunker.name: SemanticChunker,
}
