from langchain_core.documents import Document
from sentence_transformers import CrossEncoder


RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, chunks: list[Document], k: int) -> list[Document]:
    if not chunks:
        return []
    pairs = [(query, c.page_content) for c in chunks]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked[:k]]
