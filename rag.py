import argparse

from agent import load_system_prompt
from chunking import CHUNKERS, RecursiveChunker
from model_provider import ModelProvider
from vector_store_provider import VectorStoreProvider


TOP_K = 4
DEFAULT_STORE = RecursiveChunker.name

PROMPT_TEMPLATE = """{system_prompt}
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}"""


def _build_components(store: str):
    provider = ModelProvider()
    collection = CHUNKERS[store].name
    vectorstore = VectorStoreProvider().load(provider.embeddings(), collection)
    llm = provider.chat()
    return vectorstore, llm


def answer(question: str, store: str = DEFAULT_STORE) -> str:
    vectorstore, llm = _build_components(store)

    chunks = vectorstore.similarity_search(question, k=TOP_K)
    context = "\n\n---\n\n".join(
        f"[source: {c.metadata.get('source', '?')}]\n{c.page_content}" for c in chunks
    )

    prompt = PROMPT_TEMPLATE.format(
        system_prompt=load_system_prompt(),
        context=context,
        question=question,
    )
    response = llm.invoke(prompt)
    return response.content


def main():
    parser = argparse.ArgumentParser(description="Naive always-retrieve RAG.")
    parser.add_argument(
        "--store",
        choices=list(CHUNKERS),
        default=DEFAULT_STORE,
        help=f"Which collection to query (default: {DEFAULT_STORE}).",
    )
    parser.add_argument("question", nargs="+", help="The question to answer.")
    args = parser.parse_args()

    question = " ".join(args.question)
    print(answer(question, store=args.store))


if __name__ == "__main__":
    main()
