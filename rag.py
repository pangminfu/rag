import sys

from langchain_community.vectorstores import Chroma

from agent import load_system_prompt
from model_provider import ModelProvider


PERSIST_DIR = "./chroma_db"
TOP_K = 4

PROMPT_TEMPLATE = """{system_prompt}
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}"""


def _build_components():
    provider = ModelProvider()
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=provider.embeddings())
    llm = provider.chat()
    return vectorstore, llm


def answer(question: str) -> str:
    vectorstore, llm = _build_components()

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
    if len(sys.argv) < 2:
        print('Usage: uv run rag.py "<your question>"', file=sys.stderr)
        sys.exit(1)
    question = " ".join(sys.argv[1:])
    print(answer(question))


if __name__ == "__main__":
    main()
