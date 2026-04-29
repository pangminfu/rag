import os
import sys

from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings


PERSIST_DIR = "./chroma_db"
LLM_MODEL = "gemma4:26b"
TOP_K = 4

PROMPT_TEMPLATE = """You are a helpful assistant for Acme Robotics employees.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}"""


def _build_components():
    base_url = os.environ["OLLAMA_HOST"]
    embedder = OllamaEmbeddings(model="nomic-embed-text", base_url=base_url)
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedder)
    llm = ChatOllama(model=LLM_MODEL, base_url=base_url)
    return vectorstore, llm


def answer(question: str) -> str:
    vectorstore, llm = _build_components()

    chunks = vectorstore.similarity_search(question, k=TOP_K)
    context = "\n\n---\n\n".join(
        f"[source: {c.metadata.get('source', '?')}]\n{c.page_content}" for c in chunks
    )

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
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
