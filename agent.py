import argparse
import os
import sqlite3
import sys
import time

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState, before_model
from langchain_classic.retrievers import EnsembleRetriever
from langchain.tools import tool
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, RemoveMessage, ToolMessage, trim_messages
from langchain_core.retrievers import BaseRetriever
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES

from chunking import CHUNKERS, RecursiveChunker
from rerank import rerank as _rerank


STATE_DB = "state.db"
LLM_MODEL = "gemma4:26b"
TOP_K = 4
FETCH_K = 20
TRIM_MAX_TOKENS = 2000
DEFAULT_STORE = RecursiveChunker.name
DEFAULT_HYBRID = True
DEFAULT_RERANK = True

SYSTEM_PROMPT = (
    "You are a helpful assistant for Acme Robotics employees. "
    "Use the search_knowledge_base tool when the question is about "
    "Acme Robotics — its HR policies, products, FAQs, or internal runbooks. "
    "For general knowledge or arithmetic, answer directly without the tool. "
    "If the tool returns nothing relevant, say you don't know."
)


def build_vectorstore(store: str = DEFAULT_STORE) -> Chroma:
    persist_dir = CHUNKERS[store].persist_dir
    embedder = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=os.environ["OLLAMA_HOST"],
    )
    return Chroma(persist_directory=persist_dir, embedding_function=embedder)


def build_retriever(
    store: str = DEFAULT_STORE,
    hybrid: bool = DEFAULT_HYBRID,
) -> BaseRetriever:
    """Build the retriever the agent searches with.

    BM25 is built over the same chunks that live in the chosen Chroma store, so
    the lexical arm stays consistent with whatever chunking strategy ingest used.
    """
    vectorstore = build_vectorstore(store)
    vector = vectorstore.as_retriever(search_kwargs={"k": FETCH_K})

    if not hybrid:
        return vector

    raw = vectorstore.get(include=["documents", "metadatas"])
    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = FETCH_K

    return EnsembleRetriever(retrievers=[bm25, vector], weights=[0.5, 0.5])


def retrieve(
    query: str,
    retriever: BaseRetriever,
    *,
    rerank: bool = DEFAULT_RERANK,
    k: int = TOP_K,
) -> list[Document]:
    """Fetch FETCH_K candidates, optionally cross-encoder rerank to k.

    The bi-encoder vector/BM25 retrievers cast a wide net (FETCH_K). When rerank
    is on, a cross-encoder scores (query, chunk) pairs jointly to pick the
    final k — far higher fidelity than the bi-encoder ranking alone.
    """
    t0 = time.perf_counter()
    candidates = retriever.invoke(query)
    t1 = time.perf_counter()
    if not rerank:
        return candidates[:k]
    chunks = _rerank(query, candidates, k=k)
    t2 = time.perf_counter()
    print(
        f"  [retrieve {(t1 - t0) * 1000:.0f}ms, "
        f"rerank {len(candidates)}→{len(chunks)} {(t2 - t1) * 1000:.0f}ms]",
        file=sys.stderr,
    )
    return chunks


def build_agent(
    store: str = DEFAULT_STORE,
    hybrid: bool = DEFAULT_HYBRID,
    rerank: bool = DEFAULT_RERANK,
):
    llm = ChatOllama(model=LLM_MODEL, base_url=os.environ["OLLAMA_HOST"])
    retriever = build_retriever(store, hybrid)

    @tool
    def search_knowledge_base(query: str) -> str:
        """Searches Acme Robotics internal documents for HR policy, product specs, FAQs, and runbooks."""
        chunks = retrieve(query, retriever, rerank=rerank, k=TOP_K)
        if not chunks:
            return "No results found."
        return "\n\n---\n\n".join(
            f"[source: {c.metadata.get('source', '?')}]\n{c.page_content}"
            for c in chunks
        )

    @before_model
    def trim_history(state: AgentState, runtime) -> dict | None:
        messages = state["messages"]
        trimmed = trim_messages(
            messages,
            max_tokens=TRIM_MAX_TOKENS,
            strategy="last",
            token_counter=llm,
            include_system=True,
            allow_partial=False,
            start_on="human",
        )
        if len(trimmed) == len(messages):
            return None
        return {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *trimmed],
        }

    conn = sqlite3.connect(STATE_DB, check_same_thread=False)
    saver = SqliteSaver(conn)

    return create_agent(
        model=llm,
        tools=[search_knowledge_base],
        system_prompt=SYSTEM_PROMPT,
        middleware=[trim_history],
        checkpointer=saver,
    )


def run(agent, question: str, thread_id: str = "cli"):
    print(f"\n========================================")
    print(f"Q: {question}")
    print(f"========================================")
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
    )

    tool_calls = 0
    for msg in result["messages"]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls += 1
                print(f"  -> tool call: {tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            preview = msg.content.replace("\n", " ")[:120]
            print(f"  <- tool result: {preview}...")

    final = result["messages"][-1].content
    print(f"\nA: {final}")
    print(f"(tool calls made: {tool_calls})")


def main():
    import uuid

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--store",
        choices=list(CHUNKERS),
        default=DEFAULT_STORE,
        help="Which Chroma store to query (default: recursive).",
    )
    parser.add_argument(
        "--hybrid",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_HYBRID,
        help="Combine BM25 with vector retrieval via RRF (default: on; --no-hybrid to disable).",
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RERANK,
        help="Cross-encoder rerank top-FETCH_K down to TOP_K (default: on; --no-rerank to disable).",
    )
    args = parser.parse_args()

    print(f"(store: {args.store}, hybrid: {args.hybrid}, rerank: {args.rerank})")
    agent = build_agent(store=args.store, hybrid=args.hybrid, rerank=args.rerank)
    thread_id = f"cli-{uuid.uuid4()}"
    run(agent, "What is 2 + 2?", thread_id=thread_id)
    run(agent, "What's the warranty on the R-200?", thread_id=thread_id)


if __name__ == "__main__":
    main()
