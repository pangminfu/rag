import argparse
import os
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from agent import DEFAULT_HYBRID, DEFAULT_RERANK, TOP_K, build_agent, build_retriever, retrieve
from chunking import CHUNKERS, RecursiveChunker


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--store",
        choices=list(CHUNKERS),
        default=RecursiveChunker.name,
    )
    parser.add_argument(
        "--hybrid",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_HYBRID,
    )
    parser.add_argument(
        "--rerank",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_RERANK,
    )
    return parser.parse_args()


args = parse_args()

st.set_page_config(page_title="Acme Robotics Assistant", page_icon=None)
st.title("Acme Robotics Assistant")
st.caption(
    f"Local RAG agent. Ollama: {os.environ.get('OLLAMA_HOST', 'unset')} • "
    f"Store: {args.store} • Retrieval: {'hybrid' if args.hybrid else 'vector'} • "
    f"Rerank: {'on' if args.rerank else 'off'}"
)


@st.cache_resource
def get_agent(store: str, hybrid: bool, rerank: bool):
    return build_agent(store=store, hybrid=hybrid, rerank=rerank)


@st.cache_resource
def get_retriever(store: str, hybrid: bool):
    return build_retriever(store=store, hybrid=hybrid)


if "history" not in st.session_state:
    st.session_state.history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

agent = get_agent(args.store, args.hybrid, args.rerank)
retriever = get_retriever(args.store, args.hybrid)
config = {"configurable": {"thread_id": st.session_state.thread_id}}

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("sources"):
            with st.expander(f"Sources ({len(turn['sources'])})"):
                for i, src in enumerate(turn["sources"], start=1):
                    st.markdown(f"**[{i}] {src['source']}**")
                    st.text(src["text"])

def stream_assistant_text(question: str, config: dict):
    """Yield AI content tokens as they stream from the agent.

    During tool-call phases the chunks carry tool_call_chunks but no content,
    so they're naturally skipped — st.write_stream just sees a pause.
    """
    for chunk, _ in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="messages",
        config=config,
    ):
        if isinstance(chunk, AIMessageChunk) and chunk.content:
            yield chunk.content


def collect_turn_tool_queries(config: dict) -> list[str]:
    """Read this turn's search_knowledge_base queries from checkpointed state.

    Walking back to the last HumanMessage, then forward, is more robust than
    accumulating partial tool_call_chunks during the stream.
    """
    state = agent.get_state(config)
    messages = state.values.get("messages", [])
    last_human = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage):
            last_human = i
            break
    queries = []
    for msg in messages[last_human + 1 :]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "search_knowledge_base":
                    q = tc["args"].get("query", "")
                    if q:
                        queries.append(q)
    return queries


question = st.chat_input("Ask about Acme Robotics...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer = st.write_stream(stream_assistant_text(question, config))

        tool_queries = collect_turn_tool_queries(config)

        sources = []
        for q in tool_queries:
            for chunk in retrieve(q, retriever, rerank=args.rerank, k=TOP_K):
                sources.append(
                    {
                        "source": chunk.metadata.get("source", "?"),
                        "text": chunk.page_content,
                    }
                )

        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for i, src in enumerate(sources, start=1):
                    st.markdown(f"**[{i}] {src['source']}**")
                    st.text(src["text"])

        st.session_state.history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
