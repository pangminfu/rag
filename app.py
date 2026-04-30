import argparse
import os
import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import DEFAULT_HYBRID, TOP_K, build_agent, build_retriever
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
    return parser.parse_args()


args = parse_args()

st.set_page_config(page_title="Acme Robotics Assistant", page_icon=None)
st.title("Acme Robotics Assistant")
st.caption(
    f"Local RAG agent. Ollama: {os.environ.get('OLLAMA_HOST', 'unset')} • "
    f"Store: {args.store} • Retrieval: {'hybrid' if args.hybrid else 'vector'}"
)


@st.cache_resource
def get_agent(store: str, hybrid: bool):
    return build_agent(store=store, hybrid=hybrid)


@st.cache_resource
def get_retriever(store: str, hybrid: bool):
    return build_retriever(store=store, hybrid=hybrid)


if "history" not in st.session_state:
    st.session_state.history = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

agent = get_agent(args.store, args.hybrid)
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

question = st.chat_input("Ask about Acme Robotics...")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = agent.invoke(
                {"messages": [{"role": "user", "content": question}]},
                config=config,
            )

        tool_queries = []
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc["name"] == "search_knowledge_base":
                        tool_queries.append(tc["args"].get("query", ""))

        sources = []
        for q in tool_queries:
            for chunk in retriever.invoke(q)[:TOP_K]:
                sources.append(
                    {
                        "source": chunk.metadata.get("source", "?"),
                        "text": chunk.page_content,
                    }
                )

        answer = result["messages"][-1].content
        st.markdown(answer)

        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for i, src in enumerate(sources, start=1):
                    st.markdown(f"**[{i}] {src['source']}**")
                    st.text(src["text"])

        st.session_state.history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
