import os

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent import TOP_K, build_agent, vectorstore


st.set_page_config(page_title="Acme Robotics Assistant", page_icon=None)
st.title("Acme Robotics Assistant")
st.caption(f"Local RAG agent. Ollama: {os.environ.get('OLLAMA_HOST', 'unset')}")


@st.cache_resource
def get_agent():
    return build_agent()


if "history" not in st.session_state:
    st.session_state.history = []

agent = get_agent()

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
            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.history
            ]
            result = agent.invoke({"messages": messages})

        tool_queries = []
        for msg in result["messages"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc["name"] == "search_knowledge_base":
                        tool_queries.append(tc["args"].get("query", ""))

        sources = []
        for q in tool_queries:
            for chunk in vectorstore.similarity_search(q, k=TOP_K):
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
