import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings


PERSIST_DIR = "./chroma_db"
LLM_MODEL = "gemma4:26b"
TOP_K = 4

SYSTEM_PROMPT = (
    "You are a helpful assistant for Acme Robotics employees. "
    "Use the search_knowledge_base tool when the question is about "
    "Acme Robotics — its HR policies, products, FAQs, or internal runbooks. "
    "For general knowledge or arithmetic, answer directly without the tool. "
    "If the tool returns nothing relevant, say you don't know."
)


def _build_vectorstore():
    embedder = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=os.environ["OLLAMA_HOST"],
    )
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embedder)


vectorstore = _build_vectorstore()


@tool
def search_knowledge_base(query: str) -> str:
    """Searches Acme Robotics internal documents for HR policy, product specs, FAQs, and runbooks."""
    chunks = vectorstore.similarity_search(query, k=TOP_K)
    if not chunks:
        return "No results found."
    return "\n\n---\n\n".join(
        f"[source: {c.metadata.get('source', '?')}]\n{c.page_content}" for c in chunks
    )


def build_agent():
    llm = ChatOllama(model=LLM_MODEL, base_url=os.environ["OLLAMA_HOST"])
    return create_agent(
        model=llm,
        tools=[search_knowledge_base],
        system_prompt=SYSTEM_PROMPT,
    )


def run(agent, question: str):
    print(f"\n========================================")
    print(f"Q: {question}")
    print(f"========================================")
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})

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
    agent = build_agent()
    run(agent, "What is 2 + 2?")
    run(agent, "What's the warranty on the R-200?")


if __name__ == "__main__":
    main()
