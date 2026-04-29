from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_corpus():
    loader = DirectoryLoader(
        "data",
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()


def chunk_and_report(docs, chunk_size: int, chunk_overlap: int, show_first: int = 3):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    print(f"\n=== chunk_size={chunk_size}, chunk_overlap={chunk_overlap} ===")
    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:show_first]):
        source = chunk.metadata.get("source", "?")
        print(f"\n--- chunk {i} | source: {source} | len: {len(chunk.page_content)} ---")
        print(chunk.page_content)


def main():
    docs = load_corpus()
    print(f"Loaded {len(docs)} documents from data/")

    chunk_and_report(docs, chunk_size=500, chunk_overlap=50)
    chunk_and_report(docs, chunk_size=50, chunk_overlap=5)
    chunk_and_report(docs, chunk_size=5000, chunk_overlap=500)


if __name__ == "__main__":
    main()
