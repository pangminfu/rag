# RAG Assistant Template

A clone-and-go template for spinning up your own retrieval-augmented assistant
over a corpus of Markdown documents — FAQs, employee handbooks, user guides,
runbooks, anything text-shaped. Drop your `.md` files into `data/`, run
`uv run ingest.py`, and you have a working chat UI plus an eval harness.

LLM and embedding providers are pluggable — Ollama, OpenAI, and Gemini are
wired in (see `model_provider.py`). Chat and embeddings can use different
providers.

Pipeline highlights:

- Two chunking strategies (`recursive`, `semantic`) backed by independent stores
- Hybrid retrieval (BM25 + dense vectors) via `EnsembleRetriever`
- Cross-encoder rerank with `BAAI/bge-reranker-v2-m3`
- Tool-calling agent built on `langchain.agents.create_agent` with SQLite checkpointing
- Streamlit chat UI with token streaming and source attribution
- RAGAS evaluation (`faithfulness`, `answer_relevancy`, `context_precision`)

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- One of:
  - An Ollama server with the models you want pulled (e.g. `ollama pull gemma3:27b` and `ollama pull nomic-embed-text`)
  - An OpenAI API key
  - A Google AI Studio (Gemini) API key

The reranker (`BAAI/bge-reranker-v2-m3`) is local and provider-independent;
`sentence-transformers` downloads it on first use.

## Quickstart

```bash
# 1. Install dependencies into a project-local .venv
uv sync

# 2. Configure provider + branding via env vars
cp .env.example .env
# edit .env — set LLM_PROVIDER, LLM_MODEL, EMBEDDING_*, credentials,
# plus optional ASSISTANT_NAME and KNOWLEDGE_BASE_DESCRIPTION

# 3. Drop your Markdown documents into data/
cp /path/to/your/docs/*.md data/

# 4. Build a vector store
uv run ingest.py --strategy recursive

# 5. Chat with your assistant
uv run streamlit run app.py
```

That's it. The agent will answer questions from your documents and say
"I don't know" when the corpus doesn't cover the question.

## Try the demo corpus

A small fictional Acme Robotics corpus is bundled at `samples/acme/` so you
can validate the pipeline before plugging in your own content:

```bash
cp samples/acme/data/*.md data/
cp samples/acme/eval_set.jsonl eval/   # optional, for RAGAS scoring
cp samples/acme/prompts/system.md prompts/   # optional, scoped Acme prompt

uv run ingest.py --strategy recursive
uv run streamlit run app.py
```

The `.env.example` ships with `ASSISTANT_NAME` and `KNOWLEDGE_BASE_DESCRIPTION`
already filled in for the Acme demo — overwrite them when you swap in your
own content.

## Configuration

### Required env vars

| Var | Values |
|---|---|
| `LLM_PROVIDER` | `ollama` \| `openai` \| `gemini` |
| `LLM_MODEL` | provider-specific model name |
| `EMBEDDING_PROVIDER` | same set |
| `EMBEDDING_MODEL` | provider-specific |

Plus credentials for the providers you actually use: `OLLAMA_HOST`,
`OPENAI_API_KEY`, `GOOGLE_API_KEY`.

### Branding (optional)

| Var | Used by | Default |
|---|---|---|
| `ASSISTANT_NAME` | Streamlit page title and chat input placeholder | `"AI Assistant"` |
| `KNOWLEDGE_BASE_DESCRIPTION` | `search_knowledge_base` tool description (the LLM reads this to decide *when* to retrieve) | generic fallback |

`KNOWLEDGE_BASE_DESCRIPTION` materially affects retrieval quality — the more
specific you are about what your corpus covers, the better the LLM gates
tool calls. The system prompt at `prompts/system.md` is a separate lever you
can edit directly to set tone and scope.

### CLI flags

```bash
# Chunking strategy: writes to its own persist dir, so stores can coexist
uv run ingest.py --strategy recursive    # default
uv run ingest.py --strategy semantic

# Retrieval pipeline (agent.py, app.py, evaluate.py share these)
--store recursive|semantic    # which collection to query
--hybrid / --no-hybrid        # BM25 + vector ensemble (default: on)
--rerank / --no-rerank        # cross-encoder rerank (default: on)
--system-prompt-file PATH     # override the default prompts/system.md
```

## Run

### Tool-calling agent (CLI)

```bash
# Defaults: recursive store + hybrid + rerank
uv run agent.py

# Ablation
uv run agent.py --store semantic --no-hybrid --no-rerank
```

Conversation state is checkpointed in `state.db` keyed by `thread_id`. The
CLI generates a fresh UUID per run.

### Streamlit UI

```bash
uv run streamlit run app.py -- --store recursive --hybrid --rerank
```

Sources for each turn appear in an expander under the assistant message.

### Naive RAG baseline

Always-retrieve, similarity-only:

```bash
uv run rag.py "What is your refund policy?"
```

### RAGAS evaluation

Runs the agent end-to-end on `eval/eval_set.jsonl`, then replays retrieval
through the same pipeline so `context_precision` measures the retriever
rather than the agent's tool-use decision. Slow — the judge LLM scores every
answer (uses the same `LLM_PROVIDER`/`LLM_MODEL` as the agent).

```bash
# Default uses eval/eval_set.jsonl
uv run evaluate.py --store recursive --hybrid --rerank

# Or point at the bundled sample directly
uv run evaluate.py --eval-set samples/acme/eval_set.jsonl
```

Per-question traces land at `eval/trace_<store>_h<0|1>_r<0|1>.jsonl`.

## Project layout

```
chunking.py            Chunker ABC + recursive/semantic implementations
model_provider.py      LLM/embedding provider dispatch (ollama/openai/gemini)
vector_store_provider.py  Vector store backend (Chroma/Qdrant) abstraction
ingest.py              Corpus → chunks → embeddings → vector store
rerank.py              BGE cross-encoder rescoring
rag.py                 Naive always-retrieve baseline
agent.py               Tool-calling agent (create_agent, SqliteSaver)
app.py                 Streamlit chat UI
evaluate.py            RAGAS scoring harness
prompts/system.md      Default system prompt — edit to tailor scope/tone
data/                  Drop your Markdown docs here (gitignored)
eval/                  Drop your eval set here (gitignored)
samples/acme/          Bundled demo corpus + eval set + scoped system prompt
state.db               LangGraph checkpoint store (gitignored)
vectorstore/           Vector store persistence (gitignored)
```
