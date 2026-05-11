"""Score the agent on `eval/eval_set.jsonl` with RAGAS metrics.

The CLI flags mirror agent.py so the same toggles flow into both the agent's
answer and the retrieval that context_precision sees:

    uv run evaluate.py --store recursive --no-hybrid --no-rerank   # baseline
    uv run evaluate.py --store semantic                            # + semantic + hybrid + rerank

Each question runs on a fresh thread_id so prior eval state never bleeds in.
Eval threads pollute state.db with disposable rows; wipe it if that bothers you.
"""

import argparse
import json
import os
import sys
import uuid

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from agent import (
    DEFAULT_HYBRID,
    DEFAULT_RERANK,
    DEFAULT_STORE,
    DEFAULT_SYSTEM_PROMPT_PATH,
    TOP_K,
    build_agent,
    build_retriever,
    load_system_prompt,
    retrieve,
)
from chunking import CHUNKERS
from model_provider import ModelProvider

EVAL_PATH = "eval/eval_set.jsonl"
SAMPLE_EVAL_PATH = "samples/acme/eval_set.jsonl"


def load_eval_set(path: str = EVAL_PATH) -> list[dict]:
    if not os.path.exists(path):
        print(
            f"No eval set at {path}.\n"
            f"Copy the bundled sample to start: cp {SAMPLE_EVAL_PATH} {path}\n"
            f"Or write your own (one JSON object per line: "
            f"{{\"question\": ..., \"ground_truth\": ...}}).",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def collect_rows(entries, *, store, hybrid, rerank, system_prompt):
    """Run the agent on every eval question and replay retrieval.

    The agent decides on its own whether to call the search tool. We always
    replay retrieval through the same configured pipeline so context_precision
    measures the retriever, not the agent's tool-use decision.
    """
    agent = build_agent(
        store=store, hybrid=hybrid, rerank=rerank, system_prompt=system_prompt
    )
    retriever = build_retriever(store=store, hybrid=hybrid)

    rows = []
    for i, entry in enumerate(entries, 1):
        q = entry["question"]
        print(f"[{i}/{len(entries)}] {q}")

        thread_id = f"eval-{uuid.uuid4()}"
        result = agent.invoke(
            {"messages": [{"role": "user", "content": q}]},
            config={"configurable": {"thread_id": thread_id}},
        )
        answer = result["messages"][-1].content

        chunks = retrieve(q, retriever, rerank=rerank, k=TOP_K)
        contexts = [c.page_content for c in chunks]

        rows.append(
            {
                "question": q,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": entry["ground_truth"],
            }
        )
    return rows


def run_eval(*, store, hybrid, rerank, system_prompt, eval_path=EVAL_PATH):
    entries = load_eval_set(eval_path)
    provider = ModelProvider()
    print(
        f"Evaluating {len(entries)} questions "
        f"(store={store}, hybrid={hybrid}, rerank={rerank}, judge={provider.llm_model})\n"
    )

    rows = collect_rows(
        entries,
        store=store,
        hybrid=hybrid,
        rerank=rerank,
        system_prompt=system_prompt,
    )

    judge_llm = LangchainLLMWrapper(provider.chat())
    judge_embedder = LangchainEmbeddingsWrapper(provider.embeddings())

    print("\nScoring with RAGAS — Gemma judging every answer, this is slow...\n")
    scores = evaluate(
        Dataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_embedder,
    )

    print("\n=== RAGAS scores ===")
    print(f"  store:   {store}")
    print(f"  hybrid:  {hybrid}")
    print(f"  rerank:  {rerank}")
    print(scores)

    trace_path = f"eval/trace_{store}_h{int(hybrid)}_r{int(rerank)}.jsonl"
    with open(trace_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nPer-question trace written to {trace_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", choices=list(CHUNKERS), default=DEFAULT_STORE)
    parser.add_argument(
        "--hybrid", action=argparse.BooleanOptionalAction, default=DEFAULT_HYBRID
    )
    parser.add_argument(
        "--rerank", action=argparse.BooleanOptionalAction, default=DEFAULT_RERANK
    )
    parser.add_argument("--system-prompt-file", default=DEFAULT_SYSTEM_PROMPT_PATH)
    parser.add_argument(
        "--eval-set",
        default=EVAL_PATH,
        help=f"Path to a JSONL eval set (default: {EVAL_PATH}).",
    )
    args = parser.parse_args()

    system_prompt = load_system_prompt(args.system_prompt_file)
    run_eval(
        store=args.store,
        hybrid=args.hybrid,
        rerank=args.rerank,
        system_prompt=system_prompt,
        eval_path=args.eval_set,
    )


if __name__ == "__main__":
    main()
