# Acme Robotics — sample corpus

A small fictional corpus you can use to try the template end-to-end before
plugging in your own documents. Six Markdown files, a 13-question RAGAS eval
set, and a scoped system prompt:

- `data/hr_leave_policy.md`
- `data/hr_code_of_conduct.md`
- `data/product_r200_spec.md`
- `data/product_x9_spec.md`
- `data/faq_customers.md`
- `data/runbook_deploy.md`
- `eval_set.jsonl` — RAGAS eval set in the schema `evaluate.py` expects
- `prompts/system.md` — Acme-scoped system prompt (overrides the generic default at `prompts/system.md`)

## How `data/` and `eval/` work

The repo's top-level `data/` and `eval/` directories are user-content drop
zones — they ship empty (gitignored) and you fill them in:

- **`data/`** — drop your Markdown documents here, then run
  `uv run ingest.py --strategy recursive`. Only `.md` files are indexed; any
  naming convention is fine. Don't drop README files in here — they'll be
  embedded along with the corpus.
- **`eval/`** — drop a RAGAS eval set at `eval/eval_set.jsonl`. Schema:
  one JSON object per line, `{"question": "...", "ground_truth": "..."}`.
- **`prompts/system.md`** — generic by default; edit in place, or overwrite
  with a scoped version like the one in this sample.

## Use the demo

From the repo root:

```bash
cp samples/acme/data/*.md data/
cp samples/acme/eval_set.jsonl eval/
cp samples/acme/prompts/system.md prompts/   # optional, scoped Acme prompt

# Optional branding for the demo (or set in .env)
export ASSISTANT_NAME="Acme Assistant"
export KNOWLEDGE_BASE_DESCRIPTION="Searches Acme Robotics internal documents for HR policy, product specs, FAQs, and runbooks."

uv run ingest.py --strategy recursive
uv run streamlit run app.py
```

To go back to a clean template, delete the contents of `data/`,
`eval/eval_set.jsonl`, and reset `prompts/system.md`, then re-run ingest with
your own documents.
