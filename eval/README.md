# eval/

Drop your RAGAS eval set at `eval/eval_set.jsonl`, then run:

```bash
uv run evaluate.py --store recursive --hybrid --rerank
```

## Schema

One JSON object per line:

```jsonl
{"question": "...", "ground_truth": "..."}
```

See `samples/acme/eval_set.jsonl` for a 13-question example over the bundled
demo corpus. To use it:

```bash
cp samples/acme/eval_set.jsonl eval/
```

`eval_set.jsonl` and `trace_*.jsonl` (per-question evaluation outputs) are
gitignored.
