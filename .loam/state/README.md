# Loam runtime state

This directory is created by Loam observer hooks at runtime.

- `subagent-ledger.jsonl` is an append-only lifecycle audit log.
- `candidate-results.jsonl` can contain shape-valid but unvalidated opt-in
  protein or hypothesis-provenance candidates.
- `hypothesis-ledger.jsonl`, `evidence-ledger.jsonl`, and `credit-ledger.jsonl`
  are created only by explicit hypothesis-provenance reconciliation.
- `hypothesis-memory.jsonl` is a compact projection of accepted hypotheses and
  source-backed evidence for pre-LLM recall.
- It may contain delegated task goals and child summaries; do not commit it.
- Configure `LOAM_LEDGER_PATH` to write the ledger to an explicit operator-owned
  location instead of this repository-local default.
