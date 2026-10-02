# Loam runtime state

This tracked directory documents runtime artifacts only. Set `LOAM_STATE_DIR` to
an operator-owned, deployment-unique directory; Loam then creates every hook,
orchestration, and tournament artifact there.
- `run-manifest.json` is created on the first hook event and updated on later
  events. It records deployment/run identity, configured execution mode, campaign
  enablement, artifact paths, and lifecycle event counts; it contains no secrets
  or model summaries.
- `hypothesis-campaign.jsonl` records validator-only lifecycle transitions and a
  completed campaign round when its reasoner completes. Code-generator lifecycle
  events do not create campaign records.

- `subagent-ledger.jsonl` is an append-only lifecycle audit log.
- `candidate-results.jsonl` can contain shape-valid but unvalidated opt-in
  protein or hypothesis-provenance candidates.
- `hypothesis-ledger.jsonl`, `evidence-ledger.jsonl`, and `credit-ledger.jsonl`
  are created only by explicit hypothesis-provenance reconciliation.
- `hypothesis-memory.jsonl` is a compact projection of accepted hypotheses and
  source-backed evidence for pre-LLM recall.
- `hypothesis-orchestration.jsonl` is append-only campaign scheduling state. It
  associates tagged validator, code-generator, and reasoner child lifecycles with
  a Jnana tournament plan. Validator completions retain their reported calibrated
  `evidence_score`, or a `missing_or_invalid` status. These values are
  unvalidated model reports, not accepted evidence or credit.
- It may contain delegated task goals and child summaries; do not commit it.
- `LOAM_LEDGER_PATH` remains a legacy override for the lifecycle ledger only;
  prefer `LOAM_STATE_DIR` so all run outputs remain colocated.
