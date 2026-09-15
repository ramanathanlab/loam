# Loam runtime state

This directory is created by Loam observer hooks at runtime.

- `subagent-ledger.jsonl` is an append-only lifecycle audit log.
- It may contain delegated task goals and child summaries; do not commit it.
- Configure `LOAM_LEDGER_PATH` to write the ledger to an explicit operator-owned
  location instead of this repository-local default.
