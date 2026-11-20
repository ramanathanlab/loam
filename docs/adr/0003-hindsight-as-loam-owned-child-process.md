# Lesson memory runs Hindsight as a child process loam owns, from hindsight-api-slim

Lesson memory must stay on the operator's laptop (no hosted memory service), use the same ALCF/argo endpoints as the agents, and get a fresh API token when each campaign starts. So `Campaign.run` launches `hindsight-api` itself as a child process on the embedded Postgres (`pg0`), passes every setting as `HINDSIGHT_API_*` environment variables built from the loam config, waits for `/health`, and stops it when the campaign ends. Only `hindsight-api-slim[embedded-db]` and `hindsight-client` are installed, pinned exactly, with remote embeddings and the `rrf` reranker so no ML model runs locally.

## Considered Options

- **`hindsight-all` (the documented all-in-one install).** Rejected: its dependencies install torch, transformers and sentence-transformers whatever providers are configured.
- **The `hindsight-embed` daemon.** Rejected: it outlives loam and is reused as-is by the next start, so a later campaign would run with the previous campaign's environment and expired token; it also writes the API key to `~/.hindsight/profiles/<name>.env` in plain text.
- **Hindsight inside loam's process (a thread).** Rejected: Hindsight's work would share loam's event loop and process, and its in-process launcher only ships in `hindsight-all`.
- **Hindsight Cloud.** Rejected: campaign data would leave the machine.

## Consequences

- Campaigns running at the same time must share one server the operator starts, via `memory.url`, rather than each starting its own on the same database.
- The pin is exact because `HINDSIGHT_API_*` names are not a stable interface across Hindsight versions; upgrading means re-running `examples/memory_smoke.py`.
