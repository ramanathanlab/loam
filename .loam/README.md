# Loam harness integration

`.loam/` is Loam's harness-neutral integration directory. It contains the
canonical lifecycle-event contract and its adapters; it is not Hermes-specific.

## Layout

- `manifest.yaml` — portable event-to-script mapping and JSON wire contract.
- `hooks/` — executable hook implementations. Each receives one JSON object on
  standard input and may emit one JSON object on standard output.
- `adapters/hermes/plugin/` — thin Hermes plugin that converts Hermes callback
  kwargs into the Loam payload and executes the canonical scripts.
- `run-hermes` — launches Hermes with project-plugin discovery enabled.

## Hermes setup (one-time, per user)

Hermes discovers project plugins only at `.hermes/plugins/`, so this repository
contains a tiny tracked discovery shim at `.hermes/plugins/loam-hooks/`. The
actual adapter lives here under `.loam/`.

To activate the adapter only while working in this repository:

```bash
.loam/run-hermes
```

Hermes additionally requires an explicit plugin allow-list. Add `loam-hooks`
to the `plugins.enabled` list in the Hermes configuration used by this
invocation. With the normal Hermes home, that is `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - loam-hooks
```

This allow-list entry alone does not activate Loam anywhere else: discovery is
limited to this repository and only when `HERMES_ENABLE_PROJECT_PLUGINS=1`,
which `run-hermes` sets.

To avoid any global Hermes configuration at all, set `HERMES_HOME` to a
Loam-specific Hermes home before invoking `run-hermes` and place the same
`plugins.enabled` entry in `$HERMES_HOME/config.yaml`. That isolated home needs
its own model/credential configuration.

### Global Hermes installation

Loam can also be installed once as a user plugin at
`~/.hermes/plugins/loam-hooks/`. The global shim loads this repository's adapter
(default `LOAM_ROOT=/Users/msinclair/github/loam`; override that variable for a
separate checkout). Enable it once with:

```bash
hermes plugins enable loam-hooks
```

The plugin remains inert unless `LOAM_ENABLED=1` or a Loam runtime variable such
as `LOAM_LEDGER_PATH`, `LOAM_PARENT_MEMORY_PATH`, or `LOAM_WORKFLOW_ID` is set.
`run-hermes` sets `LOAM_ENABLED=1` automatically.

## Hook contract

See `manifest.yaml`. `subagent-start.py` and `subagent-stop.py` now implement an
observe-only JSONL lifecycle ledger; the remaining starter hooks are deliberately
no-ops, so enabling the adapter remains safe. Replace their bodies with Loam
policy as it is designed.

- `subagent-start.py` / `subagent-stop.py` — append lifecycle observations to the
  repo-local `.loam/state/subagent-ledger.jsonl` ledger. `subagent-stop.py` also
  extracts opt-in structured candidates into `candidate-results.jsonl`; candidates
  remain unvalidated and receive no findings, evidence, memory, or control credit
  until explicit reconciliation. The available contracts include the narrow
  protein-metadata pilot and `hypothesis_provenance_v1`, documented in
  `reconcile/HYPOTHESIS_PROVENANCE.md`.
- `pre-llm.py` runs every **N** turns, independently for each Hermes session.
  The interval is profile-configured as `pre_llm_every` (or defaults to 5 for
  the emergency environment override). The adapter supplies `loam_turn` and
  `loam_interval` in that script's JSON payload. It runs on turns N, 2N, 3N,
  and so on; set the interval to `1` to run on every turn. The count resets
  when the Hermes process restarts.

  A profile may additionally opt into `hypothesis_tournament`. When enabled,
  the hook runs a bounded, workflow-scoped Jnana/ProtoGnosis tournament before
  the model call, persists Jnana JSON state and an immutable Loam report, and
  injects only a compact Elo ranking. It requires at least two existing
  hypotheses; it does not generate hypotheses, and its rankings are advisory
  rather than empirical validation. The hook uses a nonblocking state lock and
  fails open on unavailable endpoints or contention.

  Configure `inference_endpoint` as an OpenAI-compatible API base URL (normally
  ending in `/v1`). If your endpoint is the exact Chat Completions URL, set
  `endpoint_is_chat_completions: true`; Loam removes `/chat/completions` before
  creating Jnana's client. `api_key_env` is optional for no-auth local services;
  when supplied, it names the environment variable holding that endpoint's
  credential rather than an OpenAI-specific key.

- `pre-tool.py` may return `{"action":"block","message":"..."}` to veto a
  Hermes tool call.
- `pre-llm.py` may return `{"context":"..."}` to add turn-local context.
- Observer hooks (`post-tool.py`, `subagent-start.py`, `subagent-stop.py`) have
  no behavior-changing return value in Hermes.

All hooks should fail closed with respect to themselves: malformed output or a
non-zero exit means the adapter logs the failure and lets Hermes proceed. Do
not put secrets in stdout; payloads can contain tool arguments and subagent
summaries.
