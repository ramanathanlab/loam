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
as `LOAM_STATE_DIR`, `LOAM_LEDGER_PATH`, `LOAM_PARENT_MEMORY_PATH`, or
`LOAM_WORKFLOW_ID` is set. `run-hermes` sets `LOAM_ENABLED=1` automatically.

## Hook contract

See `manifest.yaml`. `subagent-start.py` and `subagent-stop.py` now implement an
observe-only JSONL lifecycle ledger; the remaining starter hooks are deliberately
no-ops, so enabling the adapter remains safe. Replace their bodies with Loam
policy as it is designed.

- `subagent-start.py` / `subagent-stop.py` — append lifecycle observations to
  `$LOAM_STATE_DIR/subagent-ledger.jsonl` when `LOAM_STATE_DIR` is set, otherwise
  to the selected profile's `state_dir` or the legacy repository-local fallback.
  `subagent-stop.py` also extracts opt-in structured candidates into
  `candidate-results.jsonl`; candidates remain unvalidated and receive no
  findings, evidence, memory, or control credit until explicit reconciliation.
  The contract is `hypothesis_provenance_v1`, documented in
  `reconcile/HYPOTHESIS_PROVENANCE.md`.
  Set a unique directory per deployment to isolate all hook, orchestration, and
  tournament artifacts for that run:

  ```bash
  LOAM_STATE_DIR=/var/tmp/loam/${DEPLOYMENT_ID} .loam/run-hermes
  ```

  This runtime setting overrides profile `state_dir`; for an enabled tournament,
  it also writes Jnana state to `jnana-memory.json` and reports to
  `tournament-reports/` inside that directory.

- `pre-llm.py` runs every **N** turns, independently for each Hermes session.
  The interval is profile-configured as `pre_llm_every` (or defaults to 5 for
  the emergency environment override). The adapter supplies `loam_turn` and
  `loam_interval` in that script's JSON payload. It runs on turns N, 2N, 3N,
  and so on; set the interval to `1` to run on every turn. The count resets
  when the Hermes process restarts.

  A profile may additionally opt into `hypothesis_tournament`. When enabled,
  the hook uses Jnana/ProtoGnosis generation agents to seed an empty
  workflow-scoped state from the current user request, then runs a bounded
  pairwise ranking tournament before the model call. `initial_hypotheses`
  controls the one-time seed size (default `3`); existing states with at least
  two hypotheses are ranked without generating more. The hook persists Jnana
  JSON state and an immutable Loam report, and injects only a compact Elo
  ranking. Generation and rankings are advisory rather than empirical
  validation. The hook uses a nonblocking state lock and fails open on
  unavailable endpoints or contention.

  Configure `inference_endpoint` as an OpenAI-compatible API base URL (normally
  ending in `/v1`). If your endpoint is the exact Chat Completions URL, set
  `endpoint_is_chat_completions: true`; Loam removes `/chat/completions` before
  creating Jnana's client. `api_key_env` is optional for no-auth local services;
  when supplied, it names the environment variable holding that endpoint's
  credential rather than an OpenAI-specific key.
### Hypothesis-subagent orchestration

Set `hypothesis_orchestration.enabled: true` alongside `hypothesis_tournament` and an `execution` contract to make the harness LLM act as the campaign orchestrator. The pre-LLM hook runs the tournament first, then supplies a directive containing full selected hypotheses from the Jnana state. The harness MUST launch one `hypothesis-validator` per selected hypothesis in a `delegate_task` batch, using an orchestrator role. Validators MAY delegate only to leaf `code-generator` children.

`execution.mode: remote_mcp` requires code generators to use the configured MCP tool; `execution.mode: local` permits their local execution. Validator lifecycle hooks alone append `hypothesis-campaign.jsonl` start/completion records. A reasoner completion appends the round synthesis and its validator results. The structured `evidence_score` and `evidence_score_status` from each validator completion are included in the reasoner directive.

#### Payload ordering is load-bearing

Where text sits in a child's payload changes whether the child acts on it. On
2026-10-21, alphabetical `sort_keys` put the 8.6 KB `agent_definition` first,
pushing the real assignment to roughly char 9,000 of an 11 KB payload; the
validator reported that the hypothesis text "is not visible in this message"
while holding it, and sent its child only an id. The child then spent its whole
budget grepping the filesystem for that id.

So, as a contract rather than a preference:

- Campaign goal and rules belong in the first ~2 KB of a payload, and
  `tests/test_campaign_context.py` asserts that bound.
- `sort_keys` stays **off** in `_generated_validator_task`.
- Rules precede the validator's instruction in a child's context: the claim
  matters more than the wording of the request, and the validator's text may
  contradict it.
- The "authoritative — overrides any instruction below that conflicts" framing
  stays verbatim. It is what settles a conflict between campaign policy and a
  validator's paraphrase.

The goal and the hypothesis now compete for the front of that payload, so both
are explicitly budgeted (`MAX_CAMPAIGN_CHARS`, `MAX_RULES_CHARS`,
`MAX_HYPOTHESIS_CHARS`) rather than left to grow.

#### Run manifest

Every run writes `$LOAM_STATE_DIR/run-manifest.json` on its first hook event. It identifies the run, profile, execution configuration, campaign settings, colocated artifacts, and event counts without recording secrets or model summaries.

```yaml
enabled: true
active_profile: research
profiles:
  research:
    workflow_id: research-v1
    state_dir: /absolute/path/to/loam-state
    pre_llm_every: 1
    hypothesis_orchestration:
      enabled: true
    execution:
      mode: remote_mcp
      tool_name: globus_execute
      endpoint: https://compute.example/mcp
      function_name: run_python
      input_schema: run-python-v1
      runtime: python3.11
    hypothesis_tournament:
      enabled: true
      jnana_root: /absolute/path/to/jnana
      state_file: /absolute/path/to/jnana-memory.json
      reports_dir: /absolute/path/to/tournament-reports
      inference_endpoint: http://localhost:8000/v1
      model: tournament-model
```


- `pre-tool.py` may return `{"action":"block","message":"..."}` to veto a
  Hermes tool call.
- `pre-llm.py` may return `{"context":"..."}` to add turn-local context.
- Observer hooks (`subagent-start.py`, `subagent-stop.py`) have
  no behavior-changing return value in Hermes.

All hooks should fail closed with respect to themselves: malformed output or a
non-zero exit means the adapter logs the failure and lets Hermes proceed. Do
not put secrets in stdout; payloads can contain tool arguments and subagent
summaries.
