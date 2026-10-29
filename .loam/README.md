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
  The available contracts include the narrow protein-metadata pilot and
  `hypothesis_provenance_v1`, documented in `reconcile/HYPOTHESIS_PROVENANCE.md`.
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

### Hypothesis workspace (Pando-style)

With `workspace.enabled: true` (and orchestration on), every hypothesis-validator
gets its own folder, laid out like a [Pando](https://github.com/zjniu/Pando)
experiment and kept by the hooks in `.loam/hooks/hypothesis_workspace.py`:

```
$LOAM_STATE_DIR/hypotheses/<plan_id>/<hypothesis_id>/
├─ experiment.json   index + state (open -> closed)              hooks only
├─ log.jsonl         append-only history: plan / result / note /  hooks only
│                    report / violation, each line hash-chained
├─ src/              code the children ran (run_code payloads saved as rM_NN.py)
├─ runs/rM/run.json  per child: goal, state, code hashes, inputs, manifest   hooks only
├─ outputs/rM/       that run's result files; frozen when the run ends
├─ scratch/rM/       that run's temporary files; never hashed
└─ data/             campaign inputs, seeded before any child runs; read-only to agents
```

On HPC the same layout lives under `<execution.workdir_root>/<plan_id>/<hypothesis_id>/`.

- **Start.** A validator's start creates the folder. Delegating children reserves
  one run each (`[loam-run-id=rM]` in the child's goal, so its instructions can
  name `outputs/rM/`); a child's start moves its run to `started` and logs a `plan`.
- **Every tool call.** After `tool_policy` (what a role may call),
  `workspace_policy` decides where a code-generator may write: only `src/` and its
  own `outputs/rM/` and `scratch/rM/`. Record files, `data/`, other runs, other
  hypotheses, stray names and anything outside are blocked, with the legal place in
  the message. Other runs' results, this hypothesis's or another's, may be read only
  from `outputs/rK/` of a `completed` run with a verified manifest; each such read is
  recorded in the reading run's `inputs` with the source file's hash.
- **Stop.** A child's stop records the run's state, hashes `outputs/rM/` (a remote
  child prints a `LOAM_MANIFEST` block it is given the program for), logs a
  `result` and freezes the run. A validator's stop drops unfinished runs, closes
  `experiment.json` with values copied from its `validator_completed` row, logs a
  `report` and audits the folder (log chain, strays, local outputs changed after
  freeze). Problems are logged and reported as a `workspace_violation` campaign
  row; nothing is repaired. The completion signal is unchanged.
- **Batches.** `run_batch` writes to the server's shared batch folder
  (`execution.batch_root`, default `<workdir_root>/../batches`), not to the run.
  A child must name each batch `<hypothesis>-<run>-<name>` with labels
  `{plan_id, hypothesis_id, run}`, so it is recorded in `run.json` `batches` and
  the server refuses to resume it under another hypothesis. `batch_status` and
  batch-folder reads are limited to the run's own batches. The child is given a
  program that copies a finished batch into `outputs/rM/<batch_id>/`, so batch
  results are hashed, frozen and shareable like any other output.
- **Campaign inputs.** `workspace.inputs_dir` is copied into every hypothesis's
  `data/` before its children run, and each file is hashed into `experiment.json`
  `inputs`. Each hypothesis gets its own copy on purpose: editing the campaign's
  inputs afterwards cannot retroactively change what an earlier hypothesis was
  given, and a file whose hash no longer matches what was recorded is logged as a
  violation rather than silently replaced. Symlinks are resolved, so `data/` can
  never point out of the workspace, and a seed over `max_input_bytes` (default
  1 GiB) is refused whole rather than copied halfway. For remote execution the
  hook cannot write to HPC, so stage the inputs there once and set
  `remote_inputs_dir`: the validator's start launches the verifier's `seed`,
  which copies them where they are and records the hashes locally.
  `shared_inputs` is the other half of the trade: absolute paths of read-only
  reference data too large to copy per hypothesis. Children may read them in
  place and nothing may write there; each file read is recorded in the run's
  `inputs`, with a hash when the hook can see the file and `hashed: false` when
  it is HPC-side. That is weaker provenance than a copied `data/`, which is why
  it is opt-in and per-path.
- **HPC verifier.** The hooks can't reach HPC files. When configured,
  `.loam/reconcile/verify_workspace.py` is launched detached: `fill` hashes a
  remote run whose child reported no manifest (`recorded_by_loam`), and `recheck`
  re-hashes completed remote runs at validator stop, and `batches` flags a run's
  batches that were still unfinished when it closed (`batches_unfinished`). It can
  also be run by hand.

```yaml
    workspace:
      enabled: true
      root: ""               # default: <state_dir>/hypotheses
      strict: true           # false: violations are logged, not blocked
      verifier_python: /abs/path/to/venv/bin/python   # needs fastmcp
      verifier_mcp_url: http://127.0.0.1:8000/mcp     # the HPC MCP server
      inputs_dir: ""         # local dir copied into every hypothesis's data/
      remote_inputs_dir: ""  # dir already staged on HPC; the verifier copies it
      shared_inputs: []      # absolute paths children may read in place, read-only
      max_input_bytes: 0     # 0: 1 GiB; refuses an oversize seed whole
    execution:
      workdir_root: /flare/<PROJECT>/<USER>/hpc-run/work
      batch_root: ""         # default: <workdir_root>/../batches (the server's HPC_BATCH_ROOT)
      workdir_aliases:       # other spellings of the same HPC directory
        /lus/flare/projects/<PROJECT>: /flare/<PROJECT>
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
