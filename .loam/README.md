# Loam harness integration

`.loam/` is Loam's harness-neutral integration directory. It contains the
canonical lifecycle-event contract and its adapters; it is not Hermes-specific.

## Layout

- `manifest.yaml` — portable event-to-script mapping and JSON wire contract.
- `hooks/` — executable hook implementations. Each receives one JSON object on
  standard input and may emit one JSON object on standard output.
- `adapters/hermes/plugin/` — thin Hermes plugin that converts Hermes callback
  kwargs into the Loam payload and executes the canonical scripts.

## Hermes setup (one-time, per user)

Hermes discovers project plugins only at `.hermes/plugins/`, so this repository
contains a tiny tracked discovery shim at `.hermes/plugins/loam-hooks/`. The
actual adapter lives here under `.loam/`.

To activate the adapter, launch through `run-campaign` from a campaign
directory. It exports `HERMES_ENABLE_PROJECT_PLUGINS=1` and generates the
discovery shim beside itself, so the adapter is found wherever the campaign
lives:

```bash
cd ~/campaigns/my-campaign && ./run-campaign
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
limited to a directory with the shim, and only when
`HERMES_ENABLE_PROJECT_PLUGINS=1`, which `run-campaign` sets.

`run-campaign` already avoids global Hermes configuration: it builds an
isolated `HERMES_HOME` per campaign from the repo's shared home, so the
`plugins.enabled` entry belongs in that shared `config.yaml`. To point somewhere
else entirely, set `HERMES_HOME` before invoking it and put the same entry in
`$HERMES_HOME/config.yaml`. An isolated home needs its own model and credential
configuration.

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
`LOAM_WORKFLOW_ID` is set. `run-campaign` sets `LOAM_ENABLED=1` automatically.

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
  LOAM_STATE_DIR=/var/tmp/loam/${DEPLOYMENT_ID} ./run-campaign
  ```

  `run-campaign` normally assigns this itself, to the campaign directory's
  `state/`, and unsets an inherited value so a parent shell cannot redirect a
  run. Setting it on the command line is for a deployment that needs to place
  the artifacts elsewhere.

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
  workflow-scoped state from the campaign's goal — the same `campaigns/<goal>.md`
  the validators and the reasoner are given, falling back to the current user
  request for a profile that names no campaign — then runs a bounded pairwise
  ranking tournament before the model call. `initial_hypotheses` controls the
  one-time seed size (default `3`).

  The stored goal is re-set whenever it moves, because the pairwise judge reads
  it (and the evaluation criteria parsed from it) on every match: an edited goal
  file or a fresh reasoner brief reaches the ranking rather than leaving it
  judging against the first round's text. `refresh_hypotheses` (default `0`,
  off) then lets that new knowledge enter the pool as candidates: N
  brief-informed hypotheses per *new brief*, not per turn. Left at `0`, a
  seeded state is only ever re-ranked, never added to. Both of those need a
  campaign goal to measure movement against; a profile that names none keeps
  the earlier behaviour, seeding once from the turn's request and then only
  re-ranking, because that request differs every turn.

  A brief is recorded as spent only once it has actually bought hypotheses. A
  generation that comes back empty — Jnana logs an agent-level failure, stores
  nothing and raises nothing — leaves the refresh owed, and the next turn tries
  the same brief again (`status: refresh_yielded_nothing`). A re-goal or refresh
  that raises is advisory like the rest of the hook: it is reported in the turn's
  report as `goal_sync: "failed: ..."` and costs the turn its rankings, not its
  other context. A state seeded before these markers existed needs no migration:
  its knowledge-pre-pass brief is recovered from the goal it is stored inside,
  and an unmoved goal is stamped rather than re-parsed, so the first turn after
  an upgrade spends no LLM call and no refresh (`goal_sync: unchanged_stamped`).
  The hook persists Jnana
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
  Reserving is idempotent in the children's goals, because it happens inside
  request middleware that may run again for one dispatch. A child whose tag was
  replayed, re-dispatched or lost still gets a run: the same one for a replayed
  start, otherwise the run reserved for its own goal, and only failing that a new
  one. Every substitution is logged, so the record shows which run the child was
  told about and which it got — a child must never end up with no run, because
  that would refuse the writes its own instructions asked for.
- **Every tool call.** After `tool_policy` (what a role may call),
  `workspace_policy` decides where a code-generator may write: only `src/` and its
  own `outputs/rM/` and `scratch/rM/`. Record files, `data/`, other runs, other
  hypotheses, stray names and anything outside are blocked, with the legal place in
  the message. Other runs' results, this hypothesis's or another's, may be read only
  from `outputs/rK/` of a `completed` run with a verified manifest; each such read is
  recorded in the reading run's `inputs` with the source file's hash. Reads are
  read-only calls and stay allowed whatever state the run is in — including
  `skill_view` and `skills_list`, so a child whose run is frozen or whose workspace
  failed can still re-read the campaign's recipe.
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

### Metrics export (Weights & Biases)

With a `metrics: {enabled: true, project: ...}` block in the active profile, the
adapter turns on an **in-process** W&B exporter
(`adapters/hermes/plugin/wandb_metrics.py`). Unlike the spawned lifecycle hooks
above, this is not a subprocess per event: it registers two Hermes in-process
observer hooks — `post_api_request` and `post_tool_call` — which fire on every
API and tool call, a rate at which spawning a process each time is not
affordable. It is the shape of Hermes's own `observability/langfuse` plugin,
aimed at W&B and with Loam's agent-role attribution added.

It records, into **one W&B run per campaign** (keyed by `LOAM_RUN_ID`, so a
walltime resume rejoins the same run rather than forking a new one):

- **Tokens per agent** — from `post_api_request`'s canonical `usage` buckets
  (input / output / cache-read / cache-write / reasoning), attributed to the
  Loam role that owns the call's session via the map built at `subagent_start`
  (reasoner, hypothesis-validator, code-generator; an unmapped session is the
  coordinator).
- **Wall time per tool call** — from `post_tool_call`'s `duration_ms` (Hermes
  measures start-to-result; the exporter does not time the call itself).
- **Inference time per call** — from `post_api_request`'s `api_duration`.
- **The gap between an inference ending and the tool call it triggers** —
  `tool_start − inference_end`. The tool's start is stamped at `pre_tool_call`;
  each tool call is linked back to its inference by `api_request_id` (falling
  back to the session's last inference), which also carries that inference's
  token usage onto the tool-call record.
- **The skill each tool call ran under** — the reviewed recipe most recently
  read in that session (`skill_view`/`skills_list` `name`), so tool time can be
  grouped by skill as well as by tool. A call before any skill read is charged
  to `(none)`.

These land as per-call rows in an `events` table **and** as cumulative scalar
series (live charts), plus end-of-run `summary/agents`, `summary/tools` and
`summary/skills` tables. Every event is also appended to
`<state_dir>/metrics-events.jsonl`, a durable record that survives when W&B is
unreachable.

**Per tool call, in the Pando workspace.** When the per-hypothesis workspace is
on, each tool call a code-generator child makes also gets one line in
`runs/<run_id>/metrics.jsonl` inside its run folder (sibling to `run.json`),
carrying the five quantities above — tool duration, the triggering inference's
duration and token usage, and the inference→tool gap. It is append-only and
modelled on `log.jsonl`, but deliberately not the hash-chained log and not
`run.json`, so recording a metric never touches the provenance record or the
managed run index. Only a code-generator child has a run folder; other agents'
calls still reach W&B and the flat state-dir log. The write is done in-process
by the exporter (that is where Hermes surfaces the timings), resolving the
call's run exactly as `pre-tool.py`'s workspace enforcement does — its session's
latest `agent_started` row — and `hypothesis_workspace.record_tool_metrics`
no-ops for any role without a run.

The block is off by default, and the two hot-path hooks are registered **only**
when it is on, so an unconfigured campaign pays nothing. Configuration errors
(a missing `project`, an unknown `mode`) are loud — they raise in
`loam_config.py` like a bad `execution:` block. Runtime failures are the
opposite and always swallowed: no `wandb` installed, no API key, or an
unreachable host leaves the exporter inert and never touches the running
campaign. The API key is never stored in the profile: online mode authenticates
either from an environment variable — `api_key_env` names it, default
`WANDB_API_KEY` — or from an existing `wandb login` (the host's key in
`~/.netrc`), so a logged-in machine needs no exported key. For a self-hosted
`base_url`, the login must be for that host. Install the optional dependency
with `pip install -e .[metrics]` in the environment that runs Hermes. See
`examples/loam.yaml.example` for the full block.
