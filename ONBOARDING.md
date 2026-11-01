# Onboarding: running a Loam campaign end-to-end

This is the shortest path from a fresh checkout to a running campaign. It
assumes you have an HPC allocation (Aurora or Polaris) and OpenAI-compatible
inference endpoints you can reach.

Loam has **three subsystems** that must all be live *before* a campaign starts,
because the launcher's preflight is fail-closed: it performs one real inference
completion per model role **and** one real Aurora `run_code` probe, and aborts
the launch if either fails.

```
run-campaign  ──►  campaign_preflight.py   (fail-closed readiness gate)
     │
     ├─ (1) HOOKS      Hermes loads the loam-hooks plugin ──► .loam/adapters/hermes/plugin
     │                 which fires .loam/hooks/* on turn / subagent-start / stop events
     │
     ├─ (2) MCP (HPC)  Hermes mcp_servers.hpc ──► http://127.0.0.1:8000/mcp
     │                 a locally-running scripts/mcp_server.py
     │                   ──► Globus Compute endpoint ──► PBS allocation ──► Aurora
     │                 exposes run_code / inspect_filesystem / run_batch as mcp__hpc__*
     │
     └─ (3) INFERENCE  argo-proxy (:51900)          reasoner + all subagents
                       alcf_inference_proxy.py (:51911)  Jnana tournament judge
```

The data flow of one campaign round:

```
Jnana generation/ranking (pre-LLM hook)
  → hypothesis validators (one per selected hypothesis)
    → code-generator children (run_code on the HPC MCP server)
  → validator evidence scores
  → reasoner synthesis
  → next-round Jnana steering
```

---

## Step 0 — Prerequisites (once)

- A Loam virtualenv. The scripts default to `~/.loam-venv-312`; create your own
  and `pip install -e .` in this repo, then add `fastmcp`, `globus-compute-sdk`,
  `academy-py`, `pyyaml`, and a Jnana checkout.
- The `hermes` CLI, with a model provider configured.
- **The skills registered in your Hermes library.** Hermes resolves
  `campaign.skills` names against the library of the home it runs in, and a
  checkout does not register its own `skills/` tree. `run-campaign` does this for
  you on every launch, so you only run it by hand when launching some other way,
  or to install into a home outside the checkout:

  ```bash
  python3 scripts/install_skills.py                              # the checkout's shared home
  python3 scripts/install_skills.py --hermes-home ~/.hermes      # your own home
  python3 scripts/install_skills.py --hermes-home ~/.hermes --copy   # no link back to the checkout
  ```

  It symlinks by default, so one tree stays the source of truth and a `git pull`
  updates the library with it; `--copy` is for a home that must not reference the
  checkout. Re-running is safe, and it refuses to replace a library entry it does
  not own. Confirm with `hermes skills list`, or `skill_view(name="chai")`.
- A **Globus Compute endpoint already running on the HPC login node** (Aurora:
  `~/.globus_compute/loam-aurora-flex`). This is the one piece that lives on the
  cluster, not your laptop. You need its **manager UUID**. See the
  `hpc-deployment` skill's "Endpoint discovery and flex-scaling validation"
  section for how to verify it is healthy and flex-scaling.

## Step 1 — Inference endpoints

Two OpenAI-compatible bases the campaign talks to:

- **Argo tunnel on `:51900`** — the Hermes reasoner and every delegated subagent
  (`argo:gpt-5.6-sol`). Bring up this SSH tunnel yourself.
- **`scripts/alcf_inference_proxy.py` on `:51911`** (Minerva) — Jnana's ranking
  judge. **The preflight starts this** if nothing is listening, and waits for
  the port — it is the step whose ranking-role completion depends on it. Opt out
  with `ALCF_PROXY=0` when the ranking model is served another way; override the
  port, interpreter and log with `ALCF_PROXY_PORT`, `ALCF_PROXY_PYTHON` and
  `ALCF_PROXY_LOG`. On an auth error run `alcf-tokens login inference`.

## Step 2 — Bring up the HPC MCP server ("the hpc thing")

This is the `hpc-deployment` skill. From that skill's SKILL.md §Deployment:

1. **Edit `skills/hpc-deployment/set_env_akv.sh`** (or your own copy of
   `set_env.sh`) — the single definition point. See the hardcoded-value table
   below for everything a new user must change.

   The per-user copies (`set_env_akv.sh`, `set_env_hari.sh`,
   `set_env_msinclair.sh`) are tracked and name **different** endpoints, base
   dirs and allocation shapes. This step sources whichever you edited, so the
   choice is yours — but the preflight's HPC auto-start defaults to
   `set_env.sh`, which is a template that refuses to run until its `FILL_IN`
   markers are filled, so if you edited one of the per-user copies and intend
   to let it start an allocation, export
   `HPC_ENV_FILE=$PWD/skills/hpc-deployment/set_env_akv.sh`. Bringing the server
   up yourself here avoids the question entirely, which is what a real run should
   do anyway.
2. `source skills/hpc-deployment/set_env_akv.sh` — **every later step runs in
   that same shell**; the server reads its configuration from the process
   environment and nowhere else.
3. `python scripts/mcp_server.py` — starts the local FastMCP server on `:8000`,
   creates the Globus Compute Executor with your endpoint UUID, submits the PBS
   allocation, and launches the Academy agents. In **cluster mode** that is one
   `HpcClusterAgent` running the Ensemble Launcher cluster plus `HPC_POOL_SIZE`
   client agents that submit into it; EL owns tile placement. A missing env var
   or dependency fails here at once, before any job is submitted.
4. `python scripts/smoke_test_client.py` — **do not trust the server until this
   passes.** It requires a real `run_code` result and bounded
   `inspect_filesystem` output from the worker.
5. Submit a couple of concurrent cheap calls and confirm they *overlap* rather
   than serializing.

**Shut it down only with SIGINT/SIGTERM** (`Ctrl-C` / `kill -INT`), never
`kill -9`: a clean exit is what runs `Manager.close()` and releases the Aurora
allocation. `kill -9` orphans the EL cluster until walltime.

## Step 3 — Wire MCP + hooks into the Hermes config

**Copy `examples/hermes-config.yaml.example` into the Hermes home as
`config.yaml`** (the shared home is `.hermes/.loam-home/config.yaml`) and fill in
every `<PLACEHOLDER>`. Do not hand-write this file from the keys below — the
example is the complete one, and the pieces most easily missed are the ones that
are not Loam-specific at all:

- `model:` — the reasoner's model, provider, and `base_url` (the `:51900` tunnel).
  Without this block there is no campaign parent.
- `custom_providers:` — one entry per endpoint from Step 1 (Argo and Minerva).
- `delegation.base_url` / `api_key` / `model` — `delegate_task` ignores per-agent
  `model:` frontmatter, so this is the **only** place every child
  (hypothesis-validator and code-generator alike) gets its model.

The Loam-specific keys the example already contains, and what each one does:

```yaml
plugins:
  enabled: [loam-hooks]              # (1) activates the lifecycle hooks
mcp_servers:
  hpc:
    url: http://127.0.0.1:8000/mcp   # (2) connects to the server from Step 2
    timeout: 300                     # a wedged call blocks the agent loop this
    connect_timeout: 60              # long; prefer run_code's own `timeout`
  hpcctl:                            # optional: start/stop/status for the
    command: /ABS/PATH/.loam-venv/bin/python   # allocation, via hpc_control.py
    args: [/ABS/PATH/loam/skills/hpc-deployment/scripts/hpc_control.py]
delegation:
  max_spawn_depth: 2                 # coordinator → validator → code-generator
  orchestrator_enabled: true
  max_concurrent_children: 5
```

`hpc` is the one the campaign needs; `hpcctl` is a convenience that lets the
agent query or cycle the allocation itself instead of you doing it in Step 2's
shell.

`plugins.enabled: [loam-hooks]` is what makes the hooks fire. The hooks live in
`.loam/hooks/` (`pre-llm.py` runs the tournament every `pre_llm_every` turns;
`subagent-start.py`/`subagent-stop.py` write the orchestration and campaign
ledgers) and are loaded through the adapter at
`.loam/adapters/hermes/plugin/__init__.py`. `run-campaign` auto-generates the
`.hermes/plugins/loam-hooks/` discovery shim, so you do not create it by hand.

See `.loam/README.md` for the full hook contract and the global-install option.

> **Credentials are not in this file** — they live in `auth.json`, which is
> user-provided and gitignored; create it with `hermes` login, and never commit a
> filled-in one.

## Step 4 — The Loam profile (`~/.hermes/loam.yaml`)

Copy `examples/loam.yaml.example` to `~/.hermes/loam.yaml` and edit every
`<PLACEHOLDER>`. One active profile defines the tournament and the execution
contract. Key blocks:

- `hypothesis_tournament` — `jnana_root`, the generation endpoint (`:51900`) and
  ranking endpoint (`:51911`), `initial_hypotheses`, `matches_per_run`, `top_k`,
  `json_max_tokens` (reasoning models need headroom — 16384 for inkling-bf16).
- `hypothesis_orchestration.enabled: true` — makes the harness act as the
  campaign orchestrator.
- `execution:` — `mode: remote_mcp`, `tool_name: mcp__hpc__run_code`, and the
  `setup:` recipe handed verbatim to every code-generator child (venv
  activation + the exact Chai/MPNN command lines). The child runs on Aurora and
  cannot see local files, so this recipe must be self-contained.
- `campaign:` — `dir`, `goal`, `rules`, and `skills`. See Step 4b. `skills` names
  the skills a code-generator must consult before writing code; they are names
  from the Hermes skill library, not files under `dir`. Omit it and children
  write from recollection, which is what it exists to stop.

## Step 4b — Write the campaign

Loam ships no campaign content: no goal, and no rule naming a measurement.
**Start from `examples/campaign-starter/`** — a copyable skeleton of this step
and Step 4, with a template goal, two rules scoped to different tiers, and a
collaborator checklist:

```bash
cp -r <loam-repo>/examples/campaign-starter ~/campaigns/my-campaign
```

In a directory you own — the one `campaign.dir` points at — you need:

```
<campaign.dir>/campaigns/<goal>.md   # objective, success criteria,
                                     # constraints, the decision it informs
<campaign.dir>/rules/<name>.md       # rules specific to this campaign
```

Both file kinds carry `---` front matter. The goal's names itself — `name:` must
equal the filename and `campaign.goal`, so renaming the goal means editing two
other places — and a rule adds `scope:`.

The goal reaches all three tiers. A rule is policy injected into the tiers its
`scope:` front matter names; often a few lines is enough — which measurement
decides a hypothesis, or which hypotheses and tool categories are out of bounds.
Names are resolved in `campaign.dir` first, then in the repo's `.loam/`, so you
can shadow a shipped rule with your own text.

A named-but-missing file **raises**, listing every directory searched: a run that
quietly lost its objective while still looking governed is the failure this
avoids. It raises on the **first turn**, not at launch — the hook is what opens
these files, so the preflight passing says nothing about them. The starter kit's
README has a one-liner that forces the check before you spend an allocation. A campaign with no rules of its own still runs governed by the invariant
rules the repo ships (`evidence-integrity`, `experiment-discipline`), which load
whether or not a profile names them.

Everything past the front matter in these files is injected into agent context
verbatim — only the front matter is stripped. So keep both files short, since
they are injected every turn, and leave no editorial notes in them: an HTML
comment survives, and a rule's body arrives labelled as authoritative over any
conflicting instruction.

Not symmetrical with `campaign.skills`, which is only shape-checked: a
`<PLACEHOLDER>` there raises, but a well-formed name for a skill that is not
installed does not — that one fails inside a code-generator mid-round. Confirm
your names with `hermes skills list`.

See `.loam/rules/README.md` for the rule layers, the front matter, and what a
protocol rule has to answer.

## Step 4c — Optional features (workspace, provenance, Weights & Biases)

Three capabilities are off or external by default. None has a single global
switch; each is enabled where the thing it governs already lives.

### Hypothesis workspace

A Pando-style folder per hypothesis — `experiment.json`, a hash-chained
`log.jsonl`, `src/`, `runs/rM/`, `outputs/rM/`, `data/` — written and enforced
by `.loam/hooks/hypothesis_workspace.py`. Full layout: `.loam/README.md`,
"Hypothesis workspace". Turn it on in the profile:

```yaml
    workspace:
      enabled: true        # the switch; anything but true leaves it off
      strict: true         # a pre-tool hook BLOCKS a layout violation; false only logs it
      # trace_full: false  # provenance verbosity, below
      # verifier_python: /ABS/PATH/.loam-venv-312/bin/python  # hashes remote outputs; needs fastmcp
      # verifier_mcp_url: http://127.0.0.1:8000/mcp
      # inputs_dir / remote_inputs_dir / shared_inputs / max_input_bytes — campaign inputs
```

It requires `hypothesis_orchestration.enabled: true`: without orchestration no
validator is dispatched, so no folder is ever created. Off (the default), a run
leaves the campaign ledger and tournament reports but no per-hypothesis record.
`examples/loam.yaml.example` annotates every sub-key.

### Provenance tracing

The decision log, each run's `trace.jsonl`, and the reasoner→validator→
code-generator dispatch edges are **part of the workspace** — they are produced
whenever it is on, and there is no separate switch that keeps the folders while
dropping the trace. The one knob is verbosity:

```yaml
    workspace:
      enabled: true
      trace_full: true   # keep every tool result verbatim in the captured trace
```

`trace_full: false` (the default) strips the bodies of skill reads from the
trace: they are already content-hashed under each run's `skills`, so the trace
stays a record of what was read rather than a second copy of the skill library.
Set it true only when you need those bodies inline.

Independently of the workspace, every run manifest records the campaign-policy
digest — the SHA-256 of the goal and rule files, by name. So a result always
says which objective and rules governed it, even with the workspace off; that
part is always on and not configurable.

### Weights & Biases

W&B is not a Loam feature and has no profile key — Loam imports no `wandb` and
exposes no switch. Tracking happens (or not) inside the code a code-generator
runs, so you toggle it the standard W&B way, at the layer that owns the run:

- Authenticate once on whatever machine will log — `wandb login`, or set
  `WANDB_API_KEY`.
- Turn logging on or off per run with the `WANDB_MODE` environment variable:
  `online` logs to the server, `offline` records locally to sync later, and
  `disabled` makes every `wandb` call a no-op. `wandb.init(mode=...)` in the
  code does the same.
- Code-generators run on the HPC node with the `execution.setup:` recipe handed
  to them verbatim, so that recipe is where a campaign sets W&B: export
  `WANDB_MODE` (and `WANDB_API_KEY` / `WANDB_PROJECT`) in the bash activation,
  and the generated code that calls `wandb.init()` inherits it. Naming the
  bundled `weights-and-biases` skill in `campaign.skills` points the generators
  at its usage notes (`hermes skills list` shows the skill).

With nothing exporting `WANDB_MODE` and no code calling `wandb`, there is no
tracking — the default. `WANDB_MODE=disabled` is the explicit off switch for
when a skill or script would otherwise open a run.

## Step 5 — Launch a campaign

```bash
mkdir -p ~/campaigns/my-campaign
cp <loam-repo>/run-campaign ~/campaigns/my-campaign/
cd ~/campaigns/my-campaign && ./run-campaign
```

The **directory name is the campaign name**. `run-campaign` gives each campaign
its own isolated Hermes home (symlinks shared creds/skills, copies the config,
keeps memory per-campaign so campaigns cannot prime one another), runs the
fail-closed preflight into `state/preflight.json`, then `exec hermes`. If the
HPC server is not up it will start a bounded one-node debug allocation and wait
up to `LOAM_HPC_WAIT_SECONDS` (default 900) — but for a real run, have Step 2
already live.

**`run-campaign`'s header is the reference for the environment**: every variable
it reads, the ones it passes through to the preflight (`ALCF_PROXY*`,
`HPC_ENV_FILE`), and — the part worth reading before you try to override
anything — the ones it deliberately discards, so that no parent shell can
redirect or merge a run. Two in particular are not knobs: the tournament cadence
comes from the profile's `pre_llm_every`, not `LOAM_PRE_LLM_EVERY`, and the
active profile from `active_profile` in `loam.yaml`, not `LOAM_PROFILE`.

### Validate the plumbing without a cluster first

Run the five-round arithmetic campaign in `CAMPAIGN_TEST.md` with
`execution.mode: local`. It exercises the whole
hook → validator → code-generator → reasoner → next-round path deterministically
and is the fastest way to confirm the hooks and delegation are wired correctly
before you spend an allocation.

## Step 6 — Drive the campaign

`run-campaign` ends in `exec hermes "$@"`, so a successful launch drops you at an
interactive Hermes prompt. **Nothing has happened yet.** The pre-LLM hook fires
on a `pre_llm_call` event, so until you take a turn there is no tournament, no
validators, and no campaign.

### What to type

Almost anything that produces a turn. The campaign's content does not come from
your message — the hook injects the goal, the rules, and the round's scheduler
directive on its own — so the opening prompt only has to start the loop:

```
Begin the campaign.
```

Do **not** restate the objective. The goal file already reaches every tier as
`CAMPAIGN GOAL (what this experiment serves)`, and a retyped-from-memory
objective in the first message is a second, unversioned copy that can contradict
the file without the provenance record showing it.

`run-campaign` forwards its arguments to `hermes`, so `./run-campaign -z "Begin
the campaign."` takes exactly **one** turn and exits — and that turn cannot
dispatch a round, because `loam_dispatch_round` has no parent-agent context
under `-z` and a top-level delegation is backgrounded in any case. It is a
diagnostic, not an unattended launch: see *Running it unattended* below.

### What advances a round

One scheduler step per turn. Each turn the orchestration hook reads the campaign
ledger, works out where the round is, and injects the next action:

| Turn | Injected directive | What the parent does |
|---|---|---|
| 1 | Tournament runs, then: *N validators planned* | calls `loam_dispatch_round` |
| 2 | (validators still running → nothing) | waits |
| 3 | all validators complete → *N candidates pending review* (only with `evidence_review: true`) | dispatches the reviewers |
| 4 | reviewers complete → reasoner directive | synthesizes the brief |
| 5 | brief steers a fresh tournament | next round begins |

So a round is roughly three to four turns, and you advance it by giving Hermes
another turn — `continue` is enough. The directive is self-contained and the
selected hypotheses deliberately never enter the parent's context in full, so do
not compose a `delegate_task` batch by hand: a batch you write is a substitute
for the campaign's work rather than the campaign's work.

### Running it unattended

Copy `drive-campaign` next to `run-campaign` in the campaign directory and run
it **instead of** the launcher — it starts the campaign itself:

```bash
./drive-campaign --max-rounds 4 --max-hours 8
```

It polls the round phase and types `continue` only when a turn will do
something, which matters because every coordinator turn re-runs the Jnana
tournament before the orchestration hook even looks at the phase — a
timer-driven loop pays minutes of ranking to be told the validators are still
running. `--help` lists the budgets and stop conditions: a round or wall-clock
budget, a hard cap on nudges, a phase that will not advance, an HPC pool it
cannot restore, a repeated reasoner brief, and consecutive rounds that accept no
new evidence.

Three things to know about it:

- **It drives a real terminal, and it has to.** A campaign cannot be driven by
  `hermes -z`: `loam_dispatch_round` fails there with *"delegate_task requires a
  parent agent context"* (oneshot never sets the plugin manager's `_cli_ref`),
  and a top-level delegation is backgrounded in any case, so the turn would exit
  while its validators ran. The driver therefore runs the launcher in a
  pseudo-terminal and keeps one session alive across rounds.
- **It waits for quiet before typing.** Because delegation is asynchronous, the
  agent usually takes its own next turn when a child's result arrives. A nudge
  goes in only when the phase is actionable, the session has been silent for
  `--idle-seconds`, and `--min-send-gap` has passed.
- **It runs the preflight's real placement probe before any turn that dispatches
  work**, because a listening `:8000` is not evidence of a usable pool. On
  failure it lets the preflight restart what is missing, which is also how it
  survives a walltime boundary.

The session transcript lands in `state/driver/session-<stamp>.log` and every
decision in `state/driver/driver.jsonl`. Set `LOAM_NOTIFY_CMD` to a push command
(the driver appends the message as a final quoted argument) if nobody will be
reading the terminal:

```bash
LOAM_NOTIFY_CMD='curl -s https://ntfy.sh/<your-topic> -d' ./drive-campaign
```

Unattended acceptance needs `hypothesis_orchestration.evidence_review: true` in
the profile. Without it a round's claims stay `pending` unless a person writes
the review JSONL by hand, so the campaign measures things that never reach
steering. See [`.loam/reconcile/README.md`](.loam/reconcile/README.md#who-authors-the-review).

Watch progress in `<campaign-dir>/state/` — `hypothesis-campaign.jsonl` is the
orchestration ledger, `tournament-reports/` holds each round's ranking, and
`hypothesis-memory.jsonl` holds the accepted, source-backed findings once a
round's claims have been reviewed and reconciled.

---

## New-user traps (all hardcoded to a prior user — change before first run)

| File | Setting to change |
|---|---|
| `run-campaign` | `LOAM_REPO` default; `PREFLIGHT_PYTHON` (`~/.loam-venv-312`) |
| `scripts/campaign_preflight.py` | `ALCF_PROXY_PYTHON` default (`~/.loam-venv-312`), proxy port and log path |
| `.loam/README.md` global shim | `LOAM_ROOT` default (points at another user's checkout) |
| `~/.hermes/loam.yaml` | `jnana_root`, all `state_dir` paths, the `:51900` / `:51911` endpoints and model ids |
| `skills/hpc-deployment/set_env_akv.sh` | `HPC_GLOBUS_ENDPOINT_ID`, `HPC_BASE_DIR`, `HPC_CAMPAIGN_ID`, `HPC_INSPECT_ROOTS`, and the venv/paths inside `worker_init` of `HPC_ENDPOINT_CONFIG_JSON` |
| `.hermes/.loam-home/config.yaml` | Argo/Minerva provider base URLs, API keys, and model ids |

Two of these overlap with findings in the code review and are worth calling out
to a newcomer specifically:

- **`HPC_ENDPOINT_CONFIG_JSON` pins `queue: debug` and `walltime: 00:30:00`.**
  Fine for a smoke test; on a longer campaign you will hit a walltime/ManagerLost
  wall. Change the queue and walltime for real runs.
- **The preflight is the real readiness contract, not the config comments.** If
  it passes, all three subsystems are genuinely talking. If it fails, read
  `state/preflight.json` — it names the model role or the Aurora probe that
  failed.

## Where things live

| Thing | Path |
|---|---|
| Lifecycle hooks | `.loam/hooks/` |
| Hermes adapter (loads the hooks) | `.loam/adapters/hermes/plugin/__init__.py` |
| Agent role prompts | `.loam/agents/{reasoner,hypothesis-validator,code-generator}.md` |
| HPC MCP server + pool | `skills/hpc-deployment/scripts/{mcp_server,hpc_agent,pool_manager}.py` |
| Allocation control plane | `skills/hpc-deployment/scripts/hpc_control.py` |
| Per-campaign env | `skills/hpc-deployment/set_env_akv.sh` |
| Launcher + preflight | `run-campaign`, `scripts/campaign_preflight.py` |
| Invariant rules (shipped) | `.loam/rules/` |
| Profile config | `~/.hermes/loam.yaml` |
| Campaign goal + its own rules | `<campaign.dir>/{campaigns,rules}/` (yours) |
| Per-campaign state (ledgers, manifest, Jnana memory) | `<campaign-dir>/state/` |
