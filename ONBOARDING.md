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
  judge. `run-hermes`/`run-campaign` auto-start this if nothing is listening; on
  an auth error run `alcf-tokens login inference`.

## Step 2 — Bring up the HPC MCP server ("the hpc thing")

This is the `hpc-deployment` skill. From that skill's SKILL.md §Deployment:

1. **Edit `skills/hpc-deployment/set_env_akv.sh`** (or your own copy of
   `set_env.sh`) — the single definition point. See the hardcoded-value table
   below for everything a new user must change.
2. `source skills/hpc-deployment/set_env_akv.sh` — **every later step runs in
   that same shell**; the server reads its configuration from the process
   environment and nowhere else.
3. `python scripts/preflight.py` — validates local deps and worker prerequisites
   without submitting a job.
4. `python scripts/mcp_server.py` — starts the local FastMCP server on `:8000`,
   creates the Globus Compute Executor with your endpoint UUID, submits the PBS
   allocation, and launches the Academy agents. In **cluster mode** that is one
   `HpcClusterAgent` running the Ensemble Launcher cluster plus `HPC_POOL_SIZE`
   client agents that submit into it; EL owns tile placement.
5. `python scripts/smoke_test_client.py` — **do not trust the server until this
   passes.** It requires a real `run_code` result and bounded
   `inspect_filesystem` output from the worker.
6. Submit a couple of concurrent cheap calls and confirm they *overlap* rather
   than serializing.

**Shut it down only with SIGINT/SIGTERM** (`Ctrl-C` / `kill -INT`), never
`kill -9`: a clean exit is what runs `Manager.close()` and releases the Aurora
allocation. `kill -9` orphans the EL cluster until walltime.

## Step 3 — Wire MCP + hooks into the Hermes config

In the Hermes home's `config.yaml` (the shared home is
`.hermes/.loam-home/config.yaml`):

```yaml
plugins:
  enabled: [loam-hooks]              # (1) activates the lifecycle hooks
mcp_servers:
  hpc:
    url: http://127.0.0.1:8000/mcp   # (2) connects to the server from Step 2
    timeout: 300
    connect_timeout: 60
delegation:
  max_spawn_depth: 2                 # coordinator → validator → code-generator
  orchestrator_enabled: true
  max_concurrent_children: 5
```

`plugins.enabled: [loam-hooks]` is what makes the hooks fire. The hooks live in
`.loam/hooks/` (`pre-llm.py` runs the tournament every `pre_llm_every` turns;
`subagent-start.py`/`subagent-stop.py` write the orchestration and campaign
ledgers) and are loaded through the adapter at
`.loam/adapters/hermes/plugin/__init__.py`. `run-campaign` auto-generates the
`.hermes/plugins/loam-hooks/` discovery shim, so you do not create it by hand.

See `.loam/README.md` for the full hook contract and the global-install option.

> **Config templates.** Copy `examples/hermes-config.yaml.example` into your
> Hermes home as `config.yaml` and fill in the `<PLACEHOLDER>`s. **Credentials
> are not in this file** — they live in `auth.json`, which is user-provided and
> gitignored; create it with `hermes` login, never commit a filled-in one.

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
- `campaign:` — `dir`, `goal`, `rules`. See Step 4b.

## Step 4b — Write the campaign

Loam ships no campaign content: no goal, and no rule naming a measurement. In a
directory you own — the one `campaign.dir` points at — write:

```
<campaign.dir>/campaigns/<goal>.md   # objective, success criteria,
                                     # constraints, the decision it informs
<campaign.dir>/rules/<name>.md       # rules specific to this campaign
```

The goal reaches all three tiers. A rule is policy injected into the tiers its
`scope:` front matter names; often a few lines is enough — which measurement
decides a hypothesis, or which hypotheses and tool categories are out of bounds.
Names are resolved in `campaign.dir` first, then in the repo's `.loam/`, so you
can shadow a shipped rule with your own text.

A named-but-missing file **raises**, listing every directory searched: a run that
quietly lost its objective while still looking governed is the failure this
avoids. A campaign with no rules of its own still runs governed by the invariant
rules the repo ships (`evidence-integrity`, `experiment-discipline`), which load
whether or not a profile names them.

See `.loam/rules/README.md` for the rule layers, the front matter, and what a
protocol rule has to answer.

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

### Validate the plumbing without a cluster first

Run the five-round arithmetic campaign in `CAMPAIGN_TEST.md` with
`execution.mode: local`. It exercises the whole
hook → validator → code-generator → reasoner → next-round path deterministically
and is the fastest way to confirm the hooks and delegation are wired correctly
before you spend an allocation.

---

## New-user traps (all hardcoded to a prior user — change before first run)

| File | Setting to change |
|---|---|
| `run-campaign` | `LOAM_REPO` default; `PREFLIGHT_PYTHON` (`~/.loam-venv-312`) |
| `run-hermes` | `ALCF_PROXY_PYTHON`, proxy port/log paths |
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
