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

- A Python environment for Loam (any venv/conda env, py3.12): `pip install -e .`
  in this repo, plus `fastmcp`, `pyyaml`, `aiohttp`, `alcf-tokens`. Its
  `globus-compute-sdk` must match your endpoint's `globus-compute-endpoint`
  version (4.19.0 on Aurora today), and `academy-py` must be 1.0.0 to match the
  worker venv (it needs `globus-sdk>=4`, so upgrade `proxystore` to 1.x
  alongside it). `fastmcp` 4.x pulls `starlette` 1.x; upgrade `fastapi` if pip
  reports a conflict.
- In the same environment, `ensemble_launcher` at the worker's commit:
  `pip install "ensemble_launcher @ git+https://github.com/argonne-lcf/ensemble_launcher.git@c36dc43"`.
  `HpcClusterAgent` is constructed locally and then shipped to the cluster, so a
  missing install surfaces only at server start as `NameError: aurora_config`.
  Keep `pydantic` and `cloudpickle` at the worker venv's versions (2.13.4 and
  3.1.2 today); these objects are serialized between the two sides.
- One interactive Globus login for the Academy exchange, before the first
  `mcp_server.py` start (it cannot prompt when run in the background):
  `python -c "from academy.exchange.cloud.login import get_auth_headers; get_auth_headers('globus')"`.
- A Jnana checkout: `git clone -b feature/periodic-hypothesis-tournament
  https://github.com/acadev/jnana.git` (`main` lacks
  `protognosis/periodic_tournament.py`). Do **not** `pip install` it — its
  `setup.py` requires an unreleased `biomni>=1.0.0`. The tournament hook puts
  `jnana_root` on `sys.path` instead.
- **Your launcher settings** in a `loam.env` file: copy
  `examples/loam.env.example` and set `LOAM_REPO` and `LOAM_PYTHON`.
  `run-campaign` reads the campaign directory's `loam.env`; `run-hermes` reads
  the gitignored `loam.env` at the repo root. `LOAM_ENV_FILE` names another file;
  the process environment wins. Nothing in the repo hardcodes a per-user path.
- The `hermes` CLI, with a model provider configured.
- A **Globus Compute endpoint running on the HPC login node that your Globus
  identity may submit to.** Another user's endpoint returns
  `403 ENDPOINT_ACCESS_FORBIDDEN`. Register your own: on the login node, activate
  a venv with `globus-compute-endpoint` (see the worker venv in
  `set_env.example.sh`), run `globus-compute-endpoint configure <name>`, copy
  `globus_compute_endpoint_configs/alcf_aurora/user_config_template.yaml.j2`
  into `~/.globus_compute/<name>/`, then `globus-compute-endpoint start <name>`
  and note the **manager UUID**. See the `hpc-deployment` skill's "Endpoint
  discovery and flex-scaling validation" section to verify it.

## Step 1 — Inference endpoints

Two OpenAI-compatible bases the campaign talks to:

- **Argo tunnel on `:51900`** — the Hermes reasoner and every delegated subagent
  (`argo:gpt-5.6-sol`). Bring up this SSH tunnel yourself.
- **`scripts/alcf_inference_proxy.py` on `:51911`** (Minerva) — Jnana's ranking
  judge. `run-hermes`/`run-campaign` auto-start this if nothing is listening; on
  an auth error run `alcf-tokens login inference`.

## Step 2 — Bring up the HPC MCP server ("the hpc thing")

This is the `hpc-deployment` skill. From that skill's SKILL.md §Deployment:

1. **Create `skills/hpc-deployment/set_env.local.sh`** (gitignored) from
   `set_env.example.sh` and fill in every `<PLACEHOLDER>` — the single
   definition point. `set_env_akv.sh` is a filled-in example for reference.
2. `source skills/hpc-deployment/set_env.local.sh` — **every later step runs in
   that same shell**; the server reads its configuration from the process
   environment and nowhere else.
3. `python scripts/preflight.py` — validates local deps and worker prerequisites
   without submitting a job.
4. `python scripts/mcp_server.py` — starts the local FastMCP server on `:8000`,
   creates the Globus Compute Executor with your endpoint UUID, submits the PBS
   allocation, and launches the Academy agents. In **cluster mode** that is one
   `HpcClusterAgent` running the Ensemble Launcher cluster plus `HPC_POOL_SIZE`
   client agents that submit into it; EL owns tile placement.
5. `HPC_MCP_URL=http://127.0.0.1:8000/mcp python scripts/smoke_test_client.py` —
   **do not trust the server until this passes.** It requires a real `run_code`
   result and bounded `inspect_filesystem` output from the worker. `set_env.*.sh`
   does not export `HPC_MCP_URL`, so pass it on the command line or the client
   exits with `set HPC_MCP_URL, for example ...`.
6. Submit a couple of concurrent cheap calls and confirm they *overlap* rather
   than serializing.

**Run the server under `tmux`/`screen`, not a bare terminal.** It has to outlive
the whole campaign; a closed window, a dropped VPN or a sleeping laptop kills it,
and that is not a clean exit, so the Aurora allocation is orphaned until
walltime. A detached session also keeps its log readable after the fact:

```bash
tmux new -s hpc
source skills/hpc-deployment/set_env.local.sh
python skills/hpc-deployment/scripts/mcp_server.py
# Ctrl-b d to detach; `tmux attach -t hpc` to read the log
```

**Shut it down only with SIGINT/SIGTERM** (`Ctrl-C` / `kill -INT`), never
`kill -9`: a clean exit is what runs `Manager.close()` and releases the Aurora
allocation. `kill -9` orphans the EL cluster until walltime. A clean stop takes
~20s — wait for the process to disappear before restarting.

**The port binds last.** The FastMCP banner prints immediately, but `:8000` only
accepts connections after the lifespan has built the EL cluster, which waits on
the PBS block to start and boot. Two to three minutes from cold is normal; a
restart while a block is still alive takes seconds. Until then `curl` fails with
exit 7 and an HTTP code of `000`, which means *no connection*, not a server
error. Alive is `400`.

## Step 3 — Wire MCP + hooks into the Hermes config

In the Hermes home's `config.yaml` (the shared home is
`.hermes/.loam-home/config.yaml`, which is gitignored — create it from
`examples/hermes-config.yaml.example`):

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

## Step 5 — Launch a campaign

```bash
mkdir -p <campaigns-dir>/my-campaign && cd <campaigns-dir>/my-campaign
cp <loam-repo>/run-campaign .
cp <loam-repo>/examples/loam.env.example loam.env      # set LOAM_REPO, LOAM_PYTHON
cp <loam-repo>/examples/loam.yaml.example loam.yaml    # this campaign's profile
caffeinate -dis ./run-campaign                         # macOS: see below
```

**On macOS, launch under `caffeinate -dis`.** A campaign runs for hours; if the
machine sleeps, the SSH tunnel to the inference endpoint dies with it, every
in-flight subagent loses its model connection, and Hermes' no-progress timeout
interrupts the round ~20 minutes later. The lifecycle records show
`status=interrupted` with no evidence score, and the round does not count.

The **directory name is the campaign name**. Its `loam.env` and `loam.yaml` are
the campaign's own settings and profile (`loam.yaml` takes precedence over
`~/.hermes/loam.yaml`), so every campaign is self-contained. `run-campaign` gives each campaign
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

### Driving the coordinator

- **Paste the task once per round, as a single line.** Every message you send is
  another coordinator turn, and each coordinator turn advances the Jnana
  tournament — so a stray "ok" or a prompt split across two sends costs an extra
  tournament and can double-dispatch validators. Task prompts in the test docs
  are wrapped to 80 columns; join them into one line before pasting.
- **Do not type while a round is running.** After the validators are dispatched
  the coordinator resumes by itself when they finish, dispatches the reasoner,
  and then starts the next round unprompted. Silence in the UI is normal.
- **`Initializing agent...` is the pre-LLM hook, not a hang.** It blocks on Jnana
  generation plus ranking before the first token: ~seconds at
  `initial_hypotheses: 2`, several minutes at 25. Bounded by the profile's
  `timeout_seconds`.
- **Watch `state/`, not the UI.** `hypothesis-orchestration.jsonl` and
  `tool-calls.jsonl` are written continuously while the terminal shows nothing;
  they are the honest progress signal.

### Known hiccups

- **`run-campaign: LOAM_REPO is not set.`** The per-user launcher settings file
  does not exist yet. Create `~/.config/loam/env` from
  `examples/loam.env.example` (or set `LOAM_ENV_FILE`).
- **Hook directives silently truncated.** Loam's `pre_llm_call` directives run
  20-40 KB; Hermes spills any hook output over `hooks.output_spill.max_chars`
  (default 10 KB) to a file and leaves only a path in the prompt, which the model
  does not read. The visible symptom is the coordinator answering the question
  itself instead of dispatching the reasoner. `examples/hermes-config.yaml.example`
  raises the limit — keep that setting in your Hermes home's `config.yaml`.
- **`set_env_akv.sh` is not yours.** It is Archit's filled-in example: his
  endpoint UUID and `HPC_BASE_DIR=/flare/FRAME-IDP/avasan/...`, which other users
  cannot write. Copy `set_env.example.sh` to `set_env.local.sh` instead; the
  launchers default `HPC_ENV_FILE` to that path.
- **`globus-compute-endpoint list` crashes** with
  `AttributeError: 'Texttable' object has no attribute '_width'` — a texttable
  incompatibility in the table renderer, not a sick endpoint. Read
  `~/.globus_compute/<name>/endpoint.log` and `daemon.pid` instead. Do not
  `pip install` a fix into a shared worker venv.
- **The endpoint daemon lives on one login node.** `~/.globus_compute/` is on
  shared home, so its config is visible from every UAN while the process runs on
  whichever node started it. Check from that node before concluding it is down.
- **A long run needs the allocation knobs.** `min_blocks: 1` is what makes the
  provider request a replacement node when a block hits walltime; without it the
  cluster simply ends. `max_blocks: 2` lets the replacement overlap the expiring
  block, at the cost of a second concurrent job (queue policy permitting).
- **Concurrency is bounded by `HPC_POOL_SIZE`.** `top_k` validators times their
  code-generator children can exceed it. Children then queue, time out, and three
  consecutive timeouts open Hermes' circuit breaker — which is process-global per
  server name, so one child's queueing convinces every other child the endpoint
  is dead. A storm of "endpoint unreachable" against a healthy endpoint means
  raise the pool, not restart the server.

---

## Per-user files (none are tracked — create each from its template)

| File | Template | What goes in it |
|---|---|---|
| `<campaign>/loam.env` (or repo-root `loam.env` for `run-hermes`) | `examples/loam.env.example` | `LOAM_REPO`, `LOAM_PYTHON`, optional `HPC_ENV_FILE` / proxy overrides |
| `<campaign>/loam.yaml` (or repo-root `loam.yaml`) | `examples/loam.yaml.example` | `jnana_root`, `state_dir` paths, generation/ranking endpoints and model ids |
| `.hermes/.loam-home/config.yaml` | `examples/hermes-config.yaml.example` | reasoner/subagent provider base URLs and model ids, `hpcctl` paths |
| `skills/hpc-deployment/set_env.local.sh` | `skills/hpc-deployment/set_env.example.sh` | endpoint UUID, `HPC_BASE_DIR`, campaign id, inspect roots, worker venv |

Your local ports may differ from the examples (e.g. an Argo tunnel on another
port); keep them consistent across these files.

Two points are worth calling out to a newcomer specifically:

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
| Per-user HPC env | `skills/hpc-deployment/set_env.local.sh` (from `set_env.example.sh`) |
| Per-campaign launcher env and profile | `<campaign>/loam.env`, `<campaign>/loam.yaml` |
| Launcher + preflight | `run-campaign`, `scripts/campaign_preflight.py` |
| Profile config | `<campaign>/loam.yaml` (fallback `~/.hermes/loam.yaml`) |
| Per-campaign state (ledgers, manifest, Jnana memory) | `<campaign-dir>/state/` |
