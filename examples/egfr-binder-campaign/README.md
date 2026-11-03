# EGFR binder-design campaign

A complete, as-run Loam campaign: de novo protein binders against human EGFR
(UniProt P00533), scored through a design funnel on Aurora.

Unlike `../campaign-starter`, this is **not** a template. The values are the ones
msinclair actually ran on 2026-10-30, so you can diff against a working
configuration instead of guessing. That also means most of them are wrong for
you — see [What you must change](#what-you-must-change).

**Read this whole file before launching.** The campaign directory is the small
part. Everything that cost real time lives outside it: the Globus Compute
endpoint template, the on-node venv, and the Aurora software release.

## What is here

| File | Goes to | Notes |
|---|---|---|
| `loam.yaml` | `~/.hermes/loam.yaml` | Or merge its `campaign1:` block into your existing `profiles:` map. Not in the campaign dir. |
| `campaigns/binder-design.md` | stays | `campaign.dir` points at this directory. |
| `rules/*.md` | stays | Same. |
| `launch.sh` | stays | Sets the launch environment and refuses to start if a precondition fails. |
| `worker_check.sh` | copy to `$base_dir` on `/flare` | Reproduces the endpoint's `worker_init` and asserts the worker import chain. Run it after any env change. |
| — | this directory | Copy `<loam-repo>/run-campaign` here. |
| — | `<loam-repo>/.hermes/.loam-home/config.yaml` | Shared Hermes home. See [Hermes config](#hermes-config). |

The goal and rules are injected **verbatim into agent context on every turn**, and
rules arrive labelled as authoritative. Keep them short, and do not leave
editorial notes in them — HTML comments are not stripped and read as policy.

## What you must change

Nothing here is optional. A value left as msinclair's does not fail loudly; it
either writes to a directory you do not own or submits to an endpoint you cannot
reach.

### In `loam.yaml`

| Key | Current | Change to |
|---|---|---|
| `state_dir`, `campaign.dir` | `/Users/msinclair/agent/loam/campaign1` | wherever you copied this directory |
| `hypothesis_tournament.state_file` / `reports_dir` | under that same path | same |
| `jnana_root` | `/Users/msinclair/github/jnana` | your Jnana checkout |
| `inference_endpoint` | `http://localhost:61612/v1` | your own tunnel/proxy port |
| `model` | `argo:gpt-5.6-terra` | a model your endpoint serves |
| `execution.workdir_root` | `/flare/FRAME-IDP/msinclair/hpc-run/work` | your project/user path, and **create it** |
| `execution.setup` | mentions that same path | same |

`state_file` and `reports_dir` look redundant next to `state_dir` and are not.
`_tournament_config` is handed the `LOAM_STATE_DIR` override, **not** the
profile's `state_dir`; with that env var unset these two are the only source for
the paths, and without them the entire `hypothesis_tournament` block returns
`None` *silently* — the campaign then runs with no tournament at all.
`run-campaign` does export `LOAM_STATE_DIR`, so this bites only on other launch
paths (e.g. a bare `LOAM_CONFIG_PATH`).

### In `launch.sh`

`LOAM_REPO`, and `LOAM_PREFLIGHT_PYTHON` (see [Local venv](#local-venv)).

### In `skills/hpc-deployment/set_env_msinclair.sh`

Copy it to your own name and change `HPC_ACCOUNT`, `HPC_QUEUE`, `HPC_BASE_DIR`,
`HPC_INSPECT_ROOTS`, `HPC_CAMPAIGN_ID`, `HPC_GLOBUS_ENDPOINT_ID`, and `base_dir`
inside `HPC_ENDPOINT_CONFIG_JSON`. Then point `HPC_ENV_FILE` at your copy —
`run-campaign` defaults it to `set_env.sh`, which is a template full of
`FILL_IN` markers that `launch.sh` refuses to run — so a launch that forgets
this variable fails loudly instead of starting an allocation on whoever's
endpoint happened to be committed there.

Two values are guaranteed wrong for you:

- **`HPC_GLOBUS_ENDPOINT_ID=d83212d9-5751-4892-a78a-bd60d7232fc2`** is
  msinclair's personal multi-user endpoint (`flex_scaling`). You need your own,
  and you need its **manager** UUID, not a spawned user-endpoint UUID (the
  `uep.<manager>.<child>` directories under `~/.globus_compute/`). The manager
  UUID is what lets one endpoint provision different allocation shapes per
  submission.
- **`HPC_QUEUE=R8984162`** was a custom reservation (2055 nodes, 4.5 h) that
  **expired 2026-10-30 18:30 UTC**. It is dead. Use your own reservation or a
  standing queue, and add it to `QUEUE_LIMITS` in
  `skills/hpc-deployment/scripts/hpc_control.py` plus `HPC_ALLOWED_QUEUES` and
  `HPC_MAX_NODES` in the Hermes config, or the guardrail rejects it.

## The endpoint template — the part that is easy to get wrong

The campaign's allocation request is passed as `user_endpoint_config`. **The
endpoint decides which of those keys it honours**, via its
`user_config_template.yaml.j2` in `~/.globus_compute/<endpoint>/`.

The `flex_scaling` endpoint used here accepts exactly:

```
account  available_accelerators  base_dir  block_error_handler  display_name
filesystems  ifname  init_blocks  max_blocks  max_workers_per_node  min_blocks
nodes_per_block  place  prefetch_capacity  queue  walltime
```

Consequences, each of which cost time to discover:

1. **`worker_init` is NOT in that list.** The template hardcodes its own and
   never interpolates a client-supplied one, so a `worker_init` key in
   `HPC_ENDPOINT_CONFIG_JSON` is silently discarded — it does not error. Venv
   activation, module loads, `TMPDIR` and proxy exports must live in the
   template.
2. **`base_dir` is the seam that replaces it.** The template does
   `cd <base_dir>` and
   `export PYTHONPATH=<base_dir>/skills/globus-compute-mcp:<base_dir>`.
3. **Workers must import `hpc_agent` and `pool_manager`.** `mcp_server.py`
   imports `HpcAgent`/`HpcClientAgent`/`HpcClusterAgent` and
   `CampaignPool`/`ClusterPool`, and cloudpickle serializes module-level
   functions **by reference** — so `cd` alone is not enough. Copy both from
   `skills/hpc-deployment/scripts/` to your `base_dir` on `/flare`, and keep
   them byte-identical to the local copies or the manager and the workers run
   different code.
4. **`TMPDIR=/tmp` must be first.** PBS sets a ~60-char `TMPDIR`; the worker
   pool's AF_UNIX socket under it exceeds the ~108-char limit and every worker
   dies before registering, surfacing as a bare `EOFError` at the client.
5. **Compute nodes need the ALCF proxy.** The stock template sets none. Without
   it, any task that fetches a sequence, structure or model weight fails with a
   timeout that looks like a tool bug. This campaign's goal file requires
   run-time retrieval from UniProt, so it matters:
   ```bash
   export HTTP_PROXY=http://proxy.alcf.anl.gov:3128   # + HTTPS_ and lowercase
   export NO_PROXY=localhost,127.0.0.1,.alcf.anl.gov  # keeps the exchange off the proxy
   ```
6. **Do not quote `walltime` yourself.** The manager's Jinja env already quotes
   client-supplied strings; a second pair renders `""01:00:00""` →
   `ParserError`. Unquoted is also wrong — YAML 1.1 reads `01:00:00` as
   sexagesimal `3600`. A plain JSON string is exactly right.
7. **`max_workers_per_node` is 12, not `HPC_POOL_SIZE + 1`.** An Aurora node has
   6 Intel Data Center GPU Max × 2 tiles = 12 addressable XPUs, and this
   template pins `available_accelerators` to match. The `POOL_SIZE + 1` formula
   was written for a 2-node block.

**The endpoint must be restarted for a template change to take effect.**

## The on-node venv

Two independent requirements, and both were violated at first.

**Version parity with the local manager.** `academy` speaks a wire protocol and
`dill`/`cloudpickle` deserialize the payloads. A mismatch surfaces as an
unpickling error or a bare `EOFError`, never a version complaint. Verified
working set, identical on both ends:

```
academy-py 1.0.0    globus-compute-sdk 4.19.0    globus-compute-endpoint 4.19.0
dill 0.3.9          cloudpickle 3.1.2            ensemble-launcher 0.1.0
```

`ensemble_launcher` is required, not optional: `mcp_server.py` hardcodes
`create_server(cluster=True, …)` and cluster mode is EL-backed. Install the
pinned commit:

```bash
uv pip install --python <venv>/bin/python \
  "ensemble_launcher @ git+https://github.com/argonne-lcf/ensemble_launcher@c36dc432ecf3d1c8044d64c4bc20b38e82f07c35"
```

**Build it on the CURRENT Aurora frameworks release.** A venv built on an older
release keeps that release's interpreter in its `pyvenv.cfg`, so its
system-site-packages `torch` belongs to a stack whose oneapi/SYCL/MKL runtime is
no longer on the module path. The symptom is
`OSError: libmkl_intel_lp64.so.2: cannot open shared object file`, and forcing
the old MKL onto `LD_LIBRARY_PATH` just cascades to `libsycl.so.8`. Pinning
`module load frameworks/<old>` does **not** rescue it — Lmod silently resolves to
the current default even after `module purge`.

Check with `grep home <venv>/pyvenv.cfg` and compare against the default
`frameworks` module. The worker venv needs only the agent runtime above;
scientific tools have their own venvs, activated per-command inside `run_code`.

## Hermes config

In `<loam-repo>/.hermes/.loam-home/config.yaml`, which `run-campaign` copies into
a per-campaign home at launch:

```yaml
model:                                   # the campaign parent, and subagents
  default: argo:gpt-5.6-terra
  base_url: http://localhost:61612/v1
delegation:
  max_spawn_depth: 2                     # REQUIRED: coordinator -> validator -> code-generator
  max_concurrent_children: 160           # >= top_k, see below
mcp_servers:
  hpcctl:
    command: <a python that can also run mcp_server.py>
    env:
      HPC_ENV_FILE: <your set_env copy>
      HPC_ALLOWED_QUEUES: debug,debug-scaling,capacity,<your queue>
      HPC_MAX_NODES: "1024"
```

`max_concurrent_children` at its default of 5 against `top_k: 150` is the
documented disaster: validators queue on five slots, waiters time out, and three
consecutive timeouts open Hermes' circuit breaker — a **process-global** dict
keyed by server name, so one child's queueing convinces every other child the
server is dead. It produced 46 "endpoint unreachable" reports in one evening
against a healthy endpoint.

## Local venv

`hpc_control.py` launches `mcp_server.py` with `sys.executable`, so the
interpreter that runs the preflight and `hpcctl` must itself import `fastmcp`,
`academy` and `globus_compute_sdk` — not just the stdlib `hpc_control` uses.
Point both `LOAM_PREFLIGHT_PYTHON` and `hpcctl.command` at it. `launch.sh`
asserts this before starting.

## Tournament sizing

`initial_hypotheses: 400`, `matches_per_run: 200`, `top_k: 150` are chosen, not
arbitrary.

- Pairing is `random.sample(hypotheses, 2)` per match, drawn independently from
  the full pool, so matches repeat. 200 matches over 400 hypotheses touches ~253
  distinct ones.
- Every hypothesis starts at Elo `1200.0`, and selection is
  `sorted(..., key=elo, reverse=True)[:k]` with a **stable** sort. So anything
  never matched sits in a large tie block ordered by storage order. **Keep
  `top_k` at or below the expected number of match winners** (≈ `matches_per_run`)
  or most of your validators are selected arbitrarily — and a hypothesis with
  real negative evidence can rank below hypotheses nobody ever looked at.
- Ranking is **serialized**: `run_incremental_tournament` hardcodes
  `max_workers=1`. Measured ~3 s per match, so budget ~5 s. 200 matches ≈ 17 min,
  and `timeout_seconds` must clear that or the hook is killed mid-tournament.
- `initial_hypotheses` doubles as `max_workers` for generation, and is
  **one-shot**: generation is skipped once the state file holds ≥ 2 hypotheses,
  so raising it later does nothing unless you delete `jnana-memory.json`.

On a fixed wide allocation, orchestrator time is spent budget: one minute of a
1024-node block is ~17 node-hours, so a 17-minute serialized tournament idles
~28% of a one-hour block. Parallelizing ranking is the single biggest available
win and needs a code change.

## Verifying before you burn an allocation

```bash
# 1. The profile parses. Does NOT open the goal or rule files.
LOAM_CONFIG_PATH=~/.hermes/loam.yaml python3 -c "
import sys; sys.path.insert(0, '<loam-repo>/.loam/adapters/hermes/plugin')
import loam_config as c; p = c.load_profile()
print(p.campaign.goal, list(p.campaign.rules), p.execution.mode, p.hypothesis_tournament is not None)"
```
A `None` profile means it was **rejected**; `hypothesis_tournament: None` means
the tournament went inert silently. Note the loader needs `pyyaml` — a
stdlib-only interpreter returns `None` for a perfectly good config.

```bash
# 2. The goal and rules actually resolve. Run from the repo root.
cd <loam-repo> && LOAM_CONFIG_PATH=~/.hermes/loam.yaml python3 -c "
import sys; sys.path.insert(0, '<loam-repo>/.loam/adapters/hermes/plugin')
sys.path.insert(0, '<loam-repo>/.loam/hooks')
import loam_config as c, hypothesis_orchestration as o
ctx = o.campaign_context({'loam_campaign': c.payload_values(c.load_profile())['loam_campaign']})
print(ctx.goal_name, [r.name for r in ctx.rules])"

# 3. Every named skill resolves. NOT caught by either check above -- it fails
#    inside a code-generator mid-round. These must be the LEAF names in the
#    FIRST column of `hermes skills list` (`chai`), not the category in the
#    second (`structure-prediction`), which resolves to nothing.
hermes skills list | grep -E 'rfdiffusion3|proteinmpnn|chai|openmm|pyrosetta'

# 4. The worker import chain, on Aurora.
bash $base_dir/worker_check.sh      # expect "AGENT CHAIN OK"

# 5. workdir_root exists and is writable.
ssh aurora 'ls -ld <workdir_root>'
```

## Launching

```bash
cp <loam-repo>/run-campaign .
./launch.sh
```

`launch.sh` deliberately does **not** set `LOAM_SKIP_PREFLIGHT`. On a remote
campaign the gate is what starts the allocation and proves one real `run_code`
round-trip; run-campaign's own docs call skipping it "launcher diagnostics only,
never a scientific campaign."

You are left at a prompt. **The first turn starts round one.**

**Jnana's generation goal is your first message**, not the goal file.
`_generation_goal` returns `payload["user_message"]`, and the tournament hook
never reads `campaigns/binder-design.md` — that file reaches validators and the
reasoner by context injection only. So type the research question:

```
Identify designed protein binders against human EGFR, ranked by computed binding free energy.
```

Typing only "Begin the campaign." seeds every hypothesis against that string.
From round two on, `prepare_payload` appends the prior reasoner brief to your
message automatically. Afterwards one scheduler step happens per turn, so
`continue` advances a round.

**The preflight starts a bounded 1-node `debug` block, not your configured
allocation.** A wide allocation needs `hpc_restart` (the `hpcctl` MCP tool) with
the campaign's queue and node count once the server is up.

Watch `state/hypothesis-campaign.jsonl` for the orchestration ledger,
`state/tournament-reports/` for each round's ranking, and
`state/hypothesis-memory.jsonl` for accepted findings.

## Known-good values, as run 2026-10-30

For diffing only. See [What you must change](#what-you-must-change).

| | |
|---|---|
| campaign dir | `/Users/msinclair/agent/loam/campaign1` |
| ALCF account / queue | `FRAME-IDP` / `R8984162` (reservation, expired) |
| allocation | 1024 nodes × 1 h, `init/min/max_blocks` 1 |
| endpoint | `flex_scaling`, manager `d83212d9-…` (personal) |
| `base_dir` on `/flare` | `/flare/FRAME-IDP/msinclair/loam-hpc` |
| `workdir_root` | `/flare/FRAME-IDP/msinclair/hpc-run/work` |
| on-node venv | `/flare/FRAME-IDP/msinclair/envs/loam` (frameworks 2026.1.0) |
| local manager venv | `<campaign dir>/.venv` (fastmcp 4.1.4) |
| inference | `http://localhost:61612/v1`, `argo:gpt-5.6-terra`, `omit_temperature: true` |

`omit_temperature: true` is required for an Argo reasoning model: Jnana sends
`0.7`, the model rejects `temperature != 1` with HTTP 400, and Jnana **swallows
the per-worker error** — it looks like `generated_hypotheses: 0` with status
`completed`.
