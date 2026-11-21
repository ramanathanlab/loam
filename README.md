<p align="center">
  <img src="logo/v1.png" alt="loam logo" width="200" />
</p>

<h1 align="center">loam</h1>

<p align="center">
  <strong>Long-horizon Orchestration of Agents and Molecules</strong><br/>
  Hypothesis-driven computational science campaigns on <code>pydantic-ai</code>
</p>

---

**loam** is a deterministic engine for hypothesis-driven scientific campaigns, built on [`pydantic-ai`](https://ai.pydantic.dev/). It pairs **Jnana** hypothesis generation and Swiss-system Elo tournaments with parallel hypothesis validation, delegated code execution (local or multi-node HPC), independent evidence review, deterministic ledger reconciliation, and round-over-round campaign steering.

## Why `loam`?

The scientific contracts work; coaxing an LLM orchestrator to enforce them via runtime hooks does not. `loam` keeps the scientific invariants while replacing ~9,000 lines of prompt/hook policing with two structural guarantees:

1. **Plain Python orchestration (no LLM scheduler):** A campaign round is a fixed Python sequence—`Tournament → Validators → Reviewers → Reconcile → Steering → Reasoner`. Models never make scheduling decisions.
2. **Strict typed boundaries & capability-scoped toolsets:** Every agent returns a strict `pydantic` model (`extra='forbid'`), automatically retried by `pydantic-ai` on schema or validation errors. Tool restrictions are enforced by construction—each role is only given the tools it is allowed to call.

---

## Shape of a Round

```text
               goal.md + previous tournament_brief
                                │
                      ┌─────────▼──────────┐
                      │ Tournament (Jnana) │  generate/top-up hypotheses, Swiss Elo rank, select top_k
                      └─────────┬──────────┘
                    ┌───────────┼───────────┐        (asyncio.gather, bounded by concurrency)
              ┌─────▼─────┐┌────▼──────┐┌───▼───────┐
              │ Validator ││ Validator ││ Validator │  owns one hypothesis; plans; inspects outputs
              └─────┬─────┘└───────────┘└───────────┘
                    │ tool: code_generator(instruction)  ──►  Code Generator
                    │                                          (skills + write/exec/batch tools)
                    ▼
              ValidationReport {outcome, evidence_score, ledger, claims[], next_step}
                    │
                    │  claims[] + output paths only (never the validator's prose)
              ┌─────▼─────┐
              │ Reviewer  │  read-only; ordered verdict per claim: accepted / rejected / unverifiable
              └─────┬─────┘
              ┌─────▼──────┐
              │ Reconcile  │  deterministic: accepted + provenance-checked claims → ledger.jsonl
              └─────┬──────┘  (weighted by verification mode, grouped by run or source URL)
              ┌─────▼──────┐
              │ Steering   │  deterministic: Elo ⊕ evidence_rating ⊕ uncertainty → priority table
              └─────┬──────┘  across the entire Jnana hypothesis pool
              ┌─────▼─────┐
              │ Reasoner  │  no tools; synthesizes threads + steering table into Brief
              └─────┬─────┘
                    ▼
          tournament_brief → appended to the next round's Jnana goal
```

### Roles & Contracts

| Role | Configured Model | Allowed Tools | Typed Output |
| :--- | :--- | :--- | :--- |
| **Orchestrator** | — *(plain Python)* | — | `rounds/round-NNN.json` |
| **Tournament** | `tournament.generation`, `tournament.ranking` | — *(Jnana `CoScientist` + Swiss ranker)* | `list[Hypothesis]` |
| **Validator** | `models.validator` | `code_generator`, `read_skill`, `fetch_url`, read-only workspace/MCP tools | `ValidationReport` |
| **Code Generator** | `models.code_generator` | `read_skill`, `fetch_url`, full workspace/MCP execution tools | `CodeResult` |
| **Reviewer** | `models.reviewer` | `read_skill`, `fetch_url`, read-only workspace/MCP tools | `Review` |
| **Reasoner** | `models.reasoner` | *none* | `Brief` |

Each LLM role uses a `ModelSpec` (`{model, base_url, api_key_env, token_refresh}`):
- **With `base_url`:** Connects to any OpenAI-compatible server (vLLM, ALCF/Argo proxy, LiteLLM, etc.).
- **Without `base_url`:** Accepts any native `pydantic-ai` model identifier (e.g., `anthropic:claude-sonnet-4-5`, `openai:gpt-5`, `test`).
- **Secrets stay out of YAML:** API keys are referenced by environment variable name via `api_key_env`; Globus Compute and Academy credentials stay in their standard external stores.
- **ALCF tokens:** `token_refresh: alcf` makes loam fetch a fresh ALCF inference token into that endpoint's `api_key_env` just before round 1, after any allocation queue wait, with `alcf-tokens get-token inference`. It needs the `alcf-auth` extra and a one-time `alcf-tokens login inference`; `ALCF_TOKENS_BIN` can point to another install. Endpoints without the setting are never touched.

With the optional `memory:` block, the validator and code generator also get lessons from earlier work (earlier rounds of this campaign and earlier campaigns) in their prompts; no role gains a tool. See [Memory (optional)](#memory-optional).

---

## Installation

Requires **Python 3.11+** and a local checkout of **Jnana** (referenced via `tournament.jnana_root` in your campaign config).

```bash
# Using uv (recommended)
uv sync                    # core dependencies
uv sync --extra dev        # + pytest for offline test suite
uv sync --extra hpc        # + managed HPC (fastmcp, globus-compute-sdk, academy-py, ensemble-launcher)
uv sync --extra memory     # + optional cross-campaign memory (Hindsight); see "Memory (optional)"
uv sync --extra nemotron   # + optional self-hosted Nemotron inference backend; see "Inference Pool"
uv sync --extra alcf-auth  # + alcf-tokens, for endpoints with `token_refresh: alcf`
uv sync --all-extras       # everything above at once

# Or using pip
pip install -e .
pip install -e ".[dev,hpc,memory]"
```

---

## Quickstart

### 1. Run a Campaign

A campaign is driven by a single YAML configuration file (relative paths resolve against the YAML file's directory):

```bash
python -m loam examples/loam.yaml
```

### 2. Minimal Local Configuration

For local testing or dry runs without an HPC allocation (`execution.mode: local`):

```yaml
goal: campaigns/my-goal.md
rules: []
state_dir: state
skills_dirs: [../skills]
rounds: 3

models:
  validator:      {model: anthropic:claude-sonnet-4-5, api_key_env: ANTHROPIC_API_KEY}
  code_generator: {model: anthropic:claude-sonnet-4-5, api_key_env: ANTHROPIC_API_KEY}
  reviewer:       {model: anthropic:claude-sonnet-4-5, api_key_env: ANTHROPIC_API_KEY}
  reasoner:       {model: anthropic:claude-sonnet-4-5, api_key_env: ANTHROPIC_API_KEY}

execution:
  mode: local
  workdir: work

tournament:
  jnana_root: /path/to/Jnana
  generation: {model: anthropic:claude-sonnet-4-5, api_key_env: ANTHROPIC_API_KEY}
  ranking:    {model: anthropic:claude-sonnet-4-5, api_key_env: ANTHROPIC_API_KEY}
  initial_hypotheses: 6
  new_per_round: 2
  top_k: 3
  scheme: swiss
  swiss_rounds: 3
  workers: 3

concurrency: 3
max_code_calls: 4
```

See [`examples/loam.yaml`](examples/loam.yaml) and [`examples/campaigns/binder-design.md`](examples/campaigns/binder-design.md) for a full managed-HPC EGFR binder design configuration targeting ALCF Aurora (4 nodes / 48 Intel Max 1550 XPU tiles).

### 3. Run the Test Suite

The offline test suite exercises the full multi-round loop, crash recovery, reconciliation gates, steering math, local execution, and MCP tool confinement using scripted `FunctionModel` agents:

```bash
pytest
```

---

## Execution Modes

Configured under `execution.mode`:

- **`local`** — Trusted local execution for laptops and dry runs.
  - Read roles (`validator`, `reviewer`) receive `read_file` and `list_dir` confined to the hypothesis workspace.
  - `code_generator` adds `write_files` (confined to its numbered run directory) and `run_shell` (runs bash in the run directory; process group killed on timeout, **not** sandboxed—do not use for untrusted code).
- **`mcp`** — Connects to an externally managed MCP server at `execution.mcp_url`.
- **`hpc`** — Managed HPC deployment (`aurora`, `polaris` or `jupiter`) via **Globus Compute → the System's scheduler → Academy → Ensemble Launcher**.
  - On a System that allows it (Jupiter), `execution.hpc.gpus_per_node` and `cpus_per_node` override the node shape, and `science_workers_per_gpu: 2` runs two science workers on each GPU. A System that doesn't allow them rejects them when the config loads.
  - `loam` translates `execution.hpc` into the deployment server's internal environment, spawns `mcp_server.py`, waits for cluster pool readiness, and tears down the allocation cleanly on exit (`SIGINT`).
  - No separate environment file or scheduler wrapper (`qsub`/`sbatch`) is needed.

### Site Profiles (`hpc` mode)

What differs between people on the same System lives in a **Site profile**, a YAML file kept outside the repo: the allocation `account` and `queue`, `software_root` (required), `data_root`, `cache_root`, `model_weights`, `inference` endpoints, and a `worker_init` snippet. Name it from the campaign config with `site_profile: <name>`, and loam reads `~/.config/loam/site-profiles/<name>.yaml` (or `$XDG_CONFIG_HOME/loam/site-profiles/`). If the config names none, `LOAM_SITE_PROFILE` may give a profile's path.

- `execution.hpc.allocation.account` and `queue` come from the profile when the campaign config leaves them out; a value set in the config wins.
- The other values reach the deployment environment as `LOAM_SOFTWARE_ROOT`, `LOAM_DATA_ROOT`, `LOAM_CACHE_ROOT`, `LOAM_MODEL_WEIGHTS_JSON` and `LOAM_INFERENCE_JSON`. The HPC agents carry the same set (as `HPC_WORKER_ENV_JSON`) and set it for every script they run on a worker: `run_code`, `run_batch` rows and `register_tool` smoke runs.
- The roots and the profile's `worker_init` snippet (or the campaign config's `endpoint_options.worker_init`, which replaces it) are also sent as the endpoint's `worker_init`. That only takes effect on an endpoint whose template renders `{{ worker_init }}`; the Aurora endpoints do not, and ignore it. Scripts get the `LOAM_*` variables either way, from the agents.
- A missing profile, an invalid one, or an allocation value set in neither place fails when the config loads, naming the profile and the key.

Copy a redacted example from [`examples/site-profiles/`](examples/site-profiles/) (Aurora, Polaris, Jupiter) and fill it in. Never commit a filled-in profile.

### Role-Scoped MCP Tools (`mcp` & `hpc` modes)

Every MCP call is wrapped in `OwnedMCP` (enforcing per-hypothesis workspace isolation and recording batch receipts before returning to the model) and `Resilient` (converting timeouts and tool errors into model-visible tool failures rather than crashing the agent):

- **Read-only (`validator`, `reviewer`, `code_generator`):**
  `inspect_filesystem`, `batch_status`, `list_batches`, `pool_status`, `list_tools`
- **Write & Execute (`code_generator` only):**
  `write_files`, `register_tool`, `run_batch`, `run_code`

HPC campaigns are **`run_batch`-first**: `code_generator` registers reusable stages with `register_tool` (which smoke-tests one row and freezes an `expects` output contract) and fans sweeps across GPU tiles via `run_batch`, reserving `run_code` for harvesting and failed-row diagnostics.

---

## Bundled Skills (`skills/`)

`loam` discovers every `SKILL.md` under `skills_dirs`, injects a one-line-per-skill catalog into `validator` and `code_generator`, and exposes full recipes and scripts on demand via `read_skill(name, file=None)`.

| Category | Skills | Description |
| :--- | :--- | :--- |
| **Structure Prediction** | `boltz`, `chai`, `esmfold2`, `protenix` | Monomer & complex co-folding, ligand poses, binding affinity, confidence metrics (pLDDT, ipTM, PAE) |
| **Diffusion Models** | `rfdiffusion3`, `rfd4-proteina` | De novo backbone, binder, and motif/scaffold generation across XPU tiles |
| **Inverse Folding** | `proteinmpnn` | Fixed-backbone sequence design and log-probability scoring |
| **MD Simulation** | `molecular-simulations`, `openmm` | AMBER system building, explicit-solvent MD pipelines, MM-P(G)BSA, custom OpenMM runs |
| **Quantum / MLIPs** | `uma`, `reactra` | Fairchem UMA near-DFT energies/forces/relaxation and automated enzyme reaction workflows |
| **Structural Analysis** | `pyrosetta` | Rosetta scoring, FastRelax, interface $\Delta G$, shape complementarity, $\Delta\text{SASA}$ |
| **Foundation LMs** | `genslm-esm` | Codon and protein language model embeddings and sparse autoencoder (SAE) features |
| **Literature & RAG** | `fetch-literature`, `rag-build-index`, `rag-query` | PubMed / Europe PMC corpus harvesting, FAISS + MiniLM indexing, grounded Q&A |
| **HPC Infrastructure** | `hpc-deployment` | Globus Compute + PBSPro + Academy + Ensemble Launcher FastMCP server |

See [`skills/README.md`](skills/README.md) and [`skills/TOOL_SELECTION.md`](skills/TOOL_SELECTION.md) for task-by-task tool selection and ALCF Aurora environment details.

---

## Workspaces, Provenance & Steering

### Persistent Per-Hypothesis Workspaces

Orchestrator state (`state_dir/`) and agent-accessible files (`<execution.workdir>/`) live in separate directory trees so reviewers cannot inspect prior validator prose or unverified self-scores:

```text
state_dir/                                # Orchestrator-only records (written atomically)
├── jnana.json                            # Jnana hypothesis pool and Elo ratings
├── jnana-checkpoint-timings.jsonl        # One row per jnana.json checkpoint: reason, duration, size
├── ledger.jsonl                          # Reconciled, independently verified claims
├── rounds/
│   └── round-NNN.json                    # Full round record: rankings, threads, steering, brief
├── synthesis/round-NNN/<attempt>/        # The reasoner's input, every request/response, and manifest.json (the receipt)
└── hypotheses/<id>/
    ├── experiment.json                   # Hypothesis statement + per-round outcome summary
    └── runs/<rNNN-KK>.json               # Code-generator instruction, status, result, batch IDs

<execution.workdir>/<id>/                 # Agent-accessible workspace (persists across rounds)
├── outputs/<rNNN-KK>/                    # Isolated write directory for run rNNN-KK
└── data/                                 # Optional workspace_inputs (local mode only)
```

If a campaign is interrupted mid-round, restarting `python -m loam <config.yaml>` resumes cleanly after the highest completed `round-NNN.json`, replacing any partial ledger rows or round entries from the interrupted round. Within a round, Jnana's state is checkpointed atomically to `jnana.json` (at most every 5 seconds as hypotheses and matches land, and always at the end of generation), so a crash leaves the last complete checkpoint rather than a truncated file and a restart keeps the hypotheses and Elo ratings it holds. Updates since the last checkpoint can be lost. An interrupted first round generates only the hypotheses still missing from `initial_hypotheses`; a later round's restart generates a fresh `new_per_round` on top of any it had already checkpointed.

### Independent Review, Reconciliation & Steering

1. **Blind Claim Review:** The `reviewer` receives only the hypothesis statement, structured `claims[]`, and read-only access to URLs (`fetch_url`, restricted to public HTTPS addresses) and run artifacts. Its default verdict is `unverifiable`.
2. **Deterministic Reconciliation (`reconcile.py`):** Only `accepted` claims whose `verification` mode (`source_confirmed` or `artifact_attested`) and `source_ref` match an owned run output or source URL enter `ledger.jsonl`.
3. **Steering Projection (`steering.py`):** Accepted claims are grouped by source URL or run ID with within-group decay (`1.0`, `0.25`, `0.10`) so five failed gates from one run count as one contradiction rather than five independent ones:
   ```text
   evidence_rating = 1200 + (400 / ln 10) * Σ group_deltas
   uncertainty     = 1 / sqrt(1 + independent_groups)
   priority        = 0.35 * z(Elo) + 0.50 * z(evidence_rating) + 0.15 * z(uncertainty)
   ```
   The resulting whole-pool table (`priority`, `rank_evidence_disagreement`, and `recommended_action`) guides the `reasoner` in writing the next round's `tournament_brief`.

### Saturation Accounting

`loam-saturation JOB --baseline BASELINE_JOB` audits a completed single-allocation job: every allocation attempt's GPU-hours from the scheduler's accounting (requeued boots included), time-weighted GPU busy from the job's telemetry, science-slot occupancy, distinct trials per allocated GPU-hour, and how many hypotheses got a reviewed, completed test. It writes `JOB/saturation-audit.json` and prints the gains over the baseline job and whether the scale-up gates pass. The job's System (`execution.hpc.system` in its campaign config) says how to read allocated time and the per-host GPU telemetry in `campaign/telemetry/*.jsonl`. A System that declares neither (`allocation_accounting`, `gpu_telemetry` in `loam/systems.py`) cannot be audited; Jupiter declares both, Aurora and Polaris neither yet. The node shape is the plan's `gpus_per_node`, else the System's. Samples that don't cover every GPU on the node count as unobserved. The audit never submits or changes jobs.

### Scaling Recipe

`loam-scaling --system jupiter --nodes 20` (or `--config loam.yaml`) sizes a campaign to an allocation: the inference : science node split, `concurrency`, the tournament's `top_k`, `initial_hypotheses`, `new_per_round`, `gen_batch_size`, `swiss_rounds` and `workers`, `synthesis_concurrency`, `rounds` and science workers per GPU. It prints them as campaign config settings, along with whether the configuration was measured and where. The numbers were measured on Jupiter at 20 and 256 nodes of four GPUs and 288 CPUs. Another System or node shape is refused unless you pass `--allow-unmeasured`. That, or another node count, split or GPU sharing, gives a warning and an unmeasured recipe. See [docs/scaling-recipe-book.md](docs/scaling-recipe-book.md) for the measurements and how to qualify a new layout.

### Inference Pool

`loam-inference-pool ROOT` serves several replicas of one model, from any OpenAI-compatible backend (vLLM, SGLang, a hosted endpoint), as a single authenticated endpoint. Point campaign models at it like any other `base_url`. `ROOT/pool.json` names the served `model`, an optional `checkpoint` (a replica whose entry declares other weights, or none, gets no requests), the routing `policy` (`least_loaded_affinity`, `least_loaded` or `round_robin`), `max_inflight_per_backend`, and the `backends`, each with an `id`, a `base_url`, and optionally a `token_file` or `token_env` for its bearer token. Unknown top-level keys, a duplicate backend `id` or an out-of-range value refuse the file; a backend entry may carry extra keys for its resolver. The pool re-reads `pool.json` every two seconds, so replicas can be added or retired while it runs. A bad edit marks every replica not ready until it is fixed, and the model and checkpoint are fixed for the pool's lifetime. Requests are forwarded unchanged. A request that fails before it starts streaming is retried on another replica, and a backend's error text is never passed on except for context-length numbers. The pool writes its URL and the path of its client token to `ROOT/pool-service.json`, and one timing record per attempt, with no prompt or completion text, to `ROOT/pool-request-timings.jsonl`. `POST /pool/batch` fans out up to 64 independent non-streaming requests, and `GET /pool/status` reports each replica's load and readiness. Install with the `inference-pool` extra.

**Self-hosted Nemotron replicas** (`loam-nemotron`, the `nemotron` extra) are an optional backend for the pool, ported from Moeen Meigooni's Jupiter runs. Each replica runs vLLM with the measured serving profile (see [`examples/nemotron/serving-profile.json`](examples/nemotron/serving-profile.json) and the [scaling recipe book](docs/scaling-recipe-book.md)) on one node, inside a job you already hold. Nothing here submits or cancels jobs.

```bash
loam-nemotron add ROOT --profile serving-profile.json --system jupiter --job-id "$JOB" \
    --expires-at "$JOB_END_EPOCH" --pool-address 10.x.y.z --host node-a --host node-b \
    [--allocation-manifest allocation.json]   # lists the replicas in ROOT/pool.json and prints their attempts
loam-nemotron serve ROOT/replicas/ATTEMPT    # on each replica's node, in that job (srun, mpiexec, ...)
loam-nemotron pool ROOT                       # the pool over them, on the --pool-address host
loam-nemotron retire ROOT --keep N            # drain, then stop, all but the first N replicas
```

`add` checks the local snapshot (its directory is named with the pinned revision, and every shard is present) and gives each replica a fresh bearer token and a self-signed certificate. `serve` checks that it runs in its job and on its host (and, with a manifest, on one of the allocation's inference hosts). It then starts vLLM and waits for a real completion. After that it relays to vLLM over TLS, accepting requests only from the pool's address, and writes a readiness record. The pool pins that certificate, and it routes to a replica only while the readiness record names this pool, job, host, model and revision, and the job has not ended. `retire` disables replicas in `pool.json`, waits until the running pool has finished their requests, then has their `serve` shut vLLM down. You then release their nodes.

---

## Memory (optional)

Within a campaign, the brief, the ledger and each hypothesis's prior work carry knowledge from round to round. A `memory:` block carries **lessons** across campaigns, so a new campaign doesn't pay again for failures an earlier one already hit. It uses [Hindsight](https://github.com/vectorize-io/hindsight), runs on the machine that runs the orchestrator, and keeps its database there. Without a `memory:` block nothing changes.

**What is saved.** Lessons, not findings. After each round's record is written, the orchestrator saves:
- **procedural lessons**: code-generator runs whose observed status was `blocked` or `crashed` (the instruction and the error), and validator reports whose outcome was `blocked` (the blocker named in `next_step`);
- **review lessons**: claims the reviewer marked `rejected` or `unverifiable`, with the reviewer's `checked` explanation.

Accepted claims, successful runs and briefs are never saved, so a finding about one target can't mislead a campaign on another. Every lesson is tagged with `campaign:<memory.campaign>`; round, hypothesis and run ids are kept as metadata, not in the lesson text, and file paths are replaced by `<path>`. Lessons are saved as one document per (round, hypothesis, kind), so saving them again replaces that document instead of adding copies. Each run starts by re-saving the last completed round's lessons, so a crash between writing a round record and saving its lessons loses nothing.

**Where lessons are injected.** The orchestrator recalls them; agents have no memory tool and never write to memory.
- **Validator:** lessons relevant to the hypothesis summary, beside its prior work.
- **Code generator:** procedural lessons relevant to the delegated instruction.
- **Reviewer and reasoner:** nothing, so verdicts and briefs rest only on this campaign's evidence.

Recalled text is capped at `recall_tokens` per prompt, and each lesson is labelled with its source campaign. Lessons are **leads, never evidence**: the prompt says they can't be cited, and a claim must still rest on a run output or a source URL.

**Install.** `uv sync --all-extras`, or `uv sync --extra memory` if you already have the extras you need (a single `--extra` removes the others). The extra installs `hindsight-api-slim[embedded-db]` and `hindsight-client`, with no torch.

**Minimal configuration** (argo-proxy; see the commented block at the end of [`examples/loam.yaml`](examples/loam.yaml)):

```yaml
memory:
  campaign: egfr-binders     # required: tags what this campaign saves
  llm:        {model: argo:gpt-5.6-luna, base_url: http://127.0.0.1:61612/v1}
  embeddings: {model: argo:text-embedding-3-small, base_url: http://127.0.0.1:61612/v1}
```

| Field | Default | Meaning |
| :--- | :--- | :--- |
| `campaign` | *(required)* | Tag on every lesson this campaign saves |
| `llm` | *(required)* | `ModelSpec` that extracts lessons; needs a `base_url` (OpenAI-compatible) |
| `embeddings` | *(required)* | `ModelSpec` for the embedding model; needs a `base_url` |
| `bank` | `loam` | Memory bank id, shared by all campaigns |
| `url` | none | An already-running Hindsight server; loam then only connects |
| `instance` | `loam-memory` when `url` is unset | Embedded Postgres instance of the loam-started server (not allowed with `url`) |
| `recall_tokens` | `1000` | Cap on recalled lesson text per prompt |
| `drain_timeout` | `600` | Seconds to wait at campaign end for queued lessons to finish saving |
| `timeout` | `60` | Seconds per memory call |

Keys are read from `api_key_env`, as for agent models; with argo-proxy, leave it out. Use `argo:text-embedding-3-small` (1536 dimensions), not `-large`: its 3072 dimensions exceed pgvector's HNSW index limit of 2000.

**Server and data.** Without `url`, loam starts `hindsight-api` as a child process for the length of the run, on embedded Postgres instance `instance`, and stops it at the end after waiting up to `drain_timeout` for lessons still being saved. The data lives in `~/.pg0/instances/<instance>/` and persists across campaigns. Lessons still queued when `drain_timeout` runs out resume the next time loam starts the server on the same `instance` (or, with `url`, when that same server restarts); they are not recovered anywhere else.

**The embedding model is fixed per `instance`.** The vector size is set by the first embedding model used. loam records that model once the server first starts healthy (so a typo or a failed first start does not lock the instance) and afterwards refuses to start if the config names another one, checking before the server starts, since Hindsight would otherwise resize the column under the stored vectors. To switch models, set a new `instance`; it starts with an empty bank.

**Concurrent campaigns.** Start one Hindsight server yourself and point each campaign at it with `url` (and no `instance`). loam then only connects as a client. The embedding-model guard doesn't apply, because loam doesn't own that database: keeping the embedding model consistent is up to you.

**Failures.** Misconfiguration fails fast, before round 1: the extra isn't installed, a named `api_key_env` is unset, the server never becomes healthy, bank setup fails, or the embedding model doesn't match. After that, memory never costs a round: a failed recall gives the agent no lessons, a failed save is logged and dropped, and the round finishes as it would without memory.

**Smoke test.** [`examples/memory_smoke.py`](examples/memory_smoke.py) is an opt-in check against real endpoints, not part of `pytest`. It starts a server on a separate instance (`loam-memory-smoke`), queues lessons asynchronously and waits for them to be saved, checks the embedding-model guard, and recalls the lessons:

```bash
uv sync --all-extras
.venv/bin/python examples/memory_smoke.py --base-url http://localhost:61612/v1 \
    --llm-model argo:gpt-5.6-luna --embed-model argo:text-embedding-3-small
```

---

## Development

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) is the source of truth for the checks; it runs on every pull request and push to `develop` and `main`:

```bash
uv sync --all-extras
uv run pytest -q
uvx ruff@0.14.2 check .   # version pinned by [tool.ruff] required-version in pyproject.toml
```

Judgement-call rules that ruff cannot check live in [`CODING_STANDARDS.md`](CODING_STANDARDS.md).

---

## Repository Layout

```text
loam/
├── __main__.py        # CLI entry point (`python -m loam <config.yaml>`)
├── config.py          # Strict YAML configuration models (Config, ModelSpec, Execution, HPCDeployment, MemoryConfig)
├── memory.py          # Optional cross-campaign lesson memory: Memory interface, NoMemory, lesson extraction
├── hindsight.py       # Hindsight backend for memory: loam-owned server, embedding-model guard, retain/recall
├── launcher.py        # Managed HPC MCP server lifecycle
├── orchestrator.py    # Deterministic Python Campaign & round loop
├── agents.py          # Builder for validator, code_generator, reviewer, and reasoner
├── execution.py       # Local tools, OwnedMCP, Resilient wrapper, public-HTTPS fetch_url
├── schemas.py         # Strict Pydantic output contracts for every agent
├── tournament.py      # Jnana CoScientist + incremental Swiss tournament adapter
├── workspace.py       # Persistent per-hypothesis records, run directories, atomic JSON writes
├── reconcile.py       # Deterministic claim acceptance and provenance verification
├── steering.py        # Elo + evidence priority scoring across the hypothesis pool
├── synthesis.py       # Round synthesis: reduces threads + steering to the next brief within the context
├── skills.py          # SKILL.md discovery, prompt catalog, and read_skill helper
├── inference_pool.py  # One OpenAI-compatible endpoint over a pool of replicas (`loam-inference-pool`)
├── nemotron.py        # Optional pool backend: self-hosted Nemotron replicas on vLLM (`loam-nemotron`)
├── token_refresh.py   # Fresh ALCF inference tokens for endpoints with `token_refresh: alcf`
├── scaling.py         # Scaling recipe from measured density, refusing unmeasured Systems (`loam-scaling`)
├── saturation/        # Saturation accounting: allocated GPU-hours, GPU busy, reviewed hypothesis counts, scale-up gates (`loam-saturation`)
└── prompts/           # Role system instructions (validator, code_generator, reviewer, reasoner)
examples/              # Aurora EGFR binder campaign YAML, goal, campaign rules, redacted Site profiles
skills/                # Scientific tool skills and managed HPC deployment server
tests/                 # Offline end-to-end round, resilience, and steering tests
logo/                  # Project logo
.github/workflows/     # CI: pytest + ruff
CODING_STANDARDS.md    # Judgement-call coding rules (read by /code-review)
PLAN.md                # Architectural rationale, failure handling table, and design notes
```
