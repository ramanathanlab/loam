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

Each LLM role uses a `ModelSpec` (`{model, base_url, api_key_env}`):
- **With `base_url`:** Connects to any OpenAI-compatible server (vLLM, ALCF/Argo proxy, LiteLLM, etc.).
- **Without `base_url`:** Accepts any native `pydantic-ai` model identifier (e.g., `anthropic:claude-sonnet-4-5`, `openai:gpt-5`, `test`).
- **Secrets stay out of YAML:** API keys are referenced by environment variable name via `api_key_env`; Globus Compute and Academy credentials stay in their standard external stores.

With the optional `memory:` block, the validator and code generator also get lessons from earlier campaigns in their prompts; no role gains a tool. See [Memory (optional)](#memory-optional).

---

## Installation

Requires **Python 3.11+** and a local checkout of **Jnana** (referenced via `tournament.jnana_root` in your campaign config).

```bash
# Using uv (recommended)
uv sync                    # core dependencies
uv sync --extra dev        # + pytest for offline test suite
uv sync --extra hpc        # + managed HPC (fastmcp, globus-compute-sdk, academy-py, ensemble-launcher)
uv sync --extra memory     # + optional cross-campaign memory (Hindsight); see "Memory (optional)"
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
- **`hpc`** — Managed HPC deployment (`aurora` or `polaris`) via **Globus Compute → PBSPro → Academy → Ensemble Launcher**.
  - `loam` translates `execution.hpc` into the deployment server's internal environment, spawns `mcp_server.py`, waits for cluster pool readiness, and tears down the allocation cleanly on exit (`SIGINT`).
  - No separate environment file or scheduler wrapper (`qsub`/`sbatch`) is needed.

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
├── ledger.jsonl                          # Reconciled, independently verified claims
├── rounds/
│   └── round-NNN.json                    # Full round record: rankings, threads, steering, brief
└── hypotheses/<id>/
    ├── experiment.json                   # Hypothesis statement + per-round outcome summary
    └── runs/<rNNN-KK>.json               # Code-generator instruction, status, result, batch IDs

<execution.workdir>/<id>/                 # Agent-accessible workspace (persists across rounds)
├── outputs/<rNNN-KK>/                    # Isolated write directory for run rNNN-KK
└── data/                                 # Optional workspace_inputs (local mode only)
```

If a campaign is interrupted mid-round, restarting `python -m loam <config.yaml>` resumes cleanly after the highest completed `round-NNN.json`, replacing any partial ledger rows or round entries from the interrupted round.

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
| `instance` | `loam-memory` | Embedded Postgres instance of the loam-started server (not allowed with `url`) |
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
├── skills.py          # SKILL.md discovery, prompt catalog, and read_skill helper
└── prompts/           # Role system instructions (validator, code_generator, reviewer, reasoner)
examples/              # Aurora EGFR binder campaign YAML, goal, and campaign rules
skills/                # Scientific tool skills and managed HPC deployment server
tests/                 # Offline end-to-end round, resilience, and steering tests
logo/                  # Project logo
PLAN.md                # Architectural rationale, failure handling table, and design notes
```
