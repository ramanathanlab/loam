# Skill-MCP design: per-skill `run_row` functions fronted by an MCP, executed through `run_batch`

Status: **draft for discussion** (2026-11-11). Owner: avasan.

Goal: give each science skill (chai, boltz, protenix, esmfold2, rfdiffusion3,
rfd4-proteina, proteinmpnn, molecular-simulations, openmm, uma, reactra,
pyrosetta, genslm-esm, …) its own MCP front door, so the flow is:

```
model → skill-MCP tool (e.g. chai.fold) → write the run_row function (fixed) +
        a list of input rows → run_batch(function, inputs.jsonl, venv) →
        execution on Aurora via the existing hpc-deployment MCP (Academy/Globus/EL)
```

The hpc-deployment skill stays the **sole executor** and keeps its "no
domain-specific logic" boundary. A skill-MCP only (a) owns a typed input schema,
(b) owns the frozen per-row runner ("the function that takes inputs"), (c) knows
the skill's path manifest, and (d) stages + delegates to `run_batch`. It runs no
science itself.

---

## 1. What already exists (don't rebuild it)

- **The runner archetype.** `skills/diffusion-models/rfd4-proteina/scripts/rfd4_run_row.py`
  is already "a function that takes inputs": `python rfd4_run_row.py input.json`,
  one JSONL row, resolves its wrapper via `row["wrapper"] → $RFD4_WRAPPER →
  $HPC_LOAM_HOME/skills/.../run_rfd4_resident.sh`, runs the stack, writes outputs
  into the per-row cwd. `chai`'s `SKILL.md` does the same thing inline
  (`$CHAI_PY $LAUNCH fold …`).
- **The executor.** `run_batch(script_path, inputs_path, timeout_each, gpus,
  venv_path, batch_id, …)` (`hpc_agent.py`) submits one EL task per row,
  `gpus=1` = one tile, 12 rows fill an Aurora node. `_run_one` invokes
  `python script input.json` (call="json") or `--name value` (call="argv") in a
  per-row folder, promotes outputs on success, enforces an `expects` output
  contract.
- **A dormant tool registry.** `register_tool` / `run_batch(tool=…)` already
  gives typed params, versioning, a smoke run, and `expects`. The current
  `SKILL.md` deliberately disables it ("There is NO tool registry") and tells the
  model to hand-write scripts. **A skill-MCP's manifest is nearly isomorphic to a
  `register_tool` call** — see §5.

**The gap a skill-MCP closes:** today the model re-derives the runner and the
exact launcher/venv/weights/env every campaign from `SKILL.md`. That is where the
silent traps live (chai dropping a ligand with only a `Failed to tokenize` line;
rfd4's lightning shim; proteinmpnn's IPEX stub; uma's wrong XPU forces in
`default` mode). Freezing the correct invocation once, behind a typed tool, is
the whole value.

---

## 2. Target flow (concrete)

```
chai.fold(inputs=[{id, fasta, use_msa, recycles, …}, …], defaults={…})
  1. MCP validates every row against the chai input schema (typed, like register_tool params)
  2. MCP stages inputs.jsonl onto the shared FS (write_files → HPC_BASE_DIR/batches/…)
  3. MCP ensures chai_run_row.py is worker-visible (see §4.A)
  4. MCP calls hpc.run_batch(script_path=<runner>, inputs_path=<jsonl>,
                             venv_path=<chai venv or unset>, gpus=1, timeout_each=…)
  5. returns batch_id; model polls hpc.batch_status(batch_id)
```

The MCP never touches a GPU, never SSHes, never calls a scheduler. It is a typed
stager in front of `run_batch`.

---

## 3. The pathing problem, decomposed

Seven path categories cross the **laptop ↔ worker split**. The MCP and the model
run on the laptop; `run_batch` executes on `/flare` Aurora workers. **Nothing on
the laptop filesystem is worker-visible, and the MCP cannot `stat` a worker path
except through the hpc MCP's `inspect_filesystem` round-trips.** So every path
baked into a row or a runner must resolve *worker-side*.

| # | Path | Where it lives today | Who must resolve it |
|---|---|---|---|
| 1 | run_row function (`chai_run_row.py`) | loam tree | worker |
| 2 | skill launcher/wrapper (`chai_xpu.py`, `run_rfd4_resident.sh`) | loam tree, via `$HPC_LOAM_HOME` | worker |
| 3 | skill venv / interpreter (`chai-lab/.venv/bin/python`) | install dir (hardcoded `…/msinclair/software/…`) | worker |
| 4 | weights / caches (`chai-lab/downloads`, `FAIRCHEM_CACHE_DIR`) | install dir (hardcoded) | worker |
| 5 | inputs JSONL | staged by MCP under `HPC_BASE_DIR` | worker |
| 6 | per-row input refs (an input CIF, a FASTA) | `HPC_BASE_DIR` / project tree | worker |
| 7 | outputs | per-row folder under the batch dir | worker |

Three questions fall out:

- **A.** Where does the runner live, and how does the worker find it?
- **B.** Where do install-specific paths (venv, weights, launcher root) come from,
  and who owns them?
- **C.** How is the venv selected per run? (the user's specific question)

5–7 are already solved: the MCP stages inputs with `write_files`, rows carry
absolute `/flare` paths, outputs go to the per-row cwd. The open work is A, B, C.

---

## 4. Pathing solutions

### A. Locating the run_row function

- **A1 — commit the runner in the loam tree, reference it by `$HPC_LOAM_HOME`
  (current rfd4 approach).** The worker reads it from the `/flare` loam clone;
  `run_batch` copies/freezes it. Requires `HPC_INSPECT_ROOTS` to include
  `$HPC_LOAM_HOME/skills` (already done in `set_env_akv.sh`). **Cost:** editing a
  runner needs `git push` + `pull` on the clone + a fresh block to take effect
  (the known worker-sync lag).
- **A2 — MCP ships the runner body and stages it with `write_files` each batch.**
  The MCP is the single source of truth for the runner; the staged copy always
  matches the MCP version, so **no git-sync lag**. Cost: one cheap `write_files`
  per batch. Best for small pure-python runners.
- **A3 — `register_tool`: freeze the runner as a versioned, smoke-tested tool**
  under `HPC_BASE_DIR/tools`, call `run_batch(tool="chai@3")`. Adds provenance
  and the `expects` contract for free. Heavier; the registry is currently
  disabled by policy.

**Recommendation:** **A2 for the runner body** (MCP owns it, `write_files` stages
it, no clone lag), **A1 for the heavy/stack-coupled launchers** (`chai_xpu.py`,
`run_rfd4_resident.sh` — big, environment-entangled, rarely change; keep them in
the clone and reference via `$HPC_LOAM_HOME`). Revisit **A3** once the pattern is
proven, to get versioning + output contracts.

### B. Install-specific paths (venv, weights, launcher root) — resolve them WORKER-SIDE

The anti-pattern to avoid: the laptop MCP holding `…/msinclair/software/chai-lab/…`
and baking it into rows. That hardcodes one user's install in a *second* place and
breaks the moment the install moves or another user runs it.

- **B1 — a worker-side site manifest.** One file on `/flare`, e.g.
  `$HPC_LOAM_HOME/skills/site.<system>.json`, keyed by skill:
  ```json
  {
    "chai":   {"python": "/flare/.../chai-lab/.venv/bin/python",
               "launcher": "$HPC_LOAM_HOME/skills/structure-prediction/chai/scripts/chai_xpu.py",
               "weights": "/flare/.../chai-lab/downloads",
               "env": {"CHAI_DOWNLOADS_DIR": "/flare/.../chai-lab/downloads"}},
    "uma":    {"python": "/flare/.../fairchem-xpu/bin/python",
               "env": {"FAIRCHEM_CACHE_DIR": "/flare/.../hf_cache"}}
  }
  ```
  The **runner reads this at runtime** (keyed by its skill, and by system — see
  the `HPC_SYSTEM` caveat in §4.D) and resolves #2/#3/#4. The laptop MCP never
  needs the real paths. Moving an install or onboarding a new user = edit one
  file. This is the direct generalization of what `rfd4_run_row.py` already does
  ad hoc with `$HPC_LOAM_HOME`.
- **B2 — env vars in `worker_init`** (`export CHAI_PY=…`, `CHAI_DOWNLOADS_DIR=…`).
  Fine for the *few shared* values (`HPC_LOAM_HOME` already works this way) but it
  does not scale: `worker_init` becomes a dumping ground and every skill's vars
  load into every worker.
- **B3 — MCP config holds the paths and bakes them into rows.** Rejected: puts
  worker-FS knowledge on the laptop and re-hardcodes install paths.

**Recommendation:** **B1** (worker-side `site.<system>.json`, per skill), keeping
only `HPC_LOAM_HOME` + the deployment venv in `worker_init` as they are now. The
`SKILL.md` "Environment" blocks (every skill already has one) are the source
content for this manifest.

### D. Worker-side validation (2026-11-11, live `loam-hpc` pool)

Probed the assumptions with a `run_code` on an Aurora worker (8-node
debug-scaling block, campaign `akv-egfr-20261110`). Findings:

| Path / var | Worker value | Exists | Implication |
|---|---|---|---|
| `HPC_LOAM_HOME` | `/flare/FRAME-IDP/avasan/loam` | ✓ | exported in `worker_init`; **A1 confirmed** |
| chai launcher (`$HPC_LOAM_HOME/.../chai_xpu.py`) | — | ✓ | `$HPC_LOAM_HOME`-relative launchers resolve |
| rfd4 wrapper + `rfd4_run_row.py` | — | ✓ | same |
| chai venv python (`…/msinclair/software/chai-lab/.venv/bin/python`) | — | ✓ | **cross-user install paths ARE worker-visible** — B1 may reference another user's install |
| chai downloads/weights | — | ✓ | same |
| `PROTEINMPNN_PATH` | `/flare/FRAME-IDP/avasan/Software/ProteinMPNN` | ✓ | **already an exported env var (B2) for this one skill** — the manifest (B1) should unify this, not fight it |
| `HPC_SYSTEM` | *empty* | — | **not exported to workers.** A runner keyed on worker-side `$HPC_SYSTEM` fails. |

Two concrete design corrections:

1. **`HPC_SYSTEM` is set on the laptop (`set_env`) but not carried into
   `worker_init`.** So B1's "key by system" cannot read `$HPC_SYSTEM` worker-side.
   Fix one of: (a) add `export HPC_SYSTEM=…` to `worker_init` (cheapest, mirrors
   how `HPC_LOAM_HOME` is handled); (b) detect the system worker-side
   (`ZE_AFFINITY_MASK` present ⇒ aurora, `CUDA_VISIBLE_DEVICES` ⇒ polaris); or
   (c) ship a single-system manifest and skip the key. Recommend (a).
2. **One skill already resolves via an env var (`PROTEINMPNN_PATH`), others via
   hardcoded `SKILL.md` paths.** The manifest should *absorb* existing env vars
   (fall back to `$PROTEINMPNN_PATH` if the manifest lacks an entry) rather than
   introduce a third convention.

### C. venv selection — "add the venv in the function call / worker_init?"

Key constraint: **one worker serves rows of many skills during an allocation**, so
a skill venv **cannot** be activated globally in `worker_init` — only the
deployment venv belongs there. The skill venv is selected per batch/row:

- **C1 — pass `venv_path` to `run_batch`.** `_run_one` then runs
  `<skill venv>/bin/python runner.py input.json`. Clean when the runner is pure
  python importing the skill lib. `venv_path` is frozen on first submit (H4) and
  benefits from the Copper read-only mount rewrite (`_copper_venv`). But a bare
  interpreter swap does **not** give you `module load` / `module swap` /
  `LD_LIBRARY_PATH` / proxy env that several skills need (rfd4 swaps frameworks;
  protenix prepends old oneAPI libs).
- **C2 — runner runs under the deployment python and SUBPROCESSES into the skill
  python/wrapper** (what chai and rfd4 do today: `$CHAI_PY $LAUNCH`, `bash -lc`
  wrapper). Most flexible; handles module swaps, proxies, lib paths. The venv
  lives **inside the run_row function**, resolved from the site manifest (B1) —
  which is exactly the user's "add the venv to the function call."
- **C3 — C2 + B1 combined** (recommended): `worker_init` activates the deployment
  venv; the runner reads `site.<system>.json`, gets this skill's `python` +
  `env`, and subprocesses into it.

**Answer to the user's question:** *partially* in worker_init — keep the
deployment venv there — but select each **skill** venv **per batch**, because one
worker runs many skills. Prefer resolving it **inside the run_row function** from
the site manifest (C3); fall back to `venv_path` (C1) only for pure-python skills
with no module/lib-path needs. If Copper is enabled, apply the same mount rewrite
to the subprocess python that `_copper_venv` applies to `venv_path`.

---

## 5. What each skill-MCP must declare (the manifest)

A small per-skill spec next to `SKILL.md`, e.g. `skill.mcp.json`:

```jsonc
{
  "name": "chai",
  "actions": ["fold"],
  "row_schema": {                      // typed, like register_tool params
    "id": "str", "fasta": "str", "use_msa": "bool",
    "recycles": "int", "timesteps": "int", "samples": "int", "seed": "int"
  },
  "defaults": {"use_msa": false, "recycles": 3, "timesteps": 200, "samples": 5},
  "gpus": 1,
  "timeout_each": 900,
  "runner": "chai_run_row.py",          // A2 body shipped by MCP, or A1 path
  "site_keys": ["python", "launcher", "weights", "env"],   // resolved via B1
  "expects": ["*.cif", "scores*.npz"],  // output contract (L3)
  "example": {"id": "smoke", "fasta": "…/ubq.fasta"}        // for smoke test
}
```

This maps almost 1:1 onto `register_tool(name, script_path, params, defaults,
call, venv_path, expects, example)` — so a skill-MCP can target either
`run_batch(script_path=…)` with `write_files` staging (now) or
`run_batch(tool=…)` (if the registry is re-enabled later) without changing its
public surface.

---

## 6. One MCP server, or one per skill?

**Recommendation: one `skills` MCP server exposing many typed tools**
(`chai.fold`, `proteinmpnn.design`, `uma.relax`, …). Rationale: the MCP does no
science and has no heavy deps — it only validates rows, stages files, and calls
the hpc MCP — so per-skill process isolation buys nothing and multiplies
deployment. The *science* deps already live in the per-skill venvs on the worker,
which is where isolation matters.

---

## 7. Prototype plan (one skill, end to end)

Do **chai** or **proteinmpnn** first (fast, well-understood traps):

1. Write `chai_run_row.py` (A2 body): read `input.json`, resolve chai `python` +
   `launcher` + `env` from `site.aurora.json` (B1), subprocess the fold (C2/C3),
   write `*.cif` + `scores*.npz` into cwd, grep its own log for `Failed to
   tokenize` and fail the row if a ligand was dropped (bake the known trap into
   the runner, not the model's memory).
2. Write `site.aurora.json` with the chai entry from `chai/SKILL.md`'s Environment
   block.
3. Write a thin `skills` MCP tool `chai.fold(inputs, defaults)` that validates,
   `write_files`-stages `inputs.jsonl`, and calls `run_batch`.
4. Acceptance: 12 rows run concurrently on tiles 0–11 (~45 s, per the SKILL),
   `batch_status` → `unfinished == 0`, every row has a `.cif` + non-zero `iptm`
   for a complex. Confirm a dropped-ligand row is reported as a *failure*, not a
   silent pass.

Open decisions to settle before scaling to all skills:
- Runner sync policy: **A2 (`write_files`, no clone lag)** vs A1 (clone +
  push/pull/new-block).
- When to turn the `register_tool` registry back on (A3) for provenance +
  versioned output contracts.
- Whether `site.<system>.json` lives in the loam tree (git-synced) or is a
  hand-maintained file on `/flare` (no lag, but off-repo).

---

## 8. First live test (2026-11-11) — fresh-session MCP call + a staging blocker

Tested the *intended* workflow directly: launched a **fresh headless Claude
session** (`claude -p`, loam-hpc MCP included) in a second herdr pane and had it
call the MCP natively to stage + submit the chai prototype.

**Result split:**

- ✅ **The core test passed.** The fresh session reached `loam-hpc` with **no
  connection workaround** — `mcp__loam-hpc__pool_status` returned valid JSON
  (`aurora, cluster, 8 nodes × 12 = 96 tiles`), and `write_files` / `run_batch`
  resolved and were invoked as native tools. (Contrast: the *parent* session has
  the startup-`ECONNREFUSED` cached and can't see the tools at all — see
  [[loam-hpc-mcp-endpoint]]. A fresh session is the clean way to drive the MCP.)
- ❌ **Submission blocked by a deployment-version mismatch.** `write_files`
  failed: `AttributeError: Agent<HpcClientAgent> does not have an action named
  "write_files"`. Root cause: the worker's `/flare` clone
  (`/flare/FRAME-IDP/avasan/loam`) is at **`89cffcb` (PR #27)**, 9 commits behind;
  `write_files` was added in **`416228d` (PR #29)**. The deployed `hpc_agent.py`
  has no `write_files` at all (verified by reading it on the worker via
  `run_code`). This is the [[aurora-workers-import-repo-clone]] lag, concretely.

**Implications for the design:**

1. **A2 (`write_files`-stage the runner/inputs) requires the deployed agent to be
   ≥ #29.** Until the `/flare` clone is pulled *and the block is cycled* (a running
   worker has already imported the old module; `git pull` alone won't reload it),
   A2 cannot stage anything. So the staging path has the same clone-lag dependency
   A2 was meant to escape — not for the runner *source* (which the MCP ships) but
   for the `write_files` *mechanism* that delivers it. Escaping clone lag for the
   runner still assumes a one-time, ≥#29 agent deployment.
2. **Interim fallback:** `run_code` *is* present on the client agent, so files can
   be staged by writing them from inside a `run_code` payload. The hpc SKILL
   discourages this ("never run the science in `run_code`"), but for *staging* it
   is a legitimate bridge until the agent is redeployed — and it proves the
   runner/venv/pathing/fold loop today without waiting on a block restart.
3. **Driving the MCP:** use a **fresh** Claude session (or `claude -p`) for
   MCP-backed work; don't rely on a parent session that booted while the server
   was down.

**To unblock the clean path:** on an Aurora login node,
`git -C /flare/FRAME-IDP/avasan/loam pull` to ≥ #29 (or to `big-run-mcp-skills`
to get the chai prototype too), then let the current block drain / request a
fresh one so the agents re-import the updated `hpc_agent.py`.

### 8.1 Resolution — green end to end (2026-11-11, same day)

Recycled the deployment and re-ran the clean path:

1. Switched the `/flare` clone `develop` → `big-run-mcp-skills` (write_files +
   the chai prototype now on disk), via a `run_code` `git checkout` on the worker.
2. Restarted the Globus Compute endpoint (`globus-compute-endpoint restart
   loam-aurora-flex` on `uan-0011` over the MFA ControlMaster). The **stop timed
   out** and aborted the auto-start, so an explicit `start` was needed; the
   manager came back with the **same UUID**, self-heal spawned a fresh user
   endpoint → **new 8-node block `9076123`** → workers importing the updated
   module.
3. **The "recycle = replace worker processes" rule held exactly.** A fresh
   `run_code` subprocess saw `write_files` immediately (new process, new import)
   while the live agent still 500'd — the disk edit and even a server bounce were
   insufficient; only the new block's fresh worker processes picked it up. Once
   the block ran, `write_files` became a live action.
4. A **fresh headless Claude session** (`claude -p`, loam-hpc MCP) then ran the
   whole intended workflow natively: `pool_status` → `write_files`-stage
   (runner + manifest + 32-row inputs) → `run_batch` → batch `b-152a8c7abbcb`.

**Result: 32/32 succeeded, 0 failed.** One concurrent wave (~83 s wall; per-fold
90.1 / 90.3 / 97.2 s min/med/max — overlapping, not serial), spread across all 8
nodes. Every row produced `out/pred.model_idx_0.cif` + a scores `.npz`. The
runner resolved the chai venv/launcher/weights **worker-side** (site manifest +
defaults) with zero laptop-side paths — **B1/C3 confirmed on live hardware**, and
**A2 (`write_files` staging) confirmed** once the agent is ≥ #29. The clone-pull +
block-cycle is a **one-time deploy cost**, not per-run.

---

## 9. Skill roster (2026-11-11) — `loam-skills` MCP

The `loam-skills` MCP server (`skills/skill_mcp/server.py`) discovers every
`skills/**/*.mcp.json` and registers one typed tool per `(skill, action)`. Each
tool takes a LIST of rows, validates them against the manifest, stages the
runner + `site.aurora.json` + `inputs.jsonl` via the loam-hpc `write_files`, and
submits a `run_batch`. 13 skills, 17 tools:

| Skill | Tool(s) | Live-smoked? |
|---|---|---|
| chai | `chai_fold` | ✅ 32/32 folds, twice (direct + via wrapper) |
| boltz | `boltz_fold` | authored (SKILL verified cmds) |
| esmfold2 | `esmfold2_fold` | authored — see per-row load caveat |
| protenix | `protenix_fold` | authored |
| rfdiffusion3 | `rfdiffusion3_generate` | authored |
| rfd4-proteina | `rfd4_proteina_generate` | runner pre-existed; manifest authored |
| proteinmpnn | `proteinmpnn_design`, `proteinmpnn_score` | authored |
| genslm-esm | `genslm_esm_embed` | authored |
| uma | `uma_calc` | authored |
| reactra | `reactra_plan`, `reactra_run` | plan authored; `run` TODO(verify) |
| molecular-simulations | `molecular_simulations_run` | authored |
| openmm | `openmm_run` | authored |
| pyrosetta | `pyrosetta_score`, `_relax`, `_interface` | authored (debug-node verified in SKILL) |

Each runner bakes the documented silent-failure trap into a hard failure (chai
`Failed to tokenize`; boltz "failed examples"; esmfold2 multi-record FASTA;
proteinmpnn silent CPU fallback; uma wrong-XPU-forces in `default` mode; protenix
unknown-model download).

**Known caveats to clear before trusting a skill at scale** (all tagged
`# TODO(verify)` in the runners):
- **reactra `run`** — ~1 h, unverified on Aurora; `REACTRA_MECH_OVERLAY` is a
  per-campaign `FILL_IN` in `site.aurora.json`; needs `timeout_each ≥ 7200`.
- **molecular-simulations** — only `explicit` pipeline verified; `complex`/`ligand`
  need rdkit+openbabel (SKILL says not installed); prod step counts exceed one EL
  task → reduce steps per row or lean on checkpoint-resume.
- **openmm** — only `amber` input verified; gromacs/charmm/pdb untested.
- **uma** — only the `omol` head verified on XPU.
- **esmfold2** — one-process-per-row pays the ~90 s model load **every row**,
  defeating its in-process batch amortization. If esmfold2 volume is high, add a
  multi-sequence batch row (one process folds many) rather than one row per seq.
- **rfdiffusion3** — `timeout_each` margin is from one small run; no committed
  per-plan launcher (runner emits the `rfd3 design` command inline).
- **proteinmpnn** — confirm the worker's `$PROTEINMPNN_PATH` checkout has a usable
  `.venv`, else the `site.aurora.json` `python` (msinclair) stands.

**Recommended next step:** live-smoke one row-set per skill (the manifest
`example` × ≥32 distinct rows) before any real fan-out, starting with the ones the
current campaign needs. Only `chai` has cleared this.

---

## 10. Live tool test (2026-11-11) — `loam-skills` via a fresh Claude session

Deployed `loam-skills` (HTTP :8010) and had a **fresh headless Claude session**
call each tool through `--mcp-config` (loam-skills + loam-hpc), submitting 32
distinct rows per tool and polling `batch_status`. 10 of 13 skills were testable
immediately (7 inline-input + pyrosetta/openmm/proteinmpnn with existing install
test data); molecular_simulations / rfd4_proteina / reactra were deferred (need a
staged PDB+Amber build / annotated CIF / campaign yaml+mech respectively).

**9/10 PASS 32/32 with real outputs** on the first live run:

| tool | result | artifact |
|---|---|---|
| chai_fold | ✅ 32/32 | cif + scores |
| boltz_fold | ✅ 32/32 | model_0.pdb |
| esmfold2_fold | ✅ 32/32 | pred.cif |
| protenix_fold | ✅ 32/32 | cif + confidence |
| genslm_esm_embed | ✅ 32/32 | embedding .pt |
| uma_calc | ✅ 32/32 | uma_out.json |
| pyrosetta_score | ✅ 32/32 | scores.json |
| openmm_run | ✅ 32/32 | traj.dcd + state |
| proteinmpnn_design | ✅ 32/32 | seqs/*.fa |
| rfdiffusion3_generate | ❌ 0/32 | env drift (below) |

This validates the manifest-driven framework and 8 previously-untested runners on
live hardware, via the real agent→MCP tool path.

**rfdiffusion3 — environment drift, not a runner bug.** All 32 rows failed with
`OSError: libmkl_intel_lp64.so.3: cannot open shared object file`. Aurora upgraded
26.26.0 → **26.181.0**; the `frameworks/2025.3.1` module (which the `foundry`/rfd3
venv is pinned to) now resolves MKL 2025.3, which ships `libmkl_intel_lp64.so.2` —
there is **no `.so.3` anywhere** in the stack, but rfd3's torch build requires it.
The `rfdiffusion3_run_row.py` module-swap is faithful to the SKILL.md (which
predates the upgrade); the fix belongs to the install owner (rebuild rfd3's torch
deps against the current MKL, or ship a matching MKL). Do NOT symlink system libs
around the ABI mismatch.

**Still to smoke** (need input prep): molecular_simulations (PDB + Amber build),
rfd4_proteina (annotated CIF via make_rfd4_input.py), reactra (run.yaml + mech
overlay). All three are plumbing-validated (tool registered, validates, stages,
submits) but not science-validated.

### 10.1 rfd4_proteina — annotation moved INTO the tool, now passing (2026-11-11)

`rfd4_proteina_generate` was the main deferred skill because it needed a
hand-made annotated condition-spec CIF. That annotation is now done **inside the
tool**: a row gives a folded `target` (+ `target_chain` / `target_range` /
`hotspots` / `binder_len`) and the runner builds the annotated CIF itself
(`annotate_rfd4.sh` → RFD4's `write_condition_spec_cif`, in the pinned engine env),
then generates — all in one tool call. `target_range` auto-derives from the
target's atom_site when omitted (the engine contig rejects a bare chain id). A
ready `input` CIF is still accepted (back-compat).

Tested through the MCP **both ways** — direct client (`b-372658cb3575`) and a
**fresh Claude session** (`b-c86f51f126ec`) — each **32/32 PASS**: all 32 rows
built an annotated CIF *and* produced a generated backbone + receipt (~77 s/row,
one concurrent wave). So the live-validated roster is now **10 of 13 skills**
(the 9 in §10 + rfd4_proteina).

Remaining: **rfdiffusion3** blocked by the MKL drift (§10, install fix);
**molecular_simulations** and **reactra** still need input prep (PDB+Amber build;
campaign run.yaml+mech) before a smoke.

Mid-run the debug-scaling block hit its 1 h walltime; the endpoint self-replaced
it (block 9076321) and the loam-hpc pool self-healed onto the new one — a brief
`AgentTerminatedError` during the handover, then normal service. The clone-lag
rule held: deploying `annotate_rfd4.sh` needed a `git pull` on the worker clone
(the tool stages the runner via write_files, but the runner resolves its helper
scripts from `$HPC_LOAM_HOME`).
