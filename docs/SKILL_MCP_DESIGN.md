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
  The **runner reads this at runtime** (keyed by its skill + `$HPC_SYSTEM`) and
  resolves #2/#3/#4. The laptop MCP never needs the real paths. Moving an install
  or onboarding a new user = edit one file. This is the direct generalization of
  what `rfd4_run_row.py` already does ad hoc with `$HPC_LOAM_HOME`.
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
