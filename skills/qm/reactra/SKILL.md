---
name: reactra
description: Author, validate, plan, run, resume and interpret Reactra enzyme-reaction simulations on Aurora — one YAML plus a MECH mechanism file drives cofold/prepared input, an OpenMM classical model, NVT/NPT, near-attack sampling, a UMA NNP/MM reactive model and OPES reaction flooding. Use for NNP/MM reaction sampling or OPES flooding, or to inspect a .mech mechanism. Not a plain QM single-point tool — use the `uma` skill for that.
---

# Reactra on Aurora

Checkout: `/flare/FRAME-IDP/msinclair/software/reactra` (do not edit). Reactra calls UMA
(`uma_s_1p2p1`, XPU, `float32-high`, no compile) through the fairchem-xpu Nitro provider, and
OpenMM on the `OpenCL` platform.

Keep scientific intent in one `reactra.run/1` YAML, chemistry in MECH, and deployment details in
the execution profile. Invoke the public CLI; do not recreate its operations in an agent script.

- PMD and the packaged CM reference: [references/examples.md](references/examples.md)
- Campaign registration and remote execution: [references/loam-execution.md](references/loam-execution.md)
- Install docs: `AGENTS.md`, `docs/getting-started.md`, `docs/guides/{yaml,running,successful-flooding-runs}.md`

## Environment

| Layer | Interpreter | Status |
|---|---|---|
| Control CLI | `/flare/FRAME-IDP/msinclair/software/reactra/.venv/bin/reactra` (uv venv) | Works after the overlay below. Has no openmm/fairchem/ase. |
| Locked dev env | `pixi run --frozen reactra …` | **Not installed** (no `.pixi/` in this checkout). |
| Native workers (OpenMM+UMA+PLUMED) | from a deployment profile (`native.python`) | Shared: `/lus/flare/projects/FRAME-IDP/dkastner/Projects/Reactra/` |

**One-time setup: the mech-format overlay.** The uv venv is missing `mech-format`, so every
MECH-dependent command (inspect-mech, preview, validate, plan, example) refuses with "MECH
inspection requires mech-format". Build the overlay in a directory **you** own and keep it for the
campaign's lifetime — there is no shared copy to inherit:

```bash
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=$http_proxy
O=<a directory you own>          # must persist between calls
uv pip install --python /flare/FRAME-IDP/msinclair/software/reactra/.venv/bin/python \
  --target $O/mech_overlay \
  "mech-format @ git+https://github.com/MechanismAtlas/mech-format.git@29c79dc7ee9e352dc55e0ac0004750a7f8dd65dc"
# mech_format resolves its schema at <overlay>/../schema (parents[2] of its module), so:
ln -sfn mech_overlay/share/mech-format/schema $O/schema
export REACTRA_MECH_OVERLAY=$O/mech_overlay
```

The `schema` symlink is not optional — the layout, not just the package, is what makes MECH resolve.
`scripts/reactra_cli.sh` reads `$REACTRA_MECH_OVERLAY`, puts it on `PYTHONPATH` with
`PYTHONDONTWRITEBYTECODE=1`, and runs the venv's `reactra`. It refuses rather than guessing if that
variable is unset, so export it in the same `run_code` call that invokes the CLI.

## Upgrading an existing install

Pull in the checkout you already use. Editable installs pick that up; a controller or
worker running an **installed wheel does not** — reinstall into each of those interpreters:

```bash
S=/flare/FRAME-IDP/msinclair/software/reactra
PY=<the existing interpreter>
$PY -m pip install --no-deps --force-reinstall $S
$PY -c "import reactra; print(reactra.__file__)"    # the only reliable confirmation
```

The package version stays `0.1.0` across this update, so version text cannot tell you
whether an environment is current — check the import path. Do **not** rebuild the native
runtime: provisioned dependencies, model weights and the native runtime are unchanged.
Finish in-flight requests on their original engine and give changed
methods/configurations a new request ID; historical stores are export-only when method
versions differ.

Migration for YAML written against the previous version:

- **Remove `stages.analysis.escape`** from conventional `flooding_rate` YAMLs. Events now
  use the first chemically verified product frame. Single-level sampling remains available.
- **The force-check displacement default is now `0.0002` nm.** To preserve the previous
  finite-difference check, set `models.nnp_mm.force_checks.displacement_nm: 0.0001`
  explicitly.
- New three-level PMD work uses `reactra example pmd-design-screen`, with the same
  asset/profile arguments as `pmd-flooding` — which remains the historical reproduction
  example, not the current recipe.

Full notes: `$S/docs/getting-started.md`, "Upgrade an existing Aurora installation".

## Example

```bash
RX=<this skill>/scripts/reactra_cli.sh
S=/flare/FRAME-IDP/msinclair/software/reactra
rt=/lus/flare/projects/FRAME-IDP/dkastner/Projects/Reactra
export REACTRA_MECH_OVERLAY=$O/mech_overlay

# Inspect and author (all CPU-only, seconds)
$RX capabilities                                 # operations + operation kinds (JSON)
$RX schema --section stages.sampling             # authoring JSON schema
$RX inspect-mech $S/src/reactra/data/mechanisms/chorismate-mutase.mech
$RX preview $S/examples/chorismate-mutase-aroq-bounded-selection.yaml --authoring-only --format text

# Materialize a runnable packaged example into a NEW dir
$RX example cm-flooding --output $PWD/cm \
  --native-profile $rt/deployment/profile.yaml \
  --model-weights /flare/FRAME-IDP/msinclair/hf_cache/models--facebook--UMA/snapshots/ce3b2744ff57c0f49a010b763efcbfe9ba82e2e1/checkpoints/uma-s-1p2p1.pt \
  --forcefield $rt/assets/protein.ff14SB.xml --forcefield $rt/assets/tip3p.xml
#  -> cm/{run.yaml, chorismate-mutase.mech, cm-prepared.zip, CHEBI-29934.mol, execution-profile.json}

$RX validate cm/run.yaml                         # {"valid": true, "scientific_identity": "…"}
$RX plan cm/run.yaml --format text               # admits the deployment; launches no science

# Run (long: the CM empty-store run takes ~54 min; the docs ask for a 2 h `capacity` allocation)
reactra run    cm/run.yaml --store cm/store --request-id cm-example-1
reactra status cm/store --store cm/store --request-id cm-example-1
reactra result --store cm/store --request-id cm-example-1 --operation reaction_sampling_candidate_0
reactra resume --store cm/store --request-id cm-example-1
```

PBS launcher: `tools/aurora/run_root.pbs` with `REACTRA_ROOT, REACTRA_PYTHON, REACTRA_CONFIG,
REACTRA_STORE, REACTRA_REQUEST_ID, REACTRA_ACTION=run, REACTRA_THREADS`.

## Arguments and exit codes

GPU tiles come from `native_device_slots[].environment.ZE_AFFINITY_MASK` **in the deployment
profile**, not the shell. The shared profile pins tiles 0 and 2; for other tiles write your own
profile (`tools/setup/configure_local_preparation.py --native-only`) and pass it to
`reactra example --native-profile`.

| CLI exit | Meaning and next action |
|---|---|
| `0` | Completed; inspect the scientific outcome and qualification fields. |
| `2` | Refusal/error; read JSON `error.category`, diagnostics, `scientific_side_effects`. |
| `3` | Run/resume is not terminal; inspect existing work before resuming. |
| `4` | Terminal but requested outputs unmet, or an occurrence unsuccessful; retain the evidence. |

Errors are JSON `reactra.error/1` with `category` + `retry_guidance`;
`scientific_side_effects: none` means nothing ran.

## Outputs

Results live in the content-hashed store (`--store`), keyed by request ID and operation ID (take
operation IDs from `preview`/`status`). `status` is a cheap metadata read — do not replace polling
with trajectory reanalysis. Export only the needed payload:

```bash
reactra export --store STORE --request-id RUN_ID --operation OPERATION_ID \
  --trajectory m000 --output NEW_MOVIE_DIR          # needs a YAML-declared viewing output
reactra export --store STORE --request-id RUN_ID --operation OPERATION_ID \
  --slot SLOT --output NEW_ARTIFACT_PATH
```

Discover real member IDs/slots in the result. Export refuses to overwrite an existing destination,
and keeps atom maps and provenance with coordinates.

## Must-know

- **Without the overlay, every MECH command fails** with `unsupported_configuration: MECH inspection
  requires mech-format`; with the overlay but no `schema` symlink,
  `infrastructure_failure: installed MECH schema not found`.
- **`preview` of a materialized example fails** when the native profile is the shared YAML profile
  (`NativePreparationProfile native_device_slots: Input should be a valid tuple`). `validate` and
  `plan` succeed on the same run.yaml. Unresolved version skew.
- **Don't mix interpreters.** The shared deployment's `control-python` has an older Reactra
  (`reactra.run/0.3`) and rejects `reactra.run/1` YAML from this checkout.
- There is no native smoke test without `pixi install`: `tests/native/…` needs the checkout's pixi
  envs and fails under the deployment `native-python` with FileNotFoundError.
- The maintained `examples/chorismate-mutase-*.yaml` expect `.reactra/assets/uma-s-1p2p1.pt`, which
  isn't present, so full validate/preview returns `input_io`. `--authoring-only` works.
- Packaged MECH files are `maturity: draft`, so the YAML needs `allow_draft_mech: true` (the example
  YAML already sets it).
- `uma-s-1p2p1.pt` sha256 must be
  `b2673b85037b075674c25f55c34ffe1ff1e15db924be977b10a184765df0d5ce`.
- **For a new enzyme, use its own MECH atom identities, protonation, charge, cofactors, waters,
  region, CVs, target basin and integrity criteria.** PMD's atom selectors and thresholds are not
  generic defaults, and a PDB alone is not a prepared system.
- `resume` takes no replacement YAML and reuses jointly committed engine/bias/clock state. Changed
  chemistry, model, CVs, restraints, stopping rules or sampling budgets are a **new experiment with
  a new request ID**. Reuse a compatible accepted prefix via `--reuse-prefix-from` /
  `--reuse-through`; never splice coordinates or old bias into a changed run.
- **A tool exit, a crossing, a bias ceiling or a rising potential-energy trace is not an activation
  free-energy barrier.** Distinguish scalar target entry, `crossing_verified`, censored
  noncrossings, failed members and analysis qualification. Unqualified quantities stay explicitly
  unavailable, not zero, and never discard censoring or failures to improve a score.
- A timeout or lost transport response is an **unknown** execution outcome — not a scientific
  negative, and not permission to launch another controller.
- No runs on login nodes; don't restart a user-stopped calculation without permission.
