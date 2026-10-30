---
name: reactra
description: Author, review, validate, plan, run, resume and interpret Reactra enzyme-reaction simulations, with verified setup for ALCF Aurora. Reactra drives a whole pipeline from one YAML plus a MECH mechanism file: cofold/prepared input → OpenMM classical model → NVT/NPT → near-attack (NAC) sampling → UMA NNP/MM mechanical-embedding reactive model → OPES flooding/reaction sampling, with content-hashed provenance. Use it for NNP/MM reaction sampling or OPES flooding, to set up or check a chorismate-mutase (CM) or PMD run, to inspect a .mech mechanism, or to interpret Reactra results. For remote computation it uses Loam's existing HPC execution tool. It is not a standalone QM single-point tool (for plain UMA energies use the `uma` skill), not protein-binder affinity scoring, and not a replacement molecular-dynamics implementation.
---

# Reactra on Aurora

Checkout: `/flare/FRAME-IDP/msinclair/software/reactra` (do not edit; read `AGENTS.md`,
`docs/getting-started.md`, `docs/guides/{yaml,running,successful-flooding-runs}.md`,
`docs/status.md`). Reactra calls UMA (`uma_s_1p2p1`, XPU, `float32-high`, no compile)
through the fairchem-xpu Nitro provider, and OpenMM on the OpenCL platform of the GPU Max 1550.

Keep scientific intent in one concise `reactra.run/1` YAML, chemistry in MECH, and
deployment details in the execution profile. Reactra owns preparation, candidate
selection, numerical execution, validation, checkpoints, and artifacts. Loam chooses
experiments and interprets their evidence. Invoke the public CLI; do not recreate
those operations in an agent script or add a Reactra-specific HPC tool.

For PMD or the packaged CM reference, read [the example guide](references/examples.md).
For campaign registration and before remote execution, read
[Loam execution and recovery](references/loam-execution.md).

## Environments (what exists, what works)

| Layer | Interpreter | Status |
|---|---|---|
| Control/software CLI | `/flare/FRAME-IDP/msinclair/software/reactra/.venv/bin/reactra` (uv venv) | Works after the mech-format overlay below. Has no openmm/fairchem/ase. |
| Locked dev env | `pixi run --frozen reactra …` (pixi at `/lus/flare/projects/FRAME-IDP/msinclair/software/.pixi/bin/pixi`) | **Not installed** in this checkout (no `.pixi/`). `pixi install` would write into the software dir. |
| Native workers (OpenMM+UMA+PLUMED) | Comes from a deployment profile (`native.python`) | Shared deployment: `/lus/flare/projects/FRAME-IDP/dkastner/Projects/Reactra/` (`native-python`, `deployment/profile.yaml`, `assets/`) |

Loam's Python environment is not automatically the Reactra control or native
environment. A wheel alone does not install MECH/RDKit, native providers, drivers,
or weights. Commands here were checked against main `86212f7`; consult the installed
CLI's `--help` and matching docs when versions differ. There is no compatibility
promise. Upstream: [Reactra](https://github.com/MechanismAtlas/reactra),
[vision](https://github.com/MechanismAtlas/reactra/blob/main/docs/vision.md),
[successful examples](https://github.com/MechanismAtlas/reactra/blob/main/docs/guides/successful-flooding-runs.md).

### One-time setup: mech-format overlay (verified; outside the software tree)

The uv venv is missing `mech-format`. Any MECH-dependent command (inspect-mech, preview,
validate, plan, example) then refuses with "MECH inspection requires mech-format". Fix it without
touching the checkout:

Build the overlay in a directory **you** own and keep it for the campaign's lifetime.
There is no shared copy to inherit: if `$REACTRA_MECH_OVERLAY` is unset or missing,
build it here before any other command.

```bash
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=http://proxy.alcf.anl.gov:3128
O=<a directory you own>          # e.g. the campaign's work dir; must persist between calls
uv pip install --python /flare/FRAME-IDP/msinclair/software/reactra/.venv/bin/python \
  --target $O/mech_overlay \
  "mech-format @ git+https://github.com/MechanismAtlas/mech-format.git@29c79dc7ee9e352dc55e0ac0004750a7f8dd65dc"
# mech_format looks for its schema at <overlay>/../schema (parents[2] of its module), so:
ln -sfn mech_overlay/share/mech-format/schema $O/schema
export REACTRA_MECH_OVERLAY=$O/mech_overlay
```

The schema symlink is not optional — the layout, not just the package, is what
makes MECH resolve. The wrapper `scripts/reactra_cli.sh` reads
`$REACTRA_MECH_OVERLAY`, puts it on `PYTHONPATH` with `PYTHONDONTWRITEBYTECODE=1`,
and runs the venv's `reactra`. It refuses rather than guessing if that variable is
unset, so export it in the same `run_code` call that invokes the CLI.

## Choose and review the experiment

Establish only the missing inputs: intended reaction/endpoint, run YAML and MECH,
prepared molecular inputs or preparation route, model assets, execution profile,
work directory/store, request ID, and authorized compute/time budget. Reuse known
session choices; do not request approval again for already-authorized work.

- For a new enzyme, use its own MECH atom identities, protonation, charge, cofactors,
  waters, region, CVs, target basin, and integrity criteria. PMD's atom selectors
  and thresholds are not generic defaults. A PDB alone is not a prepared system.
- Use current supported methods. The maintained examples use UMA with mechanical
  embedding; EMLE is retired. At the checked revision, the qualified UMA execution
  is XPU `float32-high`, not an automatic CUDA/CPU fallback.
- Keep restraints, timestep, seeds, sampling caps, ensemble size, stopping rules,
  and requested outputs in YAML. Deployment paths and accelerator slots belong
  in the profile. Do not patch worker code to tune an experiment.

Start with authoring review: inspect effective values and their origins before
expensive work. Path-only assets are hashed; omitted component bindings require
MECH. `validate` additionally resolves scientific assets and declarations; neither
command proves native readiness. Missing assets or invalid chemistry require a
specific repair, not removed hashes, weakened checks, or a silent provider change.

`plan` returns an executable plan only with declared deployment. Ordinary `run`
already resolves and commits its plan; do not add repeated admission or historical
evidence reconstruction around it.

## Verified commands (2026-10-29, all < 25 s, no GPU needed)

```bash
RX=<skill>/scripts/reactra_cli.sh      # this skill's own wrapper
S=/flare/FRAME-IDP/msinclair/software/reactra
export REACTRA_MECH_OVERLAY=$O/mech_overlay   # from the one-time setup above
$RX capabilities                                   # operations + operation kinds (JSON)
$RX schema --section stages.sampling               # authoring JSON schema
$RX inspect-mech $S/src/reactra/data/mechanisms/chorismate-mutase.mech   # resolved MECH JSON
$RX preview $S/examples/chorismate-mutase-aroq-bounded-selection.yaml --authoring-only --format text

# Materialize a runnable packaged example (CM or PMD) into a NEW dir:
rt=/lus/flare/projects/FRAME-IDP/dkastner/Projects/Reactra
$RX example cm-flooding --output $PWD/cm \
  --native-profile $rt/deployment/profile.yaml \
  --model-weights /flare/FRAME-IDP/msinclair/hf_cache/models--facebook--UMA/snapshots/ce3b2744ff57c0f49a010b763efcbfe9ba82e2e1/checkpoints/uma-s-1p2p1.pt \
  --forcefield $rt/assets/protein.ff14SB.xml --forcefield $rt/assets/tip3p.xml
#  -> cm/{run.yaml, chorismate-mutase.mech, cm-prepared.zip, CHEBI-29934.mol, execution-profile.json}
$RX validate cm/run.yaml          # {"valid": true, "scientific_identity": "4f8d4206…"}
$RX plan cm/run.yaml --format text  # "Declared deployment admitted; no scientific work launched."
                                    # 13 fresh operations, ≤103002 dynamics steps
```
Each command above was checked to produce `inspect_mech_cm.json`,
`preview_authoring.txt`, `cm/`, `cm_validate.json` and `cm_plan.txt`. Write yours
into a directory you own; there is no shared output tree to read back.

## Running a workflow (NOT verified here; from docs)

Full enzyme computation belongs in an authorized, provisioned allocation. Through
Loam, invoke the CLI inside the existing `run_code` worker as described in the
execution reference. Full runs are long: the CM empty-store run took 53 min 43 s on
Aurora and the docs ask for a 2-h `capacity` allocation. Command shape:
```bash
reactra run cm/run.yaml --store cm/store --request-id cm-example-1
reactra status --store cm/store --request-id cm-example-1
reactra result --store cm/store --request-id cm-example-1 --operation reaction_sampling_candidate_0
reactra resume …    # restart a committed run
```
Take the operation ID from preview/status, and inspect the published metadata and
payloads relevant to the requested outcome. Status is a cheap metadata read; do not
replace polling with trajectory reanalysis. Respect the store ownership and
live-polling limitations in the execution reference.

PBS launcher: `tools/aurora/run_root.pbs` with `REACTRA_ROOT, REACTRA_PYTHON, REACTRA_CONFIG,
REACTRA_STORE, REACTRA_REQUEST_ID, REACTRA_ACTION=run, REACTRA_THREADS` (see `docs/guides/running.md`).
GPU tiles come from `native_device_slots[].environment.ZE_AFFINITY_MASK` in the deployment
profile. The shared profile pins **tiles 0 and 2**. To use other tiles, write your own profile
(`tools/setup/configure_local_preparation.py --native-only`) and pass it to `reactra example
--native-profile`. Don't just set ZE_AFFINITY_MASK in the shell, because the workers get their mask from the profile.
OpenMM uses `OpenCL` (`examples/aurora/native-platform.json`, precision mixed/double).

| CLI exit | Meaning and next action |
|---|---|
| `0` | Command completed; inspect the scientific outcome and qualification fields. |
| `2` | Refusal/error; read JSON `error.category`, diagnostics, and `scientific_side_effects`. |
| `3` | Run/resume is not terminal; inspect existing work before deciding to resume. |
| `4` | Run/resume is terminal but requested outputs are unmet or an occurrence unsuccessful; retain the evidence. |

A timeout or lost transport response is an unknown execution outcome, not a
scientific negative or permission to launch another controller.

## Iterate without redoing completed science

For unchanged interrupted work, after reconciling worker ownership, `reactra resume
--store STORE --request-id RUN_ID` uses jointly committed engine, bias, and clock
state. It takes no replacement YAML. Changed chemistry, model, CVs, restraints,
stopping rules, or sampling budgets are a new experiment with a new request ID.
Reuse a compatible accepted prefix through Reactra's paired `--reuse-prefix-from` /
`--reuse-through` options; see the execution reference. Never splice coordinates or
old bias into a changed run.

If the task prioritizes quick crossings, use an explicitly bounded discovery
experiment, edit the YAML, and retain every negative/failure. Stop an unchanged
retry loop when the same error repeats; identify the cause or ask for the missing
decision instead of consuming further allocations. A user-stopped run needs new
authorization to restart.

## Return useful evidence

Report the YAML/revision, store/request and operation IDs, model/runtime identity,
completed sampling versus wall time, outcome counts, exact artifact locations, and
the next action. Use Reactra's reports; do not create a parallel acceptance record.

Distinguish scalar target entry, `crossing_verified`, censored noncrossings, failed
members, and any analysis qualification. A tool exit, a crossing, a bias ceiling,
or a rising potential-energy trace is not an activation free-energy barrier.
Unsupported or unqualified quantities stay explicitly unavailable, not zero.
When reporting an estimator, include failed gates and model/method limitations;
never discard censoring or failures to improve a score.

Export only the needed payload or declared viewing output:

```bash
reactra export --store STORE --request-id RUN_ID --operation OPERATION_ID \
  --trajectory m000 --output NEW_MOVIE_DIR
reactra export --store STORE --request-id RUN_ID --operation OPERATION_ID \
  --slot SLOT --output NEW_ARTIFACT_PATH
```

Discover actual member IDs/slots in the result. Trajectory export requires
YAML-declared viewing output. Keep atom maps and provenance with coordinates;
export refuses overwriting existing destinations.

## Gotchas found

1. Without the overlay, every MECH-dependent command fails with `unsupported_configuration:
   MECH inspection requires mech-format`. With the overlay but no `schema` symlink, it fails with
   `infrastructure_failure: installed MECH schema not found`.
2. The checkout's `preview` of a materialized example **fails** when the native profile is the shared
   YAML profile: `NativePreparationProfile native_device_slots: Input should be a valid tuple`.
   `validate` and `plan` succeed on the same run.yaml. This is a code/profile version skew; it is unresolved.
3. Version skew: the shared deployment's `control-python` has an OLDER Reactra (`reactra.run/0.3`)
   and rejects `reactra.run/1` YAML from this checkout. Don't mix the deployment interpreter with this
   checkout's YAMLs/tests.
4. The native tests (`tests/native/…`, e.g. `test_smoke_workflow.py::test_public_tiny_prepared_workflow`)
   need the checkout's pixi envs (`.pixi/envs/parameterization/bin/python`). Run under the
   deployment `native-python`, they fail with FileNotFoundError. There is no native smoke without `pixi install`.
5. The maintained `examples/chorismate-mutase-*.yaml` expect `.reactra/assets/uma-s-1p2p1.pt`, which
   isn't present. Full validate/preview of those returns `input_io` (documented). `--authoring-only` works.
6. Packaged MECH files are `maturity: draft`, so the YAML needs `allow_draft_mech: true`. The example YAML already sets it.
7. AGENTS.md policy: no runs on login nodes, and don't restart user-stopped calculations without permission.

## Troubleshooting

- Errors are JSON `reactra.error/1` with `category` + `retry_guidance`; `scientific_side_effects: none`
  means nothing ran.
- The `uma-s-1p2p1.pt` sha256 must be `b2673b85037b075674c25f55c34ffe1ff1e15db924be977b10a184765df0d5ce`.
  The hf_cache copy and the deployment copy are identical.
