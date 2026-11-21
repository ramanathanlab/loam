# Reactra on Aurora (Intel XPU)

Checkout: `$LOAM_SOFTWARE_ROOT/reactra` (do not edit). Reactra calls UMA (`uma_s_1p2p1`, XPU,
`float32-high`, no compile) through the fairchem-xpu Nitro provider, and OpenMM on the `OpenCL`
platform (`examples/aurora/native-platform.json`, precision mixed/double). At the checked revision
(main `86212f7`) the qualified UMA execution is XPU `float32-high`, not an automatic CUDA/CPU
fallback. Consult the installed CLI's `--help` and matching docs when versions differ.

## Environment

| Layer | Interpreter | Status |
|---|---|---|
| Control CLI | `$LOAM_SOFTWARE_ROOT/reactra/.venv/bin/reactra` (uv venv) | Works after the overlay below. Has no openmm/fairchem/ase. |
| Locked dev env | `pixi run --frozen reactra …` | **Not installed** (no `.pixi/` in this checkout); `pixi install` would write into the software dir. |
| Native workers (OpenMM+UMA+PLUMED) | from a deployment profile (`native.python`) | `$REACTRA_DEPLOYMENT` (`native-python`, `deployment/profile.yaml`, `assets/`) |

`REACTRA_DEPLOYMENT` is the native Reactra deployment the campaign uses — possibly another user's
shared one. Set it in the campaign's `execution.setup`; if it is unset, or
`$REACTRA_DEPLOYMENT/deployment/profile.yaml` does not exist, report the step as blocked rather than
guessing a path.

**One-time setup: the mech-format overlay.** The uv venv is missing `mech-format`, so every
MECH-dependent command (inspect-mech, preview, validate, plan, example) refuses with "MECH
inspection requires mech-format". Build the overlay in a directory **you** own and keep it for the
campaign's lifetime — there is no shared copy to inherit:

```bash
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=$http_proxy
O=<a directory you own>          # must persist between calls
uv pip install --python $LOAM_SOFTWARE_ROOT/reactra/.venv/bin/python \
  --target $O/mech_overlay \
  "mech-format @ git+https://github.com/MechanismAtlas/mech-format.git@29c79dc7ee9e352dc55e0ac0004750a7f8dd65dc"
# mech_format resolves its schema at <overlay>/../schema (parents[2] of its module), so:
ln -sfn mech_overlay/share/mech-format/schema $O/schema
export REACTRA_MECH_OVERLAY=$O/mech_overlay
```

The `schema` symlink is not optional — the layout, not just the package, is what makes MECH resolve.
`scripts/aurora/reactra_cli.sh` reads `$REACTRA_MECH_OVERLAY`, puts it on `PYTHONPATH` with
`PYTHONDONTWRITEBYTECODE=1`, and runs the venv's `reactra`. It refuses rather than guessing if that
variable is unset, so export it in the same `run_code` call that invokes the CLI.

## Upgrading an existing install

Pull in the checkout you already use. Editable installs pick that up; a controller or worker running
an **installed wheel does not** — reinstall into each of those interpreters:

```bash
S=$LOAM_SOFTWARE_ROOT/reactra
PY=<the existing interpreter>
$PY -m pip install --no-deps --force-reinstall $S
$PY -c "import reactra; print(reactra.__file__)"    # the only reliable confirmation
```

The package version stays `0.1.0` across this update, so version text cannot tell you whether an
environment is current — check the import path. Do **not** rebuild the native runtime: provisioned
dependencies, model weights and the native runtime are unchanged. Finish in-flight requests on their
original engine and give changed methods/configurations a new request ID; historical stores are
export-only when method versions differ.

Migration for YAML written against the previous version:

- **Remove `stages.analysis.escape`** from conventional `flooding_rate` YAMLs. Events now use the
  first chemically verified product frame. Single-level sampling remains available.
- **The force-check displacement default is now `0.0002` nm.** To preserve the previous
  finite-difference check, set `models.nnp_mm.force_checks.displacement_nm: 0.0001` explicitly.
- New three-level PMD work uses `reactra example pmd-design-screen`, with the same asset/profile
  arguments as `pmd-flooding` — which remains the historical reproduction example, not the current
  recipe.

Full notes: `$S/docs/getting-started.md`, "Upgrade an existing Aurora installation".

## Example

```bash
RX=$HPC_LOAM_HOME/skills/qm/reactra/scripts/aurora/reactra_cli.sh
S=$LOAM_SOFTWARE_ROOT/reactra
rt=$REACTRA_DEPLOYMENT
export REACTRA_MECH_OVERLAY=$O/mech_overlay

# Inspect and author (all CPU-only, seconds)
$RX capabilities
$RX schema --section stages.sampling
$RX inspect-mech $S/src/reactra/data/mechanisms/chorismate-mutase.mech
$RX preview $S/examples/chorismate-mutase-aroq-bounded-selection.yaml --authoring-only --format text

# Materialize a runnable packaged example into a NEW dir
$RX example cm-flooding --output $PWD/cm \
  --native-profile $rt/deployment/profile.yaml \
  --model-weights $LOAM_CACHE_ROOT/models--facebook--UMA/snapshots/ce3b2744ff57c0f49a010b763efcbfe9ba82e2e1/checkpoints/uma-s-1p2p1.pt \
  --forcefield $rt/assets/protein.ff14SB.xml --forcefield $rt/assets/tip3p.xml
#  -> cm/{run.yaml, chorismate-mutase.mech, cm-prepared.zip, CHEBI-29934.mol, execution-profile.json}

$RX validate cm/run.yaml                         # {"valid": true, "scientific_identity": "…"}
$RX plan cm/run.yaml --format text               # 13 fresh operations, ≤103002 dynamics steps

# Run (long: the CM empty-store run took ~54 min; the docs ask for a 2 h `capacity` allocation)
$RX run    cm/run.yaml --store cm/store --request-id cm-example-1
$RX status --store cm/store --request-id cm-example-1
$RX result --store cm/store --request-id cm-example-1 --operation reaction_sampling_candidate_0
$RX resume --store cm/store --request-id cm-example-1
```

The inspect/author/validate/plan commands were verified on 2026-10-29 (each under 25 s, no GPU); the
run commands were not run here. The `uma-s-1p2p1.pt` under `$LOAM_CACHE_ROOT` and the deployment's
`assets/` copy are identical (sha256 `b2673b85…5ce`).

PBS launcher: `tools/aurora/run_root.pbs` with `REACTRA_ROOT, REACTRA_PYTHON, REACTRA_CONFIG,
REACTRA_STORE, REACTRA_REQUEST_ID, REACTRA_ACTION=run, REACTRA_THREADS` (see
`docs/guides/running.md`). It is an alternative deployment route, not a step inside a loam campaign.

## GPU tiles

GPU tiles come from `native_device_slots[].environment.ZE_AFFINITY_MASK` **in the deployment
profile**, not the shell. A shared profile may pin fixed tiles (the one used on 2026-10-29 pinned
tiles 0 and 2); for other tiles write your own profile
(`tools/setup/configure_local_preparation.py --native-only`) and pass it to
`reactra example --native-profile`. Setting `ZE_AFFINITY_MASK` in the shell does not move the
workers.

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

## Resource index

- `read_skill(name="reactra", file="SKILL.md")` — the System-neutral method
- `read_skill(name="reactra", file="references/aurora.md")` — this file
- `read_skill(name="reactra", file="references/examples.md")` — PMD and CM examples
- `read_skill(name="reactra", file="references/loam-execution.md")` — execution and recovery
- `read_skill(name="reactra", file="scripts/aurora/reactra_cli.sh")` — the CLI wrapper
