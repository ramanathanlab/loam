---
name: reactra
description: >-
  Author, review, run, resume, and interpret enzyme-reaction simulations with
  Reactra from a declared YAML experiment and MECH chemistry. Use for NNP/MM
  reaction sampling, OPES flooding, or inspecting Reactra results, including
  PMD and chorismate mutase. Uses the installed Reactra CLI and, for remote
  computation, Loam's existing HPC execution tool; not protein-binder affinity
  scoring or a replacement molecular-dynamics implementation.
---

# Run enzyme reactions with Reactra

Keep scientific intent in one concise `reactra.run/1` YAML, chemistry in MECH,
and deployment details in the execution profile. Reactra owns preparation,
candidate selection, numerical execution, validation, checkpoints, and artifacts.
Loam chooses experiments and interprets their evidence. Invoke the public CLI;
do not recreate those operations in an agent script or add a Reactra-specific
HPC tool.

## Establish the entry point

Use the requested installed revision and its control environment. In a Reactra
development checkout, `pixi install --frozen` and `pixi run --frozen reactra ...`
provide the software interface. On a provisioned host, use its installed `reactra`
or explicit control interpreter: `"$REACTRA_PYTHON" -m reactra.interfaces.cli ...`.
The examples below abbreviate this as `reactra`; Loam's Python environment is not
automatically the Reactra control or native environment.

```bash
reactra capabilities
reactra schema --section stages.sampling
reactra schema --section stages.sampling.ensemble.members
```

These discover supported controls without launching simulations. A wheel alone
does not install MECH/RDKit, native providers, drivers, or weights. For details,
read the matching installed checkout's `docs/getting-started.md`,
`docs/guides/yaml.md`, and `docs/guides/running.md`. Upstream:
[Reactra](https://github.com/MechanismAtlas/reactra),
[vision](https://github.com/MechanismAtlas/reactra/blob/main/docs/vision.md), and
[successful examples](https://github.com/MechanismAtlas/reactra/blob/main/docs/guides/successful-flooding-runs.md).
Commands here were checked against main `86212f7`; consult the installed CLI's
`--help` and matching docs when versions differ. There is no compatibility promise.

For PMD or the packaged CM reference, read [the example guide](references/examples.md).
For campaign registration and before remote execution, read
[Loam execution and recovery](references/loam-execution.md).

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
  is XPU `float32-high`, not an automatic CUDA/CPU fallback. Hardware support must
  match the installed provider and execution profile.
- Keep restraints, timestep, seeds, sampling caps, ensemble size, stopping rules,
  and requested outputs in YAML. Deployment paths and accelerator slots belong
  in the profile. Do not patch worker code to tune an experiment.

```bash
reactra preview RUN.yaml --authoring-only --format text
reactra validate RUN.yaml
```

Start with authoring review: inspect effective values and their origins before
expensive work. Path-only assets are hashed; omitted component bindings require
MECH. `validate` additionally resolves scientific assets and declarations; neither
command proves native readiness. Missing assets or invalid chemistry require a
specific repair, not removed hashes, weakened checks, or a silent provider change.

Use `reactra plan RUN.yaml --store STORE --format text` when reviewing a committed
plan or a reuse decision. It returns an executable plan only with declared
deployment. Ordinary `run` already resolves and commits its plan; do not add
repeated admission or historical evidence reconstruction around it.

## Execute and inspect

Full enzyme computation belongs in an authorized, provisioned allocation. Through
Loam, invoke the CLI inside the existing `run_code` worker as described in the
execution reference. The commands are the same for every enzyme:

```bash
reactra run RUN.yaml --store STORE --request-id RUN_ID
reactra status --store STORE --request-id RUN_ID
reactra result --store STORE --request-id RUN_ID --operation OPERATION_ID
```

Take `OPERATION_ID` from preview/status, and inspect the published metadata and
payloads relevant to the requested outcome. Status is a cheap metadata read;
do not replace polling with trajectory reanalysis. Respect the store ownership
and live-polling limitations in the execution reference.

| CLI exit | Meaning and next action |
|---|---|
| `0` | Command completed; inspect the scientific outcome and qualification fields. |
| `2` | Refusal/error; read JSON `error.category`, diagnostics, and `scientific_side_effects`. |
| `3` | Run/resume is not terminal; inspect existing work before deciding to resume. |
| `4` | Run/resume is terminal but requested outputs are unmet or an occurrence unsuccessful; retain the evidence. |

A timeout or lost transport response is an unknown execution outcome, not a
scientific negative or permission to launch another controller.

## Iterate without redoing completed science

For unchanged interrupted work, after reconciling worker ownership:

```bash
reactra resume --store STORE --request-id RUN_ID
```

Resume uses jointly committed engine, bias, and clock state. It takes no replacement
YAML. Changed chemistry, model, CVs, restraints, stopping rules, or sampling budgets
are a new experiment with a new request ID. Reuse a compatible accepted prefix
through Reactra's paired `--reuse-prefix-from` / `--reuse-through` options; see the
execution reference. Never splice coordinates or old bias into a changed run.

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
