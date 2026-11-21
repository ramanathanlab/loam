---
name: reactra
description: Author, validate, plan, run, resume and interpret Reactra enzyme-reaction simulations — one YAML plus a MECH mechanism file drives cofold/prepared input, an OpenMM classical model, NVT/NPT, near-attack sampling, a UMA NNP/MM reactive model and OPES reaction flooding. Use for NNP/MM reaction sampling or OPES flooding, or to inspect a .mech mechanism. Not a plain QM single-point tool — use the `uma` skill for that.
systems: [aurora]
---

# Reactra

## Before you run it: read the reference file for this System

The checkout, control and native interpreters, the mech-format overlay, the deployment profile and
the launch details differ per System. Read `references/<HPC_SYSTEM>.md` (for example
`read_skill(name="reactra", file="references/aurora.md")`) and follow it; it is authoritative over
this file for anything System-specific. This skill has a reference file for **aurora** only. It is
**unsupported** on any other System, including Jupiter, where no Reactra or mech-format runtime is
installed: report the step as blocked rather than adapting Aurora's instructions. Missing
prerequisites are not a scientific refutation.

Software is found only through Site profile variables: the checkout is under `$LOAM_SOFTWARE_ROOT`.
Never edit the checkout; write outputs to a directory you own.

Keep scientific intent in one `reactra.run/1` YAML, chemistry in MECH, and deployment details in
the execution profile. Reactra owns preparation, candidate selection, numerical execution,
validation, checkpoints and artifacts; loam chooses experiments and interprets their evidence.
Invoke the public CLI; do not recreate its operations in an agent script.

- PMD and the packaged CM reference: [references/examples.md](references/examples.md)
- Campaign registration, remote execution and recovery:
  [references/loam-execution.md](references/loam-execution.md)
- Docs in the checkout: `AGENTS.md`, `docs/getting-started.md`,
  `docs/guides/{yaml,running,successful-flooding-runs}.md`, `docs/status.md`. Upstream:
  [Reactra](https://github.com/MechanismAtlas/reactra).

## Choose and review the experiment

Establish only the missing inputs: intended reaction/endpoint, run YAML and MECH, prepared molecular
inputs or preparation route, model assets, execution profile, work directory/store, request ID, and
authorized compute/time budget.

- **For a new enzyme, use its own MECH atom identities, protonation, charge, cofactors, waters,
  region, CVs, target basin and integrity criteria.** PMD's atom selectors and thresholds are not
  generic defaults, and a PDB alone is not a prepared system.
- Keep restraints, timestep, seeds, sampling caps, ensemble size, stopping rules and requested
  outputs in YAML. Deployment paths and accelerator slots belong in the profile. Do not patch worker
  code to tune an experiment.
- Start with authoring review (`preview --authoring-only`): inspect effective values and their
  origins before expensive work. `validate` additionally resolves scientific assets and
  declarations; neither proves native readiness. Missing assets or invalid chemistry need a specific
  repair, not removed hashes, weakened checks or a silent provider change.
- `plan` returns an executable plan only with a declared deployment. `run` already resolves and
  commits its plan; do not add repeated admission around it.

## Commands

```bash
reactra capabilities                       # operations + operation kinds (JSON)
reactra schema --section stages.sampling   # authoring JSON schema
reactra inspect-mech MECHANISM.mech        # resolved MECH JSON
reactra preview RUN.yaml --authoring-only --format text
reactra example cm-flooding --output NEW_DIR --native-profile PROFILE \
  --model-weights uma-s-1p2p1.pt --forcefield protein.ff14SB.xml --forcefield tip3p.xml
reactra validate RUN.yaml                  # {"valid": true, "scientific_identity": "…"}
reactra plan RUN.yaml --format text        # admits the deployment; launches no science
reactra run    RUN.yaml --store STORE --request-id RUN_ID
reactra status --store STORE --request-id RUN_ID
reactra result --store STORE --request-id RUN_ID --operation OPERATION_ID
reactra resume --store STORE --request-id RUN_ID
```

The reference file says which interpreter or wrapper stands in for `reactra`, and where the profile,
weights and force fields live.

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

Report the YAML/revision, store/request and operation IDs, model/runtime identity, completed
sampling versus wall time, outcome counts, exact artifact locations and the next action. Use
Reactra's reports; do not create a parallel acceptance record.

## Must-know

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
- Stop an unchanged retry loop when the same error repeats; identify the cause or ask for the
  missing decision instead of consuming further allocations.
- Packaged MECH files are `maturity: draft`, so the YAML needs `allow_draft_mech: true` (the example
  YAML already sets it).
- `uma-s-1p2p1.pt` sha256 must be
  `b2673b85037b075674c25f55c34ffe1ff1e15db924be977b10a184765df0d5ce`.
- No runs on login nodes; don't restart a user-stopped calculation without permission.
