---
name: aurora-proteina-complexa
description: >-
  Run the Complexa binder-design pipeline on Aurora — generate, filter, evaluate
  and analyse binder structures against a target — including its subcommands and
  the oneAPI downgrade its activation causes. Use when you want an end-to-end
  binder pipeline rather than assembling generation and inverse folding yourself.
---

# Proteina-Complexa on Aurora

## Activate

```bash
R=/flare/FRAME-IDP/agentic_campaign_09_26/binder_design
source $R/envs/_scripts/activate_proteina-complexa-2026-venv.sh
```

This venv also carries `rf3`, `rfd3`, `mpnn`, `pdbfixer`, `obabel`, `prody` and
`tmol` (CPU-only), so a whole funnel can run inside it. Extra tooling lives in
`software/ProteinaComplexa-Aurora/tools`.

## Subcommands

`complexa --help` (verified rc=0) — *"Complexa Binder Design Pipeline - Single
GPU execution"*:

```
complexa {design,generate,filter,evaluate,analyze,analysis,status,
          download,demo,init,target,validate}
```

- `design` — the full pipeline
- `generate` — binder structures only
- `filter` → `evaluate` → `analyze` — the staged funnel (`analysis` runs
  evaluate → analyze together)
- `target` (also `complexa-target`) — prepare the target
- `init`, `download`, `demo`, `validate`, `status` — setup and checks

Run `complexa <subcommand> --help` for each stage's options before composing a
run. Start with `validate` and `demo` on a new system.

## Pitfalls

- **Activating this venv downgrades oneAPI**
  (`oneapi/release/2026.1.0 => oneapi/release/2025.3.1`, seen on stderr). Give it
  its own `bash -lc`; do not activate it in a shell that already set up another
  venv, and do not assume the module state afterwards.
- "Single GPU execution" means one tile per `complexa` process. Fan out with
  `run_batch` rows, not with internal parallelism.
- It bundles its own `rfd3`/`mpnn`, which may differ in version from
  `foundry-2026-venv`. Pick one venv per funnel stage and record which, or
  results stop being comparable.
