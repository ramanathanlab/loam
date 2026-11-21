---
name: rfd4-proteina
description: Generate protein backbones and target-conditioned binders with RFD4-Proteina (flow-matching all-atom design) from an annotated CIF — monomers, protein/DNA/small-molecule binders, motif scaffolding and enzyme design. Use for backbone or binder generation, including loam run_batch fan-out across a node's accelerators; for the diffusion alternative see the rfdiffusion3 skill.
systems: [aurora, jupiter]
---

# RFD4-Proteina

## Before you run it: read the reference file for this System

The install, the environment, how a worker is pinned to one accelerator, the entrypoint and the
registered-tool contracts differ per System. Read `references/<HPC_SYSTEM>.md` (for example
`read_skill(name="rfd4-proteina", file="references/aurora.md")`) and follow it; it is authoritative
over this file for anything System-specific. This skill has reference files for **aurora** and
**jupiter**. It is **unsupported** on any other System: report the step as blocked rather than
adapting another System's instructions. Missing prerequisites are not a scientific refutation.

Software is found only through Site profile variables: installs live under `$LOAM_SOFTWARE_ROOT`,
one directory per tool, and a shared pinned deployment is named by its own variable (see the Aurora
reference). Never write into an install; write outputs to the campaign workspace.

## Input — an annotated CIF

RFD4 conditions generation on an **annotated** mmCIF: the target structure plus AtomWorks
condition annotations — coordinate groups (`C_CRDGRP`), sequence (`C_SEQ`), hotspots (`C_HOT`) and
the rest, whose serialized column names differ from those configuration aliases — and
`pdbx_PDB_model_num`, with the binder expressed as a contig
`"<binder_len>, /0, <target>"`. A generic PDB will fail. Build it with this skill's input helper,
never by hand, and verify target identity, crop and hotspot numbering first:
[references/annotated-cif.md](references/annotated-cif.md). Validate a freshly built input on a one-row smoke before
fanning out.

## Stages

Backbone generation is one stage of a binder campaign, and each stage is its own registered tool
and its own batch row:

1. **Generate** backbones against the validated annotated input, one seed per row. Distinct seeds
   are distinct designs; reusing a seed is a replay.
2. **Sequence** the binder chain with the target fixed (`proteinmpnn`).
3. **Fold** the target plus the designed sequence and score the interface (`chai`, `boltz`).
4. **Analyse** contacts, interface confidence and pose dispersion, then rank.

Where a System's reference file offers a bundled pipeline, it runs those stages inside one row for
one candidate; that is a scheduling choice, not a reduction in sampling. Design count = rows ×
samples per plan: reach a large library by fanning out across accelerators, not one long serial
job. Size a fan-out to live free capacity (`pool_status` minus in-flight work from `list_batches`),
and keep each row inside the block walltime.

**Register each stage once, with its output contract**, then submit batches against it. Without
`expects`, a row that exits 0 having written nothing counts as a success, `batch_status` withholds
`ok`, and no validator may claim `executed`. Use `skip_smoke=true` for stages that need an
accelerator the submitter may lack, and keep the contract rather than dropping it to avoid a failed
stage.

## Outputs

A generated complex CIF holds the designed binder chain plus the fixed target chain(s), alongside a
receipt per sample and a run summary recording adapter/engine/checkpoint identity, precision,
device and seed. The exact paths are in the System's reference file.

Record for every stage: input and output checksums, source and checkpoint revisions, seed, the
effective overrides, and the failure logs of anything that did not finish.

## Must-know

- **Output existence alone is not completion.** Require `rc=0` **and** the run summary **and** the
  expected receipt(s), with matching input/seed identities.
- **One independent sampler per assigned accelerator.** These are independent samplers — never
  distribute one model across MPI ranks, and never issue concurrent calls against one resident
  worker process.
- Keep the accelerator assignment the launcher gave the row, and set the pinning variable before
  importing Torch — afterwards is too late.
- A generated backbone is not a validated design, and a folded complex's interface confidence is
  not affinity. Contact counts, ipTM and pose RMSDs are computational observations; report matched
  condition budgets and failed candidates rather than interpreting these as binding.
- A crop that cuts native disulfides is not a chemically intact construct for energetic ranking.
  Screen on the crop, then re-fold the actual designed sequences in the intact target and verify
  its disulfides before any energy estimate — and report a single-frame MM/GBSA number as a static
  estimate, never as a measured binding free energy.
- On timeout or SIGABRT, preserve logs, seed and input, and terminate only the owned process group.
  At most one retry after a concrete diagnosis; a second unexplained failure of a configuration
  should stop and report.
- Do not run on a login node, and do not enable broad CPU fallback to get around an allocation
  check.
