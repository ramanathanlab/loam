# ProteinMPNN on Jupiter (ARM64 GH200, CUDA)

Install: `$LOAM_SOFTWARE_ROOT/proteinmpnn`, the native upstream checkout with its pretrained weights
in the tree. It selects CUDA itself; the Aurora IPEX shim and XPU instructions do not apply, and no
stand-in package is needed.

## Environment

The Site profile's `worker_init` runs on every worker before your code (the redacted example is
`examples/site-profiles/jupiter.yaml`). Activate this tool's own venv inside each row so the Globus
worker's Python is not inherited.

```bash
MPNN="$LOAM_SOFTWARE_ROOT/proteinmpnn"
PY="$MPNN/.venv/bin/python"
"$PY" -c "import torch; print(torch.cuda.is_available())"
```

Preserve the `CUDA_VISIBLE_DEVICES` the launcher assigns; inside a one-GPU task the device is
`cuda:0`. Do not select a physical GPU, and do not use Aurora modules, `ZE_AFFINITY_MASK`, XPU
launchers, ALCF proxies or PBS submissions.

## mmCIF to PDB conversion

ProteinMPNN reads PDB, so RFD4 or Chai output needs converting first, and the helper needs gemmi in
**this** venv — gemmi installed in another tool's environment is not visible here:

```bash
"$PY" -c "import gemmi; print(gemmi.__version__)"     # the existing install has gemmi==0.7.5
```

Run the conversion with this Python, keep the original input, and audit the result as SKILL.md says.
During a campaign, report a missing dependency rather than modifying the shared software.

## Execution contract on Jupiter

- Run scientific workloads only through the campaign's HPC MCP tools, in the assigned workdir, with
  absolute campaign paths. Do not submit scheduler jobs, use SSH, replace the scientific software,
  or apply Aurora XPU shims.
- Stage the script and inputs with `write_files` (only under the campaign's `HPC_BASE_DIR`;
  read-only inspection roots do not grant write access to installed software), then run the science
  through `run_batch(gpus=1, timeout_each=6000)`, one design run per row, with a subprocess deadline
  no larger than 6000 seconds (or the campaign YAML's timeouts; allocation walltime still limits
  completion). Poll `batch_status` until `unfinished == 0`, then inspect the actual output files
  before reporting success.
- `run_code` is for bounded staging fallback, harvesting/manifests and post-error diagnosis only.
- Never pad a batch with duplicate jobs. Keep each row within the remaining Slurm walltime; queue
  waits are separate, so do not cancel queued work just because it waits.
- Stream logs to workspace files and preserve failed attempts.

## Smoke workload

`scripts/jupiter/smoke-workload.py` converts a bundled monomer PDB to mmCIF and back with gemmi,
designs one sequence for its first chain with the trained weights, and asserts the FASTA holds a
native record plus a designed sequence of the same length over the 20 standard residues. Stage it
into a fresh run directory and run it with this tool's Python through `run_batch`:

```bash
"$PY" smoke-workload.py --out <fresh-absolute-directory> --software "$LOAM_SOFTWARE_ROOT"
```

Then inspect `result.json` and the generated FASTA. It verifies installation plumbing only; reduced
settings are not a production protocol and do not count as campaign evidence.

Latest recorded manual validation (on the `moeen` branch's Jupiter port, not a fresh run of this
checkout): **passed** — mmCIF/PDB conversion and one trained-model sequence design. Evidence
directory: `$LOAM_SOFTWARE_ROOT/verification/jupiter-skills-smoke-20261110/proteinmpnn`. This
describes only the tested path at audit time (the evidence root's `summary.json` records the run);
revalidate after environment changes. Readable documentation does not establish that the software is
installed or that a run will succeed.

## Resource index

- `read_skill(name="proteinmpnn", file="SKILL.md")` — the System-neutral method
- `read_skill(name="proteinmpnn", file="references/jupiter.md")` — this file
- `read_skill(name="proteinmpnn", file="scripts/jupiter/smoke-workload.py")` — the smoke workload
