# Boltz-2 on Aurora (Intel XPU)

Install: `$LOAM_SOFTWARE_ROOT/boltz` — a fork with PyTorch Lightning removed (vanilla inference
loop) and `--accelerator xpu` added. Do not edit the install directory.

## Environment

```bash
BOLTZ=$LOAM_SOFTWARE_ROOT/boltz/.venv/bin/boltz   # no module load or activate needed
export ZE_AFFINITY_MASK=2                   # ONE tile (0-11 in FLAT mode); appears as xpu:0
export http_proxy=http://proxy.alcf.anl.gov:3128 https_proxy=$http_proxy   # only for --use_msa_server
```

Weights and the CCD cache are already in `~/.boltz` (`boltz2_conf.ckpt`, `boltz2_aff.ckpt`, `mols/`);
pass `--cache ~/.boltz`. Copies live in `$LOAM_SOFTWARE_ROOT/checkpoints/`.

## Example

```bash
# Minimal runner: scripts/aurora/run_boltz_xpu.sh <input> <out_dir> [tile] [extra args...]
bash scripts/aurora/run_boltz_xpu.sh examples/prot_no_msa.yaml out 2

# Equivalent direct call (production settings)
$BOLTZ predict examples/prot_no_msa.yaml --out_dir out --cache ~/.boltz --accelerator xpu --devices 1 \
  --recycling_steps 3 --sampling_steps 200 --diffusion_samples 5 \
  --output_format pdb --override --num_workers 0
```

## Aurora-specific arguments

| flag | default | meaning |
|---|---|---|
| `--accelerator xpu` | auto | required on Aurora (`auto`/`gpu` fall through when CUDA is absent) |
| `--cache` | — | `~/.boltz` |
| `--no_kernels` | off | not needed on XPU |

## Must-know

- Always pin `ZE_AFFINITY_MASK` to one tile. For many inputs, run one process per tile with a
  different mask.
- `--use_msa_server` needs the proxy env vars above, or it hangs.
- Harmless noise: `aten::topk` kernel-override warnings, `RequestsDependencyWarning`, dropped
  checkpoint `hyper_parameters`, and `Aten Op fallback from XPU to CPU` for `torch.linalg.svd`
  (slow but correct).
- A `ccd` code must exist in `~/.boltz/mols`.

## Resource index

- `read_skill(name="boltz", file="scripts/aurora/run_boltz_xpu.sh")` — minimal one-tile runner
- `read_skill(name="boltz", file="examples/prot_no_msa.yaml")`, `examples/lig_aff.yaml`,
  `examples/ubq_msa.fasta` — example inputs (any System)
