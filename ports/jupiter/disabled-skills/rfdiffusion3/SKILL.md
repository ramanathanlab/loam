---
name: rfdiffusion3
description: Run RFdiffusion3 (RFD3) all-atom protein design inference on the ALCF Aurora Intel XPU install at /flare/FRAME-IDP/msinclair/software/foundry. Covers environment activation (the frameworks/2025.3.1 module the venv was built against), ZE_AFFINITY_MASK tile pinning, checkpoint discovery via FOUNDRY_CHECKPOINT_DIRS, input JSON specs (unconditional, binder with hotspots, motif/enzyme scaffolding, symmetry, partial diffusion) and the `rfd3 design` CLI. Use whenever asked to generate protein backbones/designs with RFdiffusion3 on Aurora. For RNA/DNA *generation* use the repo's rfdiffusion3na skill; for structure prediction use its rf3 skill; for RFD4-Proteina use the rfd4-proteina skill.
---

# RFdiffusion3 (RFD3) on Aurora

RFD3 is the all-atom diffusion model in the Foundry monorepo. You describe the
design in a JSON/YAML file (or with `+specification.*` overrides) and launch
`rfd3 design` with Hydra-style `key=value` arguments.

Install: `/flare/FRAME-IDP/msinclair/software/foundry` (do not modify it; write
outputs elsewhere).

## 1. Environment — the part that actually bites on Aurora

The venv at `foundry/.venv` has `include-system-site-packages = true` and its
`home` is **`/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin`**.
torch/lightning/IPEX come from that module, *not* from the venv. Aurora's
current default module is `frameworks/2026.1.0`, so with the default login
environment `import torch` fails with:

```
OSError: libmkl_intel_lp64.so.2: cannot open shared object file
...then: ImportError: libsycl.so.8: cannot open shared object file
```

You must swap to the 2025.3.1 frameworks module, whose modulefile lives in an
older tree that is not on the default `MODULEPATH`:

```bash
module use /opt/aurora/26.26.0/modulefiles
module swap frameworks/2026.1.0 frameworks/2025.3.1 2>/dev/null \
  || module load frameworks/2025.3.1
```

That swap also pulls `oneapi/release/2025.3.1` and
`intel-oneapi-runtime/2025.3.1` and sets `ZE_FLAT_DEVICE_HIERARCHY=FLAT` and
`ONEAPI_DEVICE_SELECTOR="opencl:gpu;level_zero:gpu"`. Verified afterwards:

```
torch 2.10.0a0+git449b176 ; torch.xpu.is_available() == True
```

Then pin exactly one tile and point at the checkpoints:

```bash
export ZE_AFFINITY_MASK=6        # one tile; share the node politely
export FOUNDRY_CHECKPOINT_DIRS=/flare/FRAME-IDP/msinclair/software/checkpoints
```

Call the venv entry points by absolute path (no `activate` needed):
`/flare/FRAME-IDP/msinclair/software/foundry/.venv/bin/rfd3`.

Device selection is automatic — `src/foundry/utils/ddp.py` probes
CUDA -> XPU -> MPS -> CPU and logs `Intel XPU detected - using XPU accelerator`.
Precision stays at the configured `bf16-mixed`; no XPU downgrade needed.

## 2. Checkpoints

`/flare/FRAME-IDP/msinclair/software/checkpoints`:

| File | Model |
|---|---|
| `rfd3_latest.ckpt` | RFD3 (alias `rfd3`, the `ckpt_path` default) |
| `rfd3na_1190.ckpt` | RFD3NA (nucleic-acid generation) |
| `rf3_foundry_01_24_latest_remapped.ckpt` | RF3 structure prediction |
| `proteinmpnn_v_48_020.pt` | ProteinMPNN weights |

Resolution order is `~/.foundry/checkpoints` plus the colon-separated
`$FOUNDRY_CHECKPOINT_DIRS`. The repo's `.env` already sets
`FOUNDRY_CHECKPOINT_DIRS` to the flare path, so runs from the repo root work
even without the export; export it anyway when running from elsewhere.
`ckpt_path=rfd3` (default) picks `rfd3_latest.ckpt`; pass an absolute path for
anything unregistered. `foundry list-installed` prints what is visible.

## 3. Verified minimal run

Script: `scripts/run_rfd3_min.sh` (next to this skill). Exactly what was run:

```bash
module use /opt/aurora/26.26.0/modulefiles
module swap frameworks/2026.1.0 frameworks/2025.3.1 2>/dev/null || module load frameworks/2025.3.1
export ZE_AFFINITY_MASK=6
export FOUNDRY_CHECKPOINT_DIRS=/flare/FRAME-IDP/msinclair/software/checkpoints
cd /flare/FRAME-IDP/msinclair/software/foundry
.venv/bin/rfd3 design \
  inputs=null +specification.length=50 \
  out_dir=<some-writable-dir>/out_uncond50 \
  diffusion_batch_size=1 n_batches=1 \
  inference_sampler.num_timesteps=50 \
  skip_existing=False prevalidate_inputs=True
```

One 50-residue unconditional design, 50 denoising steps (default is 200).

**Verified 2026-10-29 on an Aurora debug node, tile 6: PASS, 5m50s wall**
(`user` time only 36 s — nearly all of it is import + checkpoint load off
Lustre). Output: `out_uncond50/_0_model_0.cif.gz` + `.json`, 346 atoms,
50 residues, and metrics `n_chainbreaks: 0`, both clash counts `0`,
`radius_of_gyration: 10.16`, `helix_fraction: 0.3`, `sheet_fraction: 0.3` —
a sane compact mini-protein, so XPU numerics are fine, not merely non-crashing.
The log line to look for is
`INFO foundry.utils.ddp: Intel XPU detected - using XPU accelerator`.

Note the leading `+` on `+specification.length`: `specification` is an empty
dict in `models/rfd3/configs/inference_engine/rfdiffusion3.yaml`, so Hydra's
struct mode rejects a plain `specification.length=50` with
`ConfigCompositionException: Could not override 'specification.length'`. The
`+` is required for **every** `specification.<field>` override.

`rfd3` is a single-command Typer app, so `rfd3 design <overrides>` and
`rfd3 <overrides>` are equivalent (`design` is stripped) and `rfd3 --help` does
not list `design`.

## 4. Input specs

The input file maps **spec name -> design configuration**; each top-level key is
an independent job whose name appears in the output filenames. Select a subset
with `json_keys_subset=[key1,key2]`.

Unconditional / length-only (file form of the smoke test):

```json
{ "uncond50": { "length": "50" } }
```

Binder design against a target chain, steered by hotspots:

```json
{
  "pdl1": {
    "input": "/abs/path/5o45_cropped.pdb",
    "contig": "50-120,/0,A17-131",
    "infer_ori_strategy": "hotspots",
    "select_hotspots": {"A56": "CG,OH", "A115": "CG,SD", "A123": "CD2,OH"},
    "is_non_loopy": true
  }
}
```

Motif / active-site (enzyme) scaffolding with cofactors:

```json
{
  "mysite": {
    "input": "/abs/path/M0255_1mg5.pdb",
    "ligand": "NAI,ACT",
    "unindex": "A108,A139,A152,A156",
    "length": "180-200",
    "select_fixed_atoms": {
      "A108": "ND2,CG", "A139": "OG,CB,CA",
      "A152": "OH,CZ",  "A156": "NZ,CE,CD",
      "ACT": "OXT", "NAI": ""
    },
    "allow_ligand_on_existing_chain": true
  }
}
```

Symmetric oligomer: add `"symmetry": {"id": "C5"}` to the spec and pass
`inference_sampler.kind=symmetry` on the CLI. Partial diffusion: add
`"partial_t": <angstroms>` (start ~2, typical 5-15) to re-noise an existing
structure.

**Contig strings** are comma-separated segments, `/0` = chain break:
`A40-60` takes residues 40-60 of chain A from `input` as a fixed motif; `70`
designs exactly 70 residues bonded to the previous segment; `60-80` designs a
uniformly random length in that range. `length` (`"min-max"` or int) constrains
the total.

**InputSelection** fields (`select_fixed_atoms`, `select_unfixed_sequence`,
`select_buried`, `select_partially_buried`, `select_exposed`,
`select_hbond_donor`, `select_hbond_acceptor`, `select_hotspots`) accept
`true`/`false`, a contig string, or a `residue-range -> atoms` dict with the
shorthands `ALL`, `BKBN` (= `N,CA,C,O`), `TIP`, and `""` (select nothing).

Runnable examples live in
`/flare/FRAME-IDP/msinclair/software/foundry/models/rfd3/docs/examples/`, and the
complete field table is `models/rfd3/docs/input.md`.

## 5. Key CLI options

| Argument | Default | Meaning |
|---|---|---|
| `out_dir` | required | Output directory (created if missing). |
| `inputs` | required | JSON/YAML path, or `null` with `+specification.*`. |
| `n_batches` / `diffusion_batch_size` | 1 / 8 | Total designs = product. |
| `ckpt_path` | `rfd3` | Registry alias or absolute path. |
| `inference_sampler.num_timesteps` | 200 | Denoising steps; lower = faster/rougher. |
| `inference_sampler.step_scale` | 1.5 | Higher = less diverse, more designable. |
| `inference_sampler.gamma_0` | 0.6 | Lower = more designable, less diverse. |
| `inference_sampler.kind` | `default` | `symmetry` for symmetric sampling. |
| `skip_existing` | True | Set `False` to re-run into the same `out_dir`. |
| `prevalidate_inputs` | False | `True` validates specs before loading weights. |
| `json_keys_subset` | null | Run a subset of spec keys. |
| `low_memory_mode` | False | Try on OOM. |
| `seed` | null | Reproducibility. |
| `dump_trajectories` | False | Writes denoising trajectories (large). |
| `devices_per_node` / `num_nodes` | 1 / 1 | Multi-tile / multi-node fan-out (**untested here**). |

Defaults: `models/rfd3/configs/inference_engine/rfdiffusion3.yaml` and
`.../base.yaml`; anything there is CLI-overridable.

## 6. Output layout

Per design, flat in `out_dir`:

```
<inputs-file-basename>_<spec-key>_<batch>_model_<n>.cif.gz
<inputs-file-basename>_<spec-key>_<batch>_model_<n>.json
```

With `inputs=null` there is no basename or spec key, so the verified run wrote
`_0_model_0.cif.gz` / `_0_model_0.json`. `n` starts at 0. The `.json` carries a
`metrics` block (`n_chainbreaks`, clash counts, `radius_of_gyration`,
secondary-structure fractions, `num_residues`) that is the quickest sanity check
on a design. The JSON echoes the full
specification including the *sampled* contig, so variable-length runs are
reproducible. CIF B-factors carry the sequence head's per-token confidence.

## 7. Aurora / XPU gotchas

1. **Wrong frameworks module = import error, not a GPU error.** See §1. The
   venv is pinned to `frameworks/2025.3.1`; the node's default is now
   `frameworks/2026.1.0`, and `/opt/aurora/26.26.0/modulefiles` needs a
   `module use` before the swap works.
2. **`set -u` + Lmod don't mix.** A wrapper script with `set -euo pipefail`
   dies in `/usr/share/lmod/lmod/init/bash` with
   `ZSH_EVAL_CONTEXT: unbound variable`. Use `set -eo pipefail`.
3. **Pin a tile.** `ZE_AFFINITY_MASK=<tile>` before launching; the job
   otherwise grabs tile 0 and collides with other users. With the mask set,
   `torch.xpu.device_count() == 1`, which is what single-tile runs want.
   Independent jobs on separate masks run fine concurrently.
4. **Startup is slow, not hung.** Import + weight load dominates a small run
   (minutes on cold Lustre). Hydra prints two harmless warnings first:
   `provider=hydra.searchpath in main, path=rfd3.configs is not available`.
5. **venv deps.** `lightning` comes from a checkout at
   `/flare/FRAME-IDP/msinclair/software/lightning`; `lightning_utilities` and
   `torchmetrics` were installed with `uv pip install --no-deps`. If you ever
   reinstall, keep `--no-deps` — a plain install drags in a CUDA torch wheel
   that shadows the Aurora XPU build.
6. **Multi-tile DDP (`devices_per_node>1`) is unverified** on this install.

## 8. Troubleshooting

- `ConfigCompositionException: Could not override 'specification.*'` — add the
  leading `+`.
- `"Input provided but unused"` — you gave `input` but no `contig`, `unindex`,
  `ligand`, or `partial_t`.
- Nothing generated — `skip_existing=True` (default) and outputs already exist.
- `ValueError: Multiple ligand residues on chain B...` — add
  `"allow_ligand_on_existing_chain": true`. The shipped
  `models/rfd3/docs/examples/enzyme_design.json` omits it and fails as written.
- Shipped examples use relative `../input_pdbs/*.pdb` paths — run from
  `models/rfd3/docs/examples/` or rewrite to absolute paths.
- Selection matched no atoms — run with `prevalidate_inputs=True` and check
  chain IDs / numbering.
- OOM — lower `diffusion_batch_size`, set `low_memory_mode=True`.
- Better tracebacks — call the Hydra entry point directly:
  `.venv/bin/python models/rfd3/src/rfd3/run_inference.py <overrides>`.
- H-bond conditioning does **not** need HBPLUS; `HBPLUS_PATH` is only needed
  for H-bond *metrics* and training featurization.
- `ca_rmsd_to_input` is written for partial diffusion **only** when input and
  output CA counts match; otherwise it is skipped silently.

## 9. Further reading (in the install)

- `skills/rfdiffusion3/SKILL.md` — full upstream skill (all recipes, every field).
- `skills/rfdiffusion3na/SKILL.md` — RNA/DNA generation; `skills/rf3/SKILL.md` — folding.
- `aurora_testing/README.md` — a 21-case Aurora validation of all three skills
  (all passing) plus the doc bugs it found.
- `models/rfd3/docs/input.md`, `models/rfd3/docs/designability_vs_diversity.md`.
