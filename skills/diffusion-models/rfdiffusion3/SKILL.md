---
name: rfdiffusion3
description: Generate protein backbones with RFdiffusion3 (all-atom diffusion) on Aurora Intel XPU — unconditional monomers, binders with hotspots, motif/enzyme scaffolding, symmetric oligomers and partial diffusion, from a JSON spec. Use for backbone/binder generation on Aurora; for the flow-matching alternative see the rfd4-proteina skill.
---

# RFdiffusion3 (RFD3) on Aurora (Intel XPU)

> **UNRESOLVED under `run_batch`: rows hang, do not use this for width.**
> Every row stalls inside `initialize()` right after
> `foundry.utils.ddp: Intel XPU detected - using XPU accelerator`, then burns its
> whole `timeout_each`. Measured 2026-11-11 at 48-row width: 46 rows hung with
> leases held, 2 exited `rc=15` at 142s, no output. `trainer.strategy=xpu_single`
> (below) removes the MPI *abort* but NOT the stall, so it makes the failure slow
> and silent instead of fast and loud.
>
> **For backbone generation at width use the `rfd4-proteina` skill instead.** Its
> resident-worker route is verified under `run_batch` and produced 48/48 real
> backbones at ~290s/row on 2026-11-11. Use RFD3 only interactively, on one tile,
> where a stall is visible and cheap.

Install: `/flare/FRAME-IDP/msinclair/software/foundry`. Do not edit it; write outputs elsewhere.

## How to run

Use `scripts/run_rfd3_api.py`. It drives the engine in-process and is the only
supported entrypoint here.

```python
import sys; sys.path.insert(0, "<this skill>/scripts")
from run_rfd3_api import run_design

rec = run_design(spec="/abs/spec.json", out_dir="/abs/out", seed=1234, timesteps=200)
# rec["ok"] is True only if a structure was written; rec["produced"] lists the CIFs.
```

or as a command: `python run_rfd3_api.py --spec SPEC.json --out-dir DIR [--seed N]`.

One process per XPU tile. In a `run_batch` row use `gpus=1` and let the row inherit
its pinned `ZE_AFFINITY_MASK`. Register the row script once with
`register_tool(..., expects=["*.cif.gz"])` and submit with `tool=<name>`, so a row
that exits clean without writing a structure is counted as the failure it is.

`run_design` returns `{ok, produced, metrics, error, host, strategy_applied}` and
appends stage markers to `<out_dir>/rfd3_stages.jsonl`, so a row killed by the
PBS wall still shows how far it got.

## Spec format

Spec maps **name → design configuration**; each top-level key is an independent
job whose name appears in the output filenames. Run a subset with
`json_keys_subset=[key1,key2]`.

```json
{ "uncond50": { "length": "50" },

  "pdl1": {                                   // binder against a target, steered by hotspots
    "input": "/abs/path/5o45_cropped.pdb",
    "contig": "50-120,/0,A17-131",
    "infer_ori_strategy": "hotspots",
    "select_hotspots": {"A56": "CG,OH", "A115": "CG,SD", "A123": "CD2,OH"},
    "is_non_loopy": true },

  "mysite": {                                 // motif / active-site scaffolding with cofactors
    "input": "/abs/path/M0255_1mg5.pdb",
    "ligand": "NAI,ACT",
    "unindex": "A108,A139,A152,A156",
    "length": "180-200",
    "select_fixed_atoms": {"A108": "ND2,CG", "A139": "OG,CB,CA", "A152": "OH,CZ",
                           "A156": "NZ,CE,CD", "ACT": "OXT", "NAI": ""},
    "allow_ligand_on_existing_chain": true }
}
```

- **Contigs** are comma-separated segments, `/0` = chain break. `A40-60` takes residues
  40-60 of chain A from `input` as a fixed motif; `70` designs exactly 70 residues bonded
  to the previous segment; `60-80` designs a random length in that range. `length`
  (`"min-max"` or int) constrains the total.
- **Symmetric oligomer:** add `"symmetry": {"id": "C5"}` and set `inference_sampler.kind=symmetry`.
- **Partial diffusion:** add `"partial_t": <N>` (start ~2, typical 5-15) to re-noise a structure.
- **InputSelection** fields (`select_fixed_atoms`, `select_unfixed_sequence`, `select_buried`,
  `select_partially_buried`, `select_exposed`, `select_hbond_donor`, `select_hbond_acceptor`,
  `select_hotspots`) accept `true`/`false`, a contig string, or a `residue-range → atoms` dict
  with the shorthands `ALL`, `BKBN` (= `N,CA,C,O`), `TIP` and `""` (nothing).
- Examples: `$FOUNDRY/models/rfd3/docs/examples/`; full field table: `models/rfd3/docs/input.md`.

### Residue ranges must come from the structure

RFD3 validates every component against the atom array and rejects the whole spec
(`ValidationError ... [component=A615] Residue A615 not found in atom array`), so a
guessed range costs the entire batch. Crystal numbering has offsets and gaps — the
UniProt length is not the observed range. Compute the real blocks and build the
contig from them, and check each hotspot falls inside one:

```python
import gemmi
st = gemmi.read_structure(cif); st.remove_ligands_and_waters(); st.setup_entities()
def blocks(ch):                      # contiguous observed amino-acid runs
    ns = sorted({r.seqid.num for r in ch
                 if (t := gemmi.find_tabulated_residue(r.name)) and t.is_amino_acid()})
    out, s, p = [], ns[0], ns[0]
    for n in ns[1:]:
        if n == p + 1: p = n
        else: out.append((s, p)); s = p = n
    return out + [(s, p)]
print({ch.name: blocks(ch) for ch in st[0] if len(ch)})
```

## Arguments

`run_design(spec, out_dir, seed, timesteps, diffusion_batch_size, n_batches, ckpt, strategy)`.
Total designs = `n_batches × diffusion_batch_size`.

| argument | default | meaning |
|---|---|---|
| `spec` / `out_dir` | required | spec JSON; output dir (created if missing) |
| `n_batches` / `diffusion_batch_size` | 1 / 1 | total designs = the product |
| `timesteps` | 200 | denoising steps; lower = faster/rougher |
| `ckpt` | `rfd3` | registry alias or absolute path |
| `seed` | 0 | reproducibility |
| `strategy` | `xpu_single` | single-device; see Must-know |

Other engine knobs (`inference_sampler.step_scale` 1.5 — higher = less diverse, more
designable; `gamma_0` 0.6 — lower = more designable; `low_memory_mode` for OOM;
`dump_trajectories`, large) are set in `build_engine`. Defaults live in
`models/rfd3/configs/inference_engine/rfdiffusion3.yaml` and `.../base.yaml`.

Checkpoints in `/flare/FRAME-IDP/msinclair/software/checkpoints`: `rfd3_latest.ckpt`
(alias `rfd3`, default), `rfd3na_1190.ckpt` (nucleic acids),
`rf3_foundry_01_24_latest_remapped.ckpt` (RF3 prediction), `proteinmpnn_v_48_020.pt`.

## Outputs

```
<out_dir>/<spec-basename>_<spec-key>_<batch>_model_<n>.cif.gz
<out_dir>/<spec-basename>_<spec-key>_<batch>_model_<n>.json
```

`n` starts at 0. The JSON carries a `metrics` block (`n_chainbreaks`, clash counts,
`radius_of_gyration`, secondary-structure fractions, `num_residues`) — the quickest
sanity check on a design — and echoes the full spec including the *sampled* contig, so
variable-length runs are reproducible. CIF B-factors carry the sequence head's
per-token confidence.

## Must-know

- **What is known about the hang.** Nothing in the inference path overrides
  `trainer.strategy`, so it inherits DDP from the checkpoint's training config and
  `fabric.launch()` initialises MPI. Inside a `run_batch` row (already under EL's
  `MpiExecLauncher`) that MPI job cannot route over the Slingshot NIC, and the row
  died with `MPIDI_OFI_handle_cq_error ... nic=cxi0: No route to host`, rc=143.
  `run_rfd3_api.build_engine` sets `trainer.strategy=xpu_single` in the only window
  where it is reachable — after construction, before `initialize()` — and that DOES
  remove the `cxi0` abort. It does NOT make the row complete: it then stalls in
  `initialize()` indefinitely. Dropping `PMI_*`/`PMIX_*`/`PALS_*` from the row
  environment also removes the abort and also does not fix the stall.
  Still unidentified: what `initialize()` blocks on once distributed init is
  sidestepped — checkpoint load off Lustre and a distributed barrier are both
  consistent with the evidence, and nobody has instrumented it. Resolve that before
  trusting RFD3 at width again.
- **Do not swap frameworks modules.** The venv is built against `frameworks/2026.1.0`,
  the node default, and takes torch/lightning from it via
  `include-system-site-packages`. Precision stays `bf16-mixed`.
- **`set -u` + Lmod don't mix.** A wrapper with `set -euo pipefail` dies in Lmod's init;
  use `set -eo pipefail`.
