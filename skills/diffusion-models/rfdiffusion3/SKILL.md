---
name: rfdiffusion3
description: Generate protein backbones with RFdiffusion3 (all-atom diffusion) from a JSON spec — unconditional monomers, binders with hotspots, motif/enzyme scaffolding, symmetric oligomers and partial diffusion. Use for interactive single-device backbone generation; for backbone generation at width, and for the flow-matching alternative, see the rfd4-proteina skill.
systems: [aurora]
---

# RFdiffusion3 (RFD3)

## Before you run it: read the reference file for this System

The install, the environment, device selection, the entrypoint and what is known to work differ per
System. Read `references/<HPC_SYSTEM>.md` (for example
`read_skill(name="rfdiffusion3", file="references/aurora.md")`) and follow it; it is authoritative
over this file for anything System-specific. This skill has a reference file for **aurora** only. It
is **unsupported** on any other System — the Jupiter port disabled it, and an installed CLI is not
validated inference — so report the step as blocked rather than adapting Aurora's instructions, and
use the `rfd4-proteina` skill for executable backbone generation there. Missing prerequisites are
not a scientific refutation.

Software is found only through Site profile variables: the checkout is under `$LOAM_SOFTWARE_ROOT`
and checkpoints under `$FOUNDRY_CHECKPOINT_DIRS`. Never edit the checkout; write outputs elsewhere.

**Fan-out is a per-System question.** Under loam's `run_batch` the Aurora route is unresolved (see
the reference file), so RFD3 is for interactive, one-device work until that is fixed. Register a row
script with `register_tool(..., expects=["*.cif.gz"])` and submit with `tool=<name>`, so a row that
exits clean without writing a structure is counted as the failure it is.

## Spec format

The spec maps **name → design configuration**; each top-level key is an independent job whose name
appears in the output filenames. Run a subset with `json_keys_subset=[key1,key2]`.

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
- Examples and the full field table live in the checkout: `models/rfd3/docs/examples/` and
  `models/rfd3/docs/input.md`.

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
| `strategy` | per System | the Lightning strategy the engine is forced onto |

Other engine knobs (`inference_sampler.step_scale` 1.5 — higher = less diverse, more
designable; `gamma_0` 0.6 — lower = more designable; `low_memory_mode` for OOM;
`dump_trajectories`, large) are set in `build_engine`. Defaults live in
`models/rfd3/configs/inference_engine/rfdiffusion3.yaml` and `.../base.yaml`.

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

`run_design` returns `{ok, produced, metrics, error, host, strategy_applied}` and appends stage
markers to `<out_dir>/rfd3_stages.jsonl`, so a row killed by the scheduler's wall still shows how
far it got. `ok` is true only when a structure was written: a clean exit that produced nothing is
not a success.

## Must-know

- A generated backbone is not a validated design. Read the metrics JSON, then sequence the backbone
  (`proteinmpnn`) and refold it (`chai`, `boltz`) before claiming anything about a design.
- Do not swap the framework or Python environment the checkout's venv was built against; precision
  and kernels follow from it.
