---
name: protenix
description: Predict biomolecular structures (proteins, DNA/RNA, ligands, ions, complexes) with Protenix, an AlphaFold3-style diffusion model, writing mmCIF plus pLDDT/pTM/ipTM/PAE JSON. Use for AF3-like complex or single-chain prediction, including fast no-MSA runs.
systems: [aurora, jupiter]
---

# Protenix

## Before you run it: read the reference file for this System

The environment, kernel options and launch command differ per System. Read
`references/<HPC_SYSTEM>.md` (for example `read_skill(name="protenix", file="references/aurora.md")`)
and follow it; it is authoritative over this file for anything System-specific. This skill has
reference files for **aurora** and **jupiter**. It is **unsupported** on any other System: report the
step as blocked rather than adapting another System's instructions.

Software is found only through Site profile variables: the install is under `$LOAM_SOFTWARE_ROOT`,
and model data under `$PROTENIX_ROOT_DIR`. Never edit the install directory; write outputs to the
campaign workspace.

## Input

A JSON list of tasks (example: `examples/example_gb1.json`). Each task has a unique `name` and a
`sequences` list:

```json
[{
  "name": "gb1",
  "sequences": [
    {"proteinChain": {"sequence": "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE", "count": 1}}
  ]
}]
```

Other entity types: `dnaSequence`, `rnaSequence`, `ligand` (`"CCD_ATP"` or a SMILES string), `ion`.
`count` sets copies; optional per-chain `pairedMsaPath` / `unpairedMsaPath` (absolute a3m paths),
`templatesPath`, `modifications`. Full spec: `<install>/docs/infer_json_format.md`; examples in
`<install>/examples/`. `protenix json -i x.cif -o dir` converts a PDB/mmCIF into input JSON.
Provide authoritative campaign sequences and record entity/chain identity.

## Arguments (`protenix pred`)

| flag | default | meaning |
|---|---|---|
| `-i/--input` | — | JSON file or directory |
| `-o/--out_dir` | ./output | output directory |
| `-s/--seeds` | 101 | comma list, e.g. `101,102` |
| `-c/--cycle` | 10 | Pairformer recycles |
| `-p/--step` | 200 | diffusion steps |
| `-e/--sample` | 5 | samples per seed |
| `-d/--dtype` | bf16 | `bf16` or `fp32` |
| `-n/--model_name` | protenix_base_default_v1.0.0 | must be a model whose checkpoint is present locally |
| `--use_msa` | **true** | `false` for offline/fast runs; `true` needs network access |
| `--trimul_kernel`, `--triatt_kernel` | cuequivariance | `torch` when the fused kernels are unavailable (see the reference file) |
| `--use_template`, `--use_rna_msa` | false | need kalign/hmmer + databases |
| `--use_default_params` | — | load the model's recommended cycles/steps |
| `--use_seeds_in_json` | — | take seeds from the input JSON |
| `--need_atom_confidence` | false | adds full per-atom pLDDT / PAE arrays |

Check the installed `protenix pred --help` before relying on option names.

## Outputs

```
<out_dir>/<name>/seed_<seed>/predictions/
  <name>_sample_<k>.cif                       # one structure per sample
  <name>_summary_confidence_sample_<k>.json   # plddt (0-100), ptm, iptm, ranking_score,
                                              # chain_* / chain_pair_* (iptm, pae_mean, plddt),
                                              # has_clash, disorder
```

Rank samples by `ranking_score`. `iptm` is 0 for single chains.

## Must-know

- `--use_msa` defaults to **true**, which triggers a network MSA search. Pass `--use_msa false`
  for offline runs.
- It writes `./esm_embeddings/` into the current directory, so `cd` to a scratch dir in the
  campaign workspace first and never run from inside the software dir.
- Model load dominates small jobs. Batch many inputs into one JSON or directory per call.
- If a run sits silent with no log output, check for model/asset downloads before diagnosing a
  stall, and set `PYTHONUNBUFFERED=1` to get logs live.
- Import or `--help` checks do not validate inference: inspect the actual mmCIF and confidence JSON,
  including the expected entities.
