# ProteinMPNN sequence design on Jupiter

Use the native CUDA checkout at `software/proteinmpnn`; the Aurora IPEX shim and XPU fork instructions do not apply. ProteinMPNN designs sequences on a fixed backbone; it does not generate backbones or prove binding.

## mmCIF-to-PDB conversion dependency

ProteinMPNN expects PDB input. RFD4 and folding tools may write mmCIF instead; the conversion helper requires Gemmi in the ProteinMPNN environment itself:

```bash
bash "$HPC_LOAM_HOME/scripts/install_jupiter_proteinmpnn_extras.sh"
```

This installs `gemmi==0.7.5` from the branch's pinned conversion requirements without replacing the torch/numpy stack. During a campaign, prerequisites must already be installed; agents should report missing dependencies rather than modify shared software.

Execute conversion through MCP with ProteinMPNN's absolute Python path, preserving the original input:

```python
import gemmi
structure = gemmi.read_structure(input_cif)
structure.write_pdb(output_pdb)
```

Set `input_cif` and `output_pdb` to real absolute campaign paths, with a fresh output path. Check the source and converted structure have the same intended protein chains, residue identities and finite atom coordinates. Preserve the target/binder mapping; do not infer which chain to design solely from chain order. PDB format has stricter identifier/size limits than mmCIF, so audit any renumbering or dropped information before using the result. Import success alone does not verify conversion.

## Target/binder chain control

For a complex, explicitly identify the generated binder chain(s) and keep the target sequence fixed. With single-PDB input use `--pdb_path_chains "$DESIGN_CHAINS"` (the chains to DESIGN). An empty selection can redesign every chain. For JSONL workflows, parse the PDB and use `helper_scripts/assign_fixed_chains.py --chain_list` to select design chains; inspect the generated JSONL before inference.

```bash
"$PY" "$REPO/protein_mpnn_run.py" \
  --pdb_path "$INPUT_PDB" --pdb_path_chains "$DESIGN_CHAINS" \
  --out_folder "$OUT" --num_seq_per_target 1 \
  --sampling_temp 0.1 --batch_size 1 --seed 42
```

Set the variables from actual campaign artifacts. If converting RFD4 mmCIF to PDB, retain target/binder identity and record the chain/residue mapping. The first FASTA record is the native/reference sequence; select a generated sample, not that first record. Outputs are under `OUT/seqs/`; verify the target stayed native and that the designed binder sequence matches the intended chain.

Fixed-position helper indices are 1-based positions within each parsed chain sequence, not arbitrary PDB residue identifiers. `num_seq_per_target` must be divisible by batch size. Scores are sequence negative log-likelihoods, not binding free energies. Inspect weights, device use, seed, sampled temperatures and saved FASTA before claiming success. The native checkout selects CUDA when available; no IPEX stand-in is required.

## Jupiter environment and execution contract

Run scientific workloads only through the campaign's HPC MCP tools, in the assigned workdir. Use absolute campaign paths, preserve `CUDA_VISIBLE_DEVICES`, and use `cuda:0` inside a one-GPU task. Do not submit scheduler jobs, use SSH, replace the scientific software, or apply Aurora XPU shims from an agent task. CPU-only tools do not need a GPU solely to import or score.

Source `$HPC_LOAM_HOME/scripts/jupiter_env.sh`, set `REPO` to the software directory specified in the main SKILL.md, and use its `.venv/bin/python` as `PY`. Explicitly activate that environment before upstream shell wrappers to prevent inheritance of the Globus worker's Python. Inputs/outputs in the examples are variables to set from real campaign artifacts, not preexisting files supplied by this reference.

For campaign execution, stage with `write_files` and use independent `run_batch(..., timeout_each=6000)` rows, with one design/run per task; `run_code` is reserved for staging fallback, harvesting and diagnosis. Use CPU/GPU resources appropriate to the tool. Stream logs directly to workspace files, preserve failures, and check sufficient allocation time remains. Queue waits are separate; do not cancel queued work just because it waits.

## Further packaged reading

- [Original API and provenance notes](aurora-notes.md) — historical, not a Jupiter launcher.
- [Resource index](resource-index.md) — exact `skill_view` paths for all packaged files.

Documentation accessibility does not establish software installation, checkpoint access, successful inference, or scientific validity. Check the main skill's readiness limitations and the actual runtime outputs.

## Campaign batch execution after the develop integration

For a delegated campaign code-generator, stage the script and input JSONL together with `mcp__hpc__write_files`, then run all science through `mcp__hpc__run_batch` (a one-row batch is valid). Poll `batch_status` until `unfinished == 0` and inspect actual recorded outputs before reporting success. Use `gpus=1` for GPU work, preserve assigned `CUDA_VISIBLE_DEVICES`, and activate this tool's own venv inside each row.

Use `timeout_each=6000` and a subprocess deadline no larger than 6000 seconds per scientific row. Keep the 6300-second MCP and 14400-second Hermes outer deadlines; allocation walltime still limits completion. `run_code` is for bounded staging fallback, harvesting/manifests and post-error diagnosis. Loam counts its calls (four before a batch, ten total after submitting one), so it is not a serial scientific execution path.

`write_files` may stage only under the campaign's `HPC_BASE_DIR`; additional read-only inspection roots do not grant write access to installed software. The operator's manual installation smoke suite is separate from delegated campaign execution and may still use `run_code`; no software tests were added to campaign preflight.
