---
name: proteinmpnn-cscs
description: Run ProteinMPNN inverse folding on CSCS Alps/Daint (NVIDIA GH200, aarch64, CUDA) to design amino-acid sequences for a protein backbone (PDB). CSCS/CUDA port of the Aurora `proteinmpnn` skill. Use when asked to design/redesign sequences for a structure, design selected chains, keep residues fixed, tie positions (homooligomers/symmetry), use the soluble or CA-only models, or score sequences, on Alps GH200 nodes via SLURM.
---

# ProteinMPNN on CSCS Alps/Daint (GH200, CUDA)

CSCS port of the Aurora `proteinmpnn` skill. **Upstream ProteinMPNN runs natively on
CUDA** — this is the stock `dauparas/ProteinMPNN` checkout, NOT the Aurora XPU fork, so
there is **no `intel_extension_for_pytorch` import and no IPEX shim**. The device line is
`torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`.

## Install locations (this machine)
- Code + weights (git clone, weights committed in the repo):
  `/ritom/scratch/cscs/avasan/skill-venvs/proteinmpnn-src`
  (`vanilla_model_weights/`, `soluble_model_weights/`, `ca_model_weights/`, `helper_scripts/`, `inputs/`)
- venv (python 3.12, torch 2.15.0+cu132 CUDA): `/ritom/scratch/cscs/avasan/skill-venvs/proteinmpnn`

```bash
MPNN=/ritom/scratch/cscs/avasan/skill-venvs/proteinmpnn-src
PY=/ritom/scratch/cscs/avasan/skill-venvs/proteinmpnn/bin/python
```

No `ZE_AFFINITY_MASK`, no `module load`, no PYTHONPATH shim. GPU is selected the normal
CUDA way (`CUDA_VISIBLE_DEVICES`); a whole-node Alps job sees all 4 GH200s and the code
uses `cuda:0`.

## Running on a compute node (SLURM, not the login node)

ProteinMPNN is tiny and fast; `debug` (≤30 min) is plenty. Whole-node, GH200.

```bash
#!/bin/bash
#SBATCH --job-name=mpnn
#SBATCH --partition=debug          # or low/normal for longer
#SBATCH --account=g245
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.out         # keep job I/O on /ritom/scratch (shared), NOT /tmp (node-local)
MPNN=/ritom/scratch/cscs/avasan/skill-venvs/proteinmpnn-src
PY=/ritom/scratch/cscs/avasan/skill-venvs/proteinmpnn/bin/python
$PY $MPNN/protein_mpnn_run.py \
  --pdb_path $MPNN/inputs/PDB_monomers/pdbs/5L33.pdb --pdb_path_chains "A" \
  --out_folder ./mpnn_out --num_seq_per_target 2 --sampling_temp "0.1" --seed 37 --batch_size 1
```

## Verified on Alps (2026-11-04, GH200, debug partition, job 5257564)
Single monomer 5L33 (106 aa), vanilla `v_48_020`, T=0.1, 2 sequences:
**2 sequences of length 106 generated in 0.78 s**, seq_recovery 0.40 / 0.46. PASS.
Model loads and runs on `cuda:0`; weights read from the committed `vanilla_model_weights/`.

## Everything else is identical to the Aurora skill
The multi-chain / fixed-positions / tied-positions / soluble / CA-only / score-only
workflows, all flags (`--num_seq_per_target`, `--batch_size`, `--sampling_temp`,
`--omit_AAs`, biases, `--backbone_noise`, `--save_score`, `--save_probs`, …), the
1-based chain-relative position indexing, and the output format (`OUT/seqs/<name>.fa`
with score / global_score / seq_recovery) are unchanged — see the Aurora `SKILL.md`.
Just substitute the `$MPNN` / `$PY` paths above and drop the XPU/IPEX setup.

## Troubleshooting (CSCS-specific)
- `model on device: cpu` instead of cuda: you are on the login node or no GPU is
  allocated. Run inside an sbatch/srun allocation; check `$PY -c "import torch;print(torch.cuda.is_available())"`.
- Output/weights "not found" from a compute node: make sure paths are under
  `/ritom/scratch` (mounted on compute nodes). Do not write job output to the session
  `/tmp` scratchpad — that is node-local and disappears with the job.
- There is **no** `No module named 'intel_extension_for_pytorch'` error here; if you see
  it you are pointing at the Aurora fork, not this `proteinmpnn-src` clone.
