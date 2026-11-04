---
name: rfd4-proteina-cscs
description: Run RFD4-Proteina (Baker Lab x NVIDIA flow-matching all-atom protein design) on CSCS Alps/Daint (NVIDIA GH200, aarch64, CUDA). CSCS/CUDA port of the Aurora `rfd4-proteina` skill, using the native .venv install built from a local clone. Covers the aarch64 make-setup fix, the environment, and the gated-checkpoint caveat. Use to generate/refine protein designs with RFD4 on an Alps GH200 node.
---

# RFD4-Proteina on CSCS Alps/Daint (GH200, CUDA)

CSCS port of the Aurora `rfd4-proteina` skill. Unlike the Aurora Intel-XPU install, this is
the project's **own native CUDA-13 `.venv`** built by its `setup.sh` on the GH200. The repo
ships its own in-repo skills (`skills/rfd4-proteina-{setup,generate,search,evaluate,…}`) and
`AGENTS.md` — follow those for task semantics; this doc only records the CSCS specifics.

## Install location (this machine)
- Repo clone: `/ritom/scratch/cscs/avasan/skill-venvs/software/RFD4-Proteina`
- Native venv (python 3.12, **torch 2.14.1+cu132**, CUDA): `<repo>/.venv`
  (entry points `.venv/bin/rfd4-generate`, `.venv/bin/rfd4-evaluate`)

```bash
cd /ritom/scratch/cscs/avasan/skill-venvs/software/RFD4-Proteina
source .venv/bin/activate        # or call .venv/bin/rfd4-generate by absolute path
```
No `module load`, no `ZE_AFFINITY_MASK`, no ALCF proxy. CUDA is auto (GH200).

## REQUIRED aarch64 fix to `make setup`

Out of the box `make setup` → `setup.sh` → `scripts/setup_venv.sh --runtime` **aborted on
Alps** with:
> The native CUDA 13 dependency profile currently supports Linux x86_64 only.

because the runtime guard in `scripts/setup_venv.sh` only allowed `x86_64`. The cu132 torch
backend it installs has working aarch64/GH200 wheels, so the guard was too strict. The fix
(applied here) widens it to accept aarch64:

```bash
# scripts/setup_venv.sh
host_arch="$(uname -m)"
if [[ "$runtime" == true && ( "$(uname -s)" != Linux || ( "$host_arch" != x86_64 && "$host_arch" != aarch64 ) ) ]]; then
    echo "The native CUDA 13 dependency profile currently supports Linux x86_64 and aarch64 only." >&2
    exit 1
fi
```

With that, the full native env (`uv pip install --torch-backend cu132 -e ".[all-cuda13]"`)
resolves and installs on aarch64 — **216 packages, torch 2.14.1+cu132, `rfproteina`
imports, `rfd4-generate`/`rfd4-evaluate` present** (verified 2026-11-04 on the GH200 login
node). If you re-clone, re-apply this one-line guard change before `make setup`.

> `Makefile` `test` target also hardcodes `/lib/x86_64-linux-gnu/libstdc++.so.6` for its
> `LD_PRELOAD`. Point it at the aarch64 path (`/lib/aarch64-linux-gnu/libstdc++.so.6`) if you
> run `make test`. Not needed for generation.

## Checkpoints (gated — provide access)

`make setup` also runs `scripts/download_assets.sh` into `$DATA_PATH` and the configured
release checkpoints are pulled from the **Hugging Face Hub** (`configs/inference.yaml`
`hf_ckpt_path`). These are **gated** — set a valid `HF_HOME`/HF token with access, or point
`DATA_PATH` at a local copy of the weights, before generation. The environment install
itself does not need the token.

## Running generation on a compute node (SLURM)

`rfd4-generate` uses Hydra `key=value` args (not `--key value`). Keep job I/O on
`/ritom/scratch` (shared), never the node-local `/tmp`.

```bash
#!/bin/bash
#SBATCH --job-name=rfd4
#SBATCH --partition=debug          # low/normal for real runs
#SBATCH --account=g245
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=/ritom/scratch/cscs/avasan/skill-verify/logs/rfd4-%j.out
cd /ritom/scratch/cscs/avasan/skill-venvs/software/RFD4-Proteina
source .venv/bin/activate
# preflight + smoke (from AGENTS.md); needs checkpoints present:
bash skills/rfd4-proteina-setup/scripts/preflight.sh
rfd4-generate inputs=data/benchmarks/monomer/monomer-short.cif.zst num_replicates=1 out_dir=./runs/smoke
```

## Status on Alps (2026-11-04)
- **Environment: VERIFIED** — native cu132 `.venv` builds and imports on aarch64/GH200 after
  the guard fix; torch sees the GH200 (`cuda True`).
- **Generation: PENDING WEIGHTS** — a real `rfd4-generate` run was not executed here because
  the release checkpoints are HF-gated and no token/`DATA_PATH` was configured. Once weights
  are present, run the preflight + smoke above (see the in-repo `rfd4-proteina-setup` and
  `rfd4-proteina-generate` skills).

## Differences from the Aurora skill
- Native CUDA `.venv` (torch 2.14.1+cu132) instead of the Aurora XPU frameworks venv; no
  `ZE_AFFINITY_MASK`, no lightning `seed_everything` shim, no ALCF proxy.
- `make setup` needs the one-line aarch64 guard fix above.
- Hydra CLI, config roots (`configs/inference.yaml`), task recycle schedules, and I/O layout
  are unchanged — use the repo's own skills/docs.
