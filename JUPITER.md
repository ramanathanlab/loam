# Jupiter deployment

For production settings, use the [scaling recipe book](docs/scaling-recipe-book.md).
It includes measured 20/256-node settings, source-backed historical results,
node/GPU density formulas, timeout scopes and reproducible staging commands.

## Controller and scientific installations

Use a checkout of this branch, Python 3.11+ (measured controller: Python 3.12),
and dedicated installed environments for each scientific tool and native vLLM.
The controller can be installed with `pip install -e '.[dev,hpc]'` or
`uv sync --extra dev --extra hpc`. Add the `reports` extra for plotting/audits.
`requirements-jupiter.txt` records the original controller environment; `uv.lock`
locks the current package and its optional dependencies.

Export `LOAM_SOFTWARE_ROOT` to the site's software tree and keep model weights
and environments there. The supplied assay expects `RFD4-Proteina`,
`proteinmpnn`, `molecular-simulations`, `ambertools`, `jnana-engine`,
`nemotron-vllm` and `redis` installations. Read the individual skills for other
tool-specific dependencies and limitations. Inference/compiler/runtime paths
can be overridden; model weights must be staged before compute jobs because
Jupiter compute nodes have no Internet. Scientific settings are not reduced
by the port. PyRosetta, for example, requires a licensed ARM64 installation.

`scripts/jupiter_env.sh` scopes cache paths, Amber tools and the RFD4 runtime
libraries to `LOAM_SOFTWARE_ROOT`. Source it inside each scientific subprocess.
Preserve the scheduler's assigned `CUDA_VISIBLE_DEVICES`; a one-GPU row uses
`cuda:0` within that mask. Active skills refer to `$HPC_LOAM_HOME` and
`$LOAM_SOFTWARE_ROOT` rather than a particular user's installation.

## One allocation for inference and science

```bash
export LOAM_SOFTWARE_ROOT=/shared/your/software
export LOAM_SCIENCE_INPUTS=/shared/science/egfr/inputs
export PYTHONPATH="$PWD:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
bash examples/scaling/run-20i.sh /shared/scratch/new-experiment \
  --account YOUR_ACCOUNT --partition booster
```

This stages a new job without submitting it. Inspect its `plan.json`, generated
`job.sh`, source/input seals and `preflight.json`. Add `--submit` to a fresh
staging invocation when ready. Use a fresh `LOAM_RUN_LABEL` for every attempt;
existing jobs/submission intents are refused. A potentially lost submission
acknowledgement must be reconciled, not repeated automatically.

The parent allocation hosts private Redis, inference replicas, the gateway,
science MCP/Ensemble Launcher, controller and all-GPU telemetry. No Globus
endpoint, Argo, Tara, login tunnel or second queued allocation is required for
this route. Child steps are restricted to their assigned role hostnames. A
source snapshot is created before queueing; startup checks its hashes.
Credentials and model-startup clocks begin after allocation starts. Cleanup
terminates only this controller's own child process groups.

The measured Jupiter Slurm settings were account `deepwing`, partition
`booster`, four GPUs and 288 CPU cores per node. `bne_deepwing` was an earlier
short-lived reservation, not a portable queue/account default. Specify site
account, partition and an actually valid reservation explicitly. Other node
GPU/CPU counts can be configured for the Slurm/CUDA adapter, but model memory,
launch integration, scientific numerical behavior and telemetry integrators
need qualification on each new machine.

## Optional external-provider / Globus route

`launch.sh` and root `loam.yaml` retain the separate externally hosted model
route. The configuration supports `${LOAM_REPO_ROOT}`, `${LOAM_SOFTWARE_ROOT}`
and `${LOAM_RUN_ROOT}` placeholders; relative paths resolve against its directory.
Set these variables and configure your own Globus endpoint and model URLs.
`./launch.sh --check` performs offline checks; normal execution can allocate
managed HPC resources. It is distinct from the single-allocation launcher.

For Argo/Tara access from isolated compute nodes, the tunnel helper is
`scripts/inference_tunnel.sh`. Authentication is via `alcf-tokens` in an isolated
environment (`requirements-alcf-auth.txt`); tokens are referenced through
environment variables and must never enter Git.

Public source control excludes scientific structures, original research prompts,
proposal texts, authentication stores and private campaign artifacts. The current
computer's preserved private reference is outside this checkout. Supply your own
approved input bundle and optional proposal file; see the recipe book.
