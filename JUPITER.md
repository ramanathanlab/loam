# Julich Jupiter port

Branch: `feat/julich-jupiter`, based on `develop`. The GH200 cluster launch follows
`origin/feat/cscs-alps-port`; the Jupiter endpoint and paths are separate.

## Software layout

The workspace contains `loam/` and `software/` as siblings. Every installed tool
has its own `software/<tool>/.venv`. Python 3.12.11 is uv-managed under
`software/.python/`; uv's download cache is `software/.uv-cache/`. No shell
startup files need modification. Source `scripts/jupiter_env.sh` to set shared
model caches under `software/`.

Select the `*-jupiter` skills in campaign profiles. The original skills retain
Aurora-specific instructions and historical test results; their paths, XPU
launchers and timings do not establish Jupiter readiness.

Run `../software/loam/.venv/bin/python scripts/verify_jupiter.py` to regenerate
`ports/jupiter/status.json`. This records import outcomes, package versions and
source commits. Import success does not verify model downloads, GPU kernels,
scientific outputs, or end-to-end campaign execution.

## Scheduler and endpoint

Observed on this account: ARM64 nodes, 288 Grace CPU cores and four GH200 GPUs;
Slurm account `deepwing`, partitions `booster` and `largebooster`, QOS `normal`.

The dedicated runtime is `../software/globus-compute/.venv`, with
`globus-compute-endpoint==4.20.0`, `globus-compute-sdk==4.20.0`, Loam, Academy,
Parsl and Ensemble Launcher. Installed dependencies pass `uv pip check`.
The endpoint profile is **configured but has never been started**:

`../software/globus-compute/endpoints/loam-julich-jupiter/`

This is a private template-capable endpoint running as the current POSIX user,
without privileged identity mapping. Its user template uses SlurmProvider,
SrunLauncher, and `ib0`. The default Loam request is one `deepwing` / `booster`
node for 15 minutes, four client agents plus one cluster-agent worker slot.
Initial/minimum blocks are zero, so the endpoint waits for submitted work
before requesting an allocation. EL controls GPU placement; Parsl does not pin
these five worker slots to individual GPUs. Each task uses `cuda:0` within its
assigned CUDA_VISIBLE_DEVICES. All campaign tasks must use the HPC MCP tools.

To start it yourself, on the login host where the endpoint will run:

```bash
cd /e/project1/deepwing/meigooni1/agent
source software/globus-compute/env.sh
globus-compute-endpoint start loam-julich-jupiter
```

First start can require interactive Globus authentication. Registration assigns
the real UUID and writes `endpoint.json` in that profile. No UUID was invented
or registered during configuration. Once the endpoint is running, sourcing
`loam/skills/hpc-deployment/set_env_jupiter.sh` discovers that UUID automatically.
The SDK and endpoint share the isolated GLOBUS_COMPUTE_USER_DIR, including auth
state. Do not omit the activation helper when starting this profile.

The next steps, **after the user starts the endpoint**, are:

```bash
source loam/skills/hpc-deployment/set_env_jupiter.sh
"$HPC_VENV/bin/python" "$HPC_LOAM_HOME/skills/hpc-deployment/scripts/mcp_server.py"
# In another shell, source the same env before the smoke client.
"$HPC_VENV/bin/python" "$HPC_LOAM_HOME/skills/hpc-deployment/scripts/smoke_test_client.py"
```

The MCP server is a separate local service at `http://127.0.0.1:8000/mcp`.
Starting it submits work through Globus; it was not started during this setup.
Stop it with SIGINT/SIGTERM to release the cluster cleanly. The hpcctl config
uses the dedicated Globus Python and Jupiter env file, capped to one node and
30 minutes. Queue/walltime/node overrides still obey site policy.

Configuration checks passed using the installed CLI's `render-user-config`,
its CoreEndpointConfig and UserEndpointConfig loaders, and real worker-module
imports under worker_init. No jobs were submitted by those checks. Real network
reachability, worker startup, GPU placement and concurrency require the endpoint
to be started and the MCP smoke tests to pass.

## LLM tunnel

The supplied base `http://localhost:50856/v1` is correct on `jpbl-s02-04`:
`GET /v1/models` returned the Argo list, and a real completion on
`argo:gpt-5.6-sol` returned `LOAM_BACKEND_OK` with HTTP 200. The shared Hermes
config now points the reasoner, delegation and custom provider to that base.
The HPC MCP URL stays on port 8000; the LLM tunnel is a different service.
Tournament generation/ranking URLs belong in the eventual campaign's loam.yaml;
use the same Argo base rather than the original ALCF proxy. `ALCF_PROXY=0` is
set in the Jupiter deployment environment.

## Remaining readiness gates

- The backend is verified and the shared Hermes config is updated; the eventual
  campaign tournament profile still needs its generation/ranking settings.
- The endpoint is configured and stopped. The user will start/authenticate it;
  then pass real MCP execution, filesystem and concurrency smoke tests.
- Resolve the inaccessible Reactra source and exact `genslm_esm_sae` fork; upstream
  `genslm_esm` is installed but does not establish SAE support.
- PyRosetta's public installer reports `aarch64`, which it does not support.
  A compatible licensed distribution or source build is required; installing
  `pyrosetta-installer` alone does not install PyRosetta.
- Molecular-simulations and ARM64 AmberTools 26 are installed. The environment
  script sets AMBERHOME; a Trp-cage AMBER system build passes. GPU MD still needs validation.
- Protenix's Torch 2.7.1 needs the cu128 ARM64 index; its Triton 3.3.1 pin lacks
  ARM64 wheels. `ports/jupiter/protenix-overrides.txt` records the
  compatibility override (Triton 3.8.0). Model import and the `pred --help` CLI
  pass with LAYERNORM_TYPE=torch; real inference still needs validation.
- RFD3 weights and the pinned RFD4 bundle are downloaded. Other model assets may
  download on first use; acquire them before relying on offline compute nodes.
- ESMFold2 uses the published `esm.models.esmfold2.EsmFold2Model` API in esm 3.4.2.
  The old Aurora transformers fork is unnecessary. Its x86-only flash-attention
  wheel is unavailable on ARM64; native PyTorch attention is the fallback.
- The README's `ramanathanlab/jnana` URL currently checks out an older Loam tree.
  The actual engine was found at `acadev/Jnana`, branch
  `feature/periodic-hypothesis-tournament`, and cloned into software/jnana-engine.
  Use this path for jnana_root; the tournament import passes. Its unavailable Biomni dependency is excluded in
  ports/jupiter/jnana-requirements.txt; Biomni functionality is not provided.
  LOAM_HOOK_PYTHON selects this isolated environment for Loam hook subprocesses.

Before a binder campaign, pass the real MCP run_code/filesystem/concurrency smoke,
all configured model-role completions, the local arithmetic campaign described
in `CAMPAIGN_TEST.md`, and one tiny design/sequence/refold pipeline. Choose an
explicit target and bounded design count, and retain all failed designs in the
results. No binder campaign has been launched yet.

## Campaign runtime

Hermes is provisioned through its own `pm.build_environment` into
`software/hermes-agent/.venv`, with stable Python 3.14.8 and the MCP extra.
Its `hermes --help` command passes. The PM tool store and bootstrap environment
stay under the Hermes software directory. Use that `.venv/bin` on PATH when
launching a campaign; source `scripts/jupiter_env.sh` to select the Jnana hook
interpreter. The older Python prerelease bootstrap environment is retained as
`.venv-bootstrap-rc2` and is not the selected runtime.

The full Loam test suite passes: **474 passed, 3 skipped**, including the port
and hook-interpreter regressions. The installed Globus Compute loader accepts
the rendered endpoint as a UserEndpointConfig with GlobusComputeEngine and
SlurmProvider; no endpoint was started by that validation.
This does not establish compatibility of real Hermes delegation, hooks, model
calls or Globus workers: those acceptance checks require the backend and endpoint.

## Current handoff

The dedicated Globus endpoint profile is configured and stopped, with no PID file,
registration UUID or submitted allocation. The user will perform the first start.
The LLM tunnel has passed a real inference completion. MCP execution, the arithmetic
campaign and the binder campaign remain untested. See software/globus-compute for
its environment, copied endpoint files, rendered config, user options and versions.
