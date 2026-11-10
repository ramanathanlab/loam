# Julich Jupiter port

Branch: `feat/julich-jupiter`, based on `develop`. The GH200 cluster launch follows
`origin/feat/cscs-alps-port`; the Jupiter endpoint and paths are separate.

## Required Hermes version: 0.20.6

The active Hermes installation is pinned to upstream **0.20.6**, calendar tag
`v2026.8.27`, commit `5fc308a70719a83cccdbba4c0e39c23f5a8239d5`.
`software/hermes-agent` is a symlink to `software/hermes-agent-0.20.6`, whose
`.venv` uses Python 3.12.11 and the release's locked dependencies with the MCP
extra. Existing launchers continue using `software/hermes-agent/.venv/bin/hermes`.
Check it with `../software/hermes-agent/.venv/bin/hermes --version` from this repo.
Do not run `hermes update` or apply the newer gateway patch to this pinned release.

The previous development checkout, including its local gateway fix, is preserved
at `software/hermes-agent-newer-backup`. The gateway patch and Python 3.14 notes
later in this document describe that previous installation; they are historical,
not setup instructions for 0.20.6. Campaign configuration and state are retained.
Machine-readable pin: `ports/jupiter/hermes-version.json`. Version, MCP and Loam
adapter imports passed; a real Argo CLI completion returned `HERMES_0206_OK`.

## Software layout

The workspace contains `loam/` and `software/` as siblings. Every installed tool
has its own `software/<tool>/.venv`. Python 3.12.11 is uv-managed under
`software/.python/`; uv's download cache is `software/.uv-cache/`. No shell
startup files need modification. Source `scripts/jupiter_env.sh` to set shared
model caches under `software/`.

After installing the RFD4 Python environment, install its native Open Babel runtime:

```bash
# From the workspace containing loam/ and software/:
bash loam/scripts/install_jupiter_rfd4_runtime.sh
source loam/scripts/jupiter_env.sh
```

This uses micromamba at `software/micromamba/bin/micromamba` (or
`MICROMAMBA_EXE`) to download the exact Linux ARM64 packages in
`ports/jupiter/rfd4-runtime-linux-aarch64.explicit.txt`, with package hashes. It installs
`libXrender`, `libX11`, `libXext`, XCB and their runtime dependencies into
`software/RFD4-Proteina/runtime/`; the Python `.venv` is unchanged.
`jupiter_env.sh` adds that dedicated library directory to `LD_LIBRARY_PATH`.
The package provenance and SHA-256 hashes are recorded in
`ports/jupiter/rfd4-runtime-packages.json`. Re-running the installer is safe.

The login node supplies libraries that compute nodes do not. Validate
`from openbabel import pybel`, `rfd4-generate --help`, and an actual generation
through MCP on a compute node after installation. These are manual installation
checks, not additional campaign preflight gates.

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

### Argo tunnel when SSH agent forwarding is unavailable

Symptom: Jupiter has an empty `SSH_AUTH_SOCK`; `ssh-add -l` reports
"Could not open a connection to your authentication agent". The direct homes
connection fails at `mmeigooni@logins.cels.anl.gov` with "Permission denied
(publickey)", followed by "Connection closed by UNKNOWN port 65535". In the
observed case CELS recognized the public key, but Jupiter could not sign with
it. This does not by itself establish a site-wide ban on agent forwarding.

The working alternative starts both SSH connections on the laptop where the
CELS and Jupiter keys are loaded. No private key needs to be copied to Jupiter.

1. In laptop terminal 1, expose homes' Argo port on laptop loopback port 50855:

   ```bash
   ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
     -J mmeigooni@logins.cels.anl.gov \
     -L 127.0.0.1:50855:localhost:50856 \
     mmeigooni@homes-02.cels.anl.gov
   ```

2. In laptop terminal 2, expose that laptop port on the campaign login node:

   ```bash
   ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
     -R 127.0.0.1:50856:127.0.0.1:50855 \
     <Jupiter-SSH-destination>
   ```

   Replace the placeholder with a destination that lands on **jpbl-s02-04**,
   where this campaign runs. If another Jupiter host is required, specify it
   with `-J` so the final SSH server is jpbl-s02-04. A forward on another login
   node is not available through jpbl-s02-04's localhost. This requires SSH
   remote port forwarding to be permitted; it does not require agent forwarding.

3. Keep both laptop sessions open and verify on jpbl-s02-04:

   ```bash
   curl --fail --max-time 10 http://localhost:50856/v1/models
   ```

This bridge was verified by a successful model-list request on 2026-11-06.
The route is Jupiter:50856 -> laptop:50855 -> homes-02:50856. Loam continues
using `http://localhost:50856/v1` in `.hermes/.loam-home/config.yaml` and the
campaign's generation/ranking settings. A successful model list checks the
route; campaign preflight must still make real inference requests.

Do not run the old `~/.alias` `tunnel` function alongside this bridge: it tries
the direct Jupiter-to-homes SSH route and cannot repair missing authentication.
If a port is occupied, identify its existing listener before changing anything.
If a laptop connection ends or sleeps, restore both connections and recheck the
URL before resuming a campaign. Redis on 50857 and MCP on 8000 are separate
services; restoring Argo does not replace their compute-node checks.

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

### Academy messaging through login-node Redis

Jupiter uses `RedisExchangeFactory` because compute nodes cannot reach the public
Academy HTTP exchange. Start the private Redis service on the same login node as
the campaign MCP server:

```bash
software/globus-compute/.venv/bin/python loam/scripts/start_jupiter_redis.py --bind 10.128.1.8
source software/redis/env.sh
software/globus-compute/.venv/bin/python loam/skills/hpc-deployment/scripts/redis_exchange_smoke.py
```

Paths above are relative to the workspace containing `loam` and `software`. Use
the current login node's compute-reachable internal address for `--bind`. Jupiter's
setup script loads `software/redis/env.sh` automatically. Redis listens only on
loopback and that internal address, requires a randomly generated password stored
in a private file outside Git, and disables persistence. It uses unencrypted TCP
on the cluster network. Keep Redis running throughout the campaign.

The Academy factory is serialized to Globus workers, allowing remote actions and
results over Redis without Academy cloud login. Globus Compute authentication is
still required. This does not provide internet access for database tools or Argo.
Other deployments continue to default to HTTP; select explicitly with
`ACADEMY_EXCHANGE_TYPE=http`. Redis configuration uses `ACADEMY_REDIS_HOST`,
`ACADEMY_REDIS_PORT`, `ACADEMY_REDIS_DB`, and `ACADEMY_REDIS_PASSWORD_FILE`.

The conda Redis binary's allocator is incompatible with Jupiter's page size. Build
Redis from source with `make MALLOC=libc` and install `src/redis-server` under
`software/redis/bin`; the Python client environment lives at `software/redis/.venv`.

Redis transport was verified in `campaign-test2` on 2026-11-06: the login-node
manager on `jpbl-s02-04` dispatched code to compute node `jpbo-090-29` in Slurm
job 2726820, inspected the shared campaign directory, and received concurrent
results from GPU-pinned agents. The two three-second tasks overlapped by 2.70
seconds. Evidence is in `campaign-test2/state/redis-transport-test.json` outside
the repository. Redis Python 8 needs `socket_timeout=None` explicitly for
Academy's blocking message listener; connection establishment remains bounded.

### Hermes campaign startup checks

The installed Hermes build runs interactive sessions through a gateway. It does
not load the campaign's project-local `.hermes/plugins` shim in that gateway.
Register the generated `loam-hooks` shim under the campaign's
`.hermes/home/plugins/loam-hooks` as well; a symlink to
`<campaign>/.hermes/plugins/loam-hooks` keeps the generated shim authoritative.
After installing it, reload gateway plugins and start a fresh campaign session.
Keep any gateway restart scoped to the campaign home; the HPC MCP server and
Globus endpoint are separate processes and should remain running.

The Hermes Python environment also needs `PyYAML` for Loam's profile loader.
Without it Loam falls back to an `environment-override` profile with tournament
and orchestration disabled. Verify the run manifest identifies the campaign
profile and has both features enabled before claiming that scientific work has
started. PyYAML 6.0.3 is installed in `software/hermes-agent/.venv`.

### Gateway delegation: missing parent agent

The Hermes gateway's plugin dispatcher previously resolved parents only through
the legacy CLI reference. Loam round dispatch then failed with
`delegate_task requires a parent agent context`. Apply the reproducible
[Hermes patch](ports/jupiter/hermes-parent-agent.md) with
`scripts/apply_hermes_jupiter_patch.sh` before starting a campaign gateway. The
patch supplies the executing agent through a scoped ContextVar on both tool
execution paths; it does not assign a shared global parent or bypass native
delegation. Set `plugins.hook_callback_timeout: 600` in the campaign's
`config.yaml` as well, so the 480-second tournament hook can finish.

Hermes 0.20.6 requires single-child delegations to use `goal`/`context` mode;
its `tasks` batch mode requires at least two entries. Loam's Hermes adapter
converts a one-entry canonical task list for both round dispatch and delegation
middleware, preserving the child's full context and orchestrator role. This
compatibility change is in Loam; the pinned Hermes checkout stays unmodified.

The validator policy checks role/plan tags on a single task's `goal`, treating
`context` as supporting scientific material. Legacy `task`/`context` identity
fallbacks apply only when `goal` is absent. A tagged supporting context cannot
legitimize an untagged goal. This avoids rejecting valid single-task delegation
after converting the one-entry batch for Hermes 0.20.6.

## Compute-node internet through the login proxy

Jupiter compute nodes cannot reach public database endpoints directly. Loam's
optional authenticated HTTP/HTTPS forward proxy uses the same internal network
path as Redis: compute -> login `10.128.1.8:50858` -> public HTTP/HTTPS service.
The proxy runs as the user with one acceptor, not as a system service. HTTPS
uses CONNECT forwarding with normal client certificate verification; there is
no TLS interception. This is HTTP/HTTPS proxying, not transparent routing for
all protocols. Clients that disable environment proxies need explicit settings.

From the workspace containing `loam` and `software`:

```bash
software/RFD4-Proteina/.venv/bin/uv venv --python software/.python/cpython-3.12.11-linux-aarch64-gnu/bin/python3.12 software/internet-proxy/.venv
software/RFD4-Proteina/.venv/bin/uv pip install --python software/internet-proxy/.venv/bin/python -r loam/ports/jupiter/internet-proxy-requirements.txt
software/internet-proxy/.venv/bin/python loam/scripts/start_jupiter_proxy.py --bind 10.128.1.8
```

Use the current login node's compute-reachable internal address when changing
hosts. `software/internet-proxy/env.sh` privately exports lowercase and uppercase
HTTP/HTTPS proxy variables. Do not print it: the URLs contain authentication.
Credentials, client settings, PID and logs stay outside Git under that software
directory. Authentication is mandatory; the listener binds only the selected
internal interface. The client-to-proxy connection uses HTTP on the internal
network, so proxy authentication itself is not TLS-encrypted.

`set_env_jupiter.sh` sources the private file in Globus worker initialization;
`scripts/jupiter_env.sh` also sources it for scientific subprocesses. No secret
values are embedded in the submitted endpoint configuration. Start a fresh MCP
server after enabling it so worker environments inherit the settings.
`NO_PROXY` preserves direct loopback and cluster-internal access for MCP, Redis
and login-node Argo. This does not change Argo's laptop tunnel or its base URL.

Verify after the allocation starts by running
`skills/hpc-deployment/scripts/internet_proxy_smoke.py` through MCP `run_code`.
It compares direct versus proxied access, retrieves canonical UniProt P00533
through Python urllib and curl, checks their sequences match, and retrieves
RCSB entry 1IVO through requests. It reports the worker host/job and proxy host,
never proxy credentials. `campaign-test2/verify_internet_proxy.py` waits without
a queue deadline and records `state/internet-proxy-test.json`.

Keep the proxy alive for the allocation. To stop it, send SIGTERM to the PID in
`software/internet-proxy/state/proxy.pid`; this drains its child acceptor.
To disable it for future workers, remove the private env file and restart MCP;
unset proxy variables in any already-configured shell. Do not disable TLS
verification to fix certificate errors.

JSC recommends pre-staging internet files on login nodes where practical and
contacting the project mentor or support for sustained scraping workflows:
https://apps.fz-juelich.de/jsc/hps/jupiter/batchsystem.html#internet-access

Verified on 2026-11-06 from compute node `jpbo-003-01`, Slurm job 2728598:
direct urllib access failed with `URLError`, while the inherited proxy returned
HTTP 200 for UniProt P00533 JSON via urllib and RCSB 1IVO JSON via requests.
Curl independently retrieved the same 1210-residue P00533 FASTA sequence.
Sequence SHA256: `dbb4583267ad40a7dfd854ee3c44ce4be38e137d0570e3eb76cf1854a74e7246`.
Evidence: `campaign-test2/state/internet-proxy-test.json` in the workspace.
The authenticated-forwarding integration test passed; Loam suite: 488 passed,
3 skipped. This verifies database networking, not a completed design campaign.

### Missing-path inspections must not trip the MCP connection breaker

The HPC MCP dispatcher distinguishes expected inspection failures from worker
unavailability. Missing paths return `status: not_found` and `exists: false`;
permission failures return `denied`; invalid operations/paths and binary reads
return `invalid_request`. These are explicit application failures in normal MCP
responses. Successful inspection formats are unchanged. The worker's read-only
path/security checks remain unchanged, and inspection never creates directories.

Pool lookup failures and unexpected worker failures still raise MCP ToolError.
This avoids Hermes 0.20.6 treating three legitimate filesystem errors as a
connection outage, without patching Hermes or weakening its trust checks.
A regression uses an isolated loopback fixture and the stock Hermes MCP client:
three missing-path results followed by run_code succeed; genuine pool failures
still open the breaker. The fixture has no arbitrary code execution or private
file-reading capability; its trust setting is confined to a temporary test home.
See `tests/test_mcp_dispatch.py` and `tests/test_mcp_dispatch_hermes.py`.
Start a fresh MCP server to load the dispatcher change. This fix does not
restart a scientific campaign or turn a previously blocked attempt into a result.

Validation for the dispatcher fix: 495 Loam tests passed, 3 skipped, including
the unmodified Hermes 0.20.6 client regression. Lint and whitespace checks passed.
The completed campaign-test3 pool was stopped after testing; the fix loads on
the next MCP start. No Hermes source changes or production trust changes were
required.

### Binder generator and command timeout

Use `rfd4-proteina-jupiter` and `software/RFD4-Proteina/.venv` for backbone generation. RFdiffusion3 recipes are archived outside the discoverable skills tree in `ports/jupiter/disabled-skills/`; a CLI/import check does not establish an available inference installation. Use 900 seconds (15 minutes) for setup/scientific subprocesses and `run_code(timeout=900)`, with an MCP client request timeout of at least 900 seconds. Read RFD4’s named checkout skills directly instead of recursively scanning environments. This does not change the unbounded scheduler queue wait.

### Hermes tool execution deadlines

Hermes 0.20.6 reads `timeouts.tools.sequential_call` (not `sequential`) and
`timeouts.tools.concurrent_batch`. Set both to 1800 seconds in the campaign
`config.yaml` and the shared Hermes profile. These outer tool deadlines cover
MCP calls and delegation; increasing only subprocess or MCP timeouts leaves
the default 420-second tool executor deadline in place. Keep scientific
subprocess/run_code limits at 900 seconds and MCP request timeout at 960 seconds.
New Hermes sessions are required to load configuration changes. These settings
do not extend Slurm walltime or the campaign's overall time budget.

### RFD4 native-runtime validation result

Manual MCP validation on `jpbo-016-46`, Slurm job `2799516`, passed Open Babel/AtomWorks import,
`rfd4-generate --help`, and one real `monomer-short` generation using the pinned
model bundle, batch size 1, seed 20261109, and `compile_lengths=null`.
The generated CIF parses and contains finite coordinates. See
`ports/jupiter/rfd4-runtime-validation.json` for source/input/output hashes and
the retained artifact path. This is an unconditional protein installation test,
not evidence of an EGFR binder or affinity. These checks were not added to
campaign preflight.

### Remaining campaign-test driver defaults

Use `--max-rounds 20 --max-hours 12 --max-turns 100 --hpc-wait-seconds 6000`
for remaining Jupiter tests. These are also the branch's `drive_campaign.py`
defaults and are recorded in `ports/jupiter/campaign-test-defaults.json`.
They are maximum limits, not guarantees: existing barren-round, stuck-turn,
and pool-failure stop guards still apply. Worker allocations remain one node
for 30 minutes at a time, with replacement permitted within the driver budget.
The initial scheduler startup wait remains unbounded.

### Extended generation limits (remade campaign-test8 onward)

Per-design scientific subprocess and remote `run_code` limits are 6000 seconds;
MCP requests use 6300 seconds and Hermes `sequential_call`/`concurrent_batch` use
14400 seconds. Use one design per task, with immediate log-file output.
Worker allocations for these tests are one node for two hours, within the
unchanged 12-hour/20-round/100-prompt driver limits and 6000-second HPC recovery wait.
Activate the RFD4 venv explicitly before upstream shell wrappers, with
`CLUSTER=local`, to prevent inheritance of the Globus worker's Python environment.
No RFD4 installation checks were added to campaign preflight.

### RFD4 data-loader default

Use `num_workers=0` for the current single-design Jupiter tests, alongside the
agreed `recycle_strategy=medium/front_loaded`. This loads data in the main
process rather than spawning four DataLoader subprocesses. The successful
manual generation used zero workers. Campaign-test10 stopped showing progress
at the loader stage; zero workers is a controlled retry, not yet proof of its
root cause. `install_jupiter_rfd4_runtime.sh` now invokes
`configure_jupiter_rfd4.py` to set only the installed inference configuration's
`num_workers` default to zero, preserving all other fields. To apply separately:

```bash
python3 loam/scripts/configure_jupiter_rfd4.py
```

Always include `num_workers=0` in campaign commands for traceability. These
changes do not add any checks to campaign preflight.

### ProteinMPNN mmCIF conversion dependency

After installing ProteinMPNN, run `bash loam/scripts/install_jupiter_proteinmpnn_extras.sh`
from the workspace. This adds pinned `gemmi==0.7.5` to
`software/proteinmpnn/.venv`, enabling conversion of RFD4/Chai mmCIF output
to the PDB input consumed by ProteinMPNN. Other tools' isolated environments
do not supply this dependency. The installer preserves the existing model
stack; record and verify target/binder chain identity after conversion.
No conversion check was added to campaign preflight.

### Manual software workflow audit

Import checks in `ports/jupiter/status.json` are installation inventory, not
scientific execution evidence. Run the separate workflow suite from the workspace
with a healthy MCP pool. It uses each tool's own venv, small real model/MD/design
workloads, and actual output checks. It runs sequentially to leave campaign
capacity available. Use a fresh output directory and sufficient allocation time:

```bash
software/globus-compute/.venv/bin/python loam/scripts/smoke_jupiter_skills.py \
  --output "$PWD/software/verification/<fresh-audit-directory>"
# Or select cases, for example after repairing both MD environments:
bash loam/scripts/install_jupiter_openmm_cuda.sh
software/globus-compute/.venv/bin/python loam/scripts/smoke_jupiter_skills.py \
  --only openmm molecular-simulations \
  --output "$PWD/software/verification/<fresh-md-audit-directory>"
```

These are manual installation tests, not campaign preflight requirements. They
do not start or stop scheduler allocations, restart services, or write campaign
evidence. Each skill exposes its test instructions through
`skill_view(name="<tool>-jupiter", file_path="references/smoke-test.md")`.
Literature/RAG tests include public database retrieval, real pretrained embeddings,
persisted/reloaded FAISS retrieval, and a login-node Argo completion. Their tiny
retrieval passages are labeled synthetic. Model tests use trained checkpoints;
missing/gated assets and missing licensed software cannot count as passes.

OpenMM 8.6.1 requires its separate CUDA extra in **both** MD venvs. The repair
installer uses `ports/jupiter/openmm-cuda-requirements.txt`. Boltz currently needs
`--no_kernels --num_workers 0` because the optional cuEquivariance package is
absent. See the corresponding Jupiter workflow references for the exact APIs.

Reactra is excluded from the required software audit at the user's request; the default suite tests the other 14 Jupiter software skills. Its references remain packaged for reading.

The current [workflow report](ports/jupiter/software-smoke-report.md) and
[machine-readable results](ports/jupiter/software-smoke-status.json) preserve
all audit attempts and distinguish successful workflows from authorization or
licensed-software blockers. Refresh a report with
`scripts/summarize_jupiter_smoke.py --runs <audit-dir> [<repair-dir> ...]`.
This also updates each skill's own `references/smoke-test.md` with its status.

### Develop integration: batched science and tournament options

See [the integration inventory](ports/jupiter/develop-integration.md) for the four incoming commits, compatibility decisions and remaining operational requirements. The Jupiter runtime now uses Jnana commit `4c4cdfb42bf042a462eaf091dfbdf24a2ce34598`, which implements `batch_size` generation and `scheme`, `rounds`, `workers` tournament options. Reproduce the source update with `bash scripts/update_jupiter_jnana.sh` after the base Jnana environment is installed. Its dependencies are unchanged; Hermes remains stock 0.20.6. This is not a campaign preflight check.

Optional profile settings under `hypothesis_tournament`:

```yaml
scheme: swiss
rounds: 2
workers: 2
batch_size: 2
```

Omitted fields retain `random`, 1, 1, 1 respectively. Parallel match workers require Swiss pairings; random tournaments remain single-worker. These options do not change the selected Argo endpoint/model.

Fresh delegated campaigns stage scripts/JSONL with `write_files` and execute every scientific job through `run_batch`, including one-row jobs. Review inherited `execution.setup` text before reusing an old campaign; instructions to run science inside `run_code` conflict with the new policy. Keep Jupiter-specific skill names and their direct CUDA/native RFD4 recipe, `num_workers=0`, 6000-second row timeouts and the existing driver/allocation limits. Manual operator installation smoke tests remain separate.

Existing MCP servers and workers retain their loaded Python classes and tool schemas. Before a new campaign using `write_files`, restart/recreate the MCP server and pool together so the writer action and latest batch behavior are available on both ends. This repository integration did not restart live services or cancel allocations. Additional software inspection roots remain read-only for `write_files`; only the campaign tree is writable through that staging tool.

Allocation recovery inherits the launch's `HPC_ENV_FILE` (export the campaign's `hpc_env.sh` path before launch); Jupiter defaults preserve an existing value. The controller permits the agreed two-hour test walltime while retaining the one-node cap. Source the Jupiter environment before invoking Hermes so its `${HPC_ENV_FILE}` reference resolves.
