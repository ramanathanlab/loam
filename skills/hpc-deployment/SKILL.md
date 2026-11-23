---
name: hpc-deployment
description: >-
  Deploy a local MCP server backed by a user-supplied Globus
  Compute endpoint and an Ensemble Launcher cluster on the System's batch
  allocation (Aurora, Polaris, Jupiter).
license: Apache-2.0
systems: [aurora, polaris, jupiter]
---

# Globus Compute → batch allocation → Academy-agent MCP deployment

Read `references/$HPC_SYSTEM.md` before deployment for the selected System's
endpoint, scheduler checks, device selection and launcher. Supported on Aurora,
Polaris and Jupiter; unsupported on other Systems. Site-specific paths, accounts
and environments come from the campaign's Site profile.

## Sole responsibility

This skill does exactly this, and nothing else:

1. Start a **local FastMCP server**. Cluster mode exposes nine tools: `run_code`, `write_files`, `inspect_filesystem`, `register_tool`, `list_tools`, `run_batch`, `batch_status`, `list_batches`, and `pool_status`. Direct mode omits `run_batch`.
2. The server uses a **Globus Compute `Executor`** targeting the **endpoint UUID supplied by the user**.
3. The submitted endpoint job uses the System's **Parsl provider** to request the **user-supplied allocation** (`account`, `queue`, `walltime`, `nodes_per_block`, block count, filesystem/place settings, and worker environment).
4. Within that allocation, start Academy agents in one of two modes. **Direct mode** starts one `HpcAgent` per pinned GPU/tile, and the agent runs payloads itself. **Cluster mode** (`create_server(cluster=True)`, used by the executable entry point) starts one `HpcClusterAgent` that runs an Ensemble Launcher cluster, plus exactly `HPC_POOL_SIZE` unpinned `HpcClientAgent`s that submit into it. Server startup fails and cleans up if the complete requested client pool is unavailable.

The harness never uses SSH, `qsub`, `qstat`, `sbatch`, or a campaign-local scheduler wrapper to execute campaign work. The endpoint and its Parsl provider own scheduling. The agent has no domain-specific actions and no arbitrary shell action.

## Campaign inputs

The user-authored loam YAML is the sole campaign settings file. Select
`execution.mode: hpc` and place deployment settings under `execution.hpc`:

| YAML input | Server environment generated internally by loam |
|---|---|
| `launch_mode` | `HPC_LAUNCH_MODE`: `globus` (default) or `existing` |
| `endpoint_id` | `HPC_GLOBUS_ENDPOINT_ID` (Globus submission only) |
| `pool_size` | `HPC_POOL_SIZE` |
| `base_dir` | `HPC_BASE_DIR` |
| `worker_code_dir` | `HPC_LOAM_HOME` |
| `campaign_id` | `HPC_CAMPAIGN_ID` |
| `exchange_url` | `ACADEMY_EXCHANGE_URL` |
| `system` | `HPC_SYSTEM`, plus the System spelled out: `HPC_CPU_IDS`, `HPC_GPU_IDS`, `HPC_GPU_PIN_VAR`, `HPC_NODE_DISCOVERY`, `HPC_MPI_FLAVOR` (with `gpus_per_node` / `cpus_per_node` overrides, where the System allows them) |
| `science_workers_per_gpu` | `HPC_SCIENCE_WORKERS_PER_GPU` |
| `allocation.nodes` | `HPC_NNODES` |
| `inspect_roots` | `HPC_INSPECT_ROOTS` |
| `cold_endpoint_timeout` | `HPC_COLD_ENDPOINT_TIMEOUT` |
| `cold_endpoint_backoff` | `HPC_COLD_ENDPOINT_BACKOFF` |
| Site profile (`site_profile`) | `LOAM_SOFTWARE_ROOT`, `LOAM_DATA_ROOT`, `LOAM_CACHE_ROOT`, `LOAM_MODEL_WEIGHTS_JSON`, `LOAM_INFERENCE_JSON`, and `HPC_WORKER_ENV_JSON`, which the agents set for every script they run on a worker (endpoint templates may ignore a client `worker_init`) |

The YAML also supplies `server_script`, local `host` and `port`,
`startup_timeout`, and the allocation's `account`, `queue`, `walltime`, block
limits, idle timeout, and endpoint-specific options. The account and queue may
instead come from the Site profile the YAML names; a value in the YAML wins. Loam renders the
allocation as `HPC_ENDPOINT_CONFIG_JSON` and starts the server with an internal
environment. Users must not create, edit, or source a second campaign
environment file. The server may continue consuming these generated variables.

Never guess, hard-code, or inherit these inputs from an earlier campaign. Use
placeholders in examples, not endpoint UUIDs, accounts, queues, or shared
filesystem paths from a live deployment. Model API keys and proxy tokens remain
external and are referenced by environment-variable name in model
`api_key_env`. Globus and Academy credentials remain in their external login or
credential stores; neither secret nor token belongs in YAML.

Cluster-mode health settings that are not user-facing campaign choices retain
their server defaults. In cluster mode, the endpoint needs one worker slot per
client agent plus one for the long-running `HpcClusterAgent`; derive `workers`
from `pool_size + 1`. Worker initialization must activate the worker environment,
make `base_dir` importable, and export the generated `HPC_LOAM_HOME` so
worker-visible skill paths resolve.

The allocation mapping is endpoint-specific. Before deployment, the operator
must inspect the live endpoint user-config schema and Jinja template, verify
that every requested field is accepted and rendered, and check current queue
limits. A permissive schema alone does not prove that the template supports the
request. No endpoint or queue has been verified merely because it appears in a
configuration example.

Each System is defined once in `loam.systems`: its cores, accelerators, the
variable that pins a process to one accelerator, how nodes are found, the launcher,
and how the endpoint config sizes its workers. Loam spells these out in the generated environment, and neither
the server nor the cluster agent branches on `HPC_SYSTEM`; it is only a label.
Adding a System means adding a definition there, not editing these scripts.

In direct mode, request one worker per accelerator in `HPC_GPU_IDS` and pin each
with a distinct value of `HPC_GPU_PIN_VAR`; `pool_size` is requested nodes ×
accelerators per node unless the endpoint exposes another count. System-specific
pinning details live in `references/$HPC_SYSTEM.md`. In
cluster mode EL owns accelerator placement and `pool_size` is the configured
client-agent count, not the accelerator count.

## Architecture

```text
MCP harness
  → local FastMCP server (nine public tools in cluster mode)
  → Globus Compute Executor(endpoint_id=HPC_GLOBUS_ENDPOINT_ID,
                            user_endpoint_config=HPC_ENDPOINT_CONFIG_JSON)
  → endpoint manager / user endpoint
  → the System's Parsl provider allocation
  → direct mode:  one worker per pinned GPU/tile
                  → one Academy HpcAgent per worker
  → cluster mode: HpcClusterAgent → EL cluster (owns GPU placement)
                  → exactly HPC_POOL_SIZE HpcClientAgents submit into it
  → run_code for cheap single calls; run_batch for campaign work

```
### Attaching to an existing allocation

With `launch_mode: existing` the server runs inside an allocation that is already running (the operator starts it from within the job) and nothing is submitted to Globus: Academy hosts the agents in the server's own event loop, and EL places science across the allocation's nodes. The server finds those nodes when it starts, the way `HPC_NODE_DISCOVERY` says (`PBS_NODEFILE`, or `SLURM_JOB_NODELIST`), and fails at once if neither is set, rather than falling back to the one host it is on. Their count must equal `allocation.nodes`. When the allocation is split into roles, `HPC_SCIENCE_NODELIST` (set by whoever splits it, in the scheduler's host notation) names the science share; the rest serve inference. For whoever starts services inside the allocation, two loam modules check it first. `loam.allocation_startup` waits for fresh telemetry from every node showing all of that System's GPUs. This works only on Systems whose GPU telemetry loam can read. `loam.allocation` checks that each process runs in its assigned job, role and host. The server itself calls neither of them.

The endpoint UUID routes the request; it does not select the queue. The endpoint's accepted user configuration is rendered by its Parsl provider into the scheduler request. In multi-user deployments, record both the manager UUID and the spawned user-endpoint UUID: they are different by design.

## Endpoint discovery and flex-scaling validation

When the user asks which endpoint to use or whether a single endpoint supports varying queues, node counts, and walltimes, SSH to the endpoint host and inspect `~/.globus_compute/` conservatively. Identify candidates from endpoint directories, read their `endpoint.json`, `config.yaml`, `user_config_schema.json`, and `user_config_template.yaml.j2`, then check recent endpoint logs/process state. Do not expose credential files, certificates, or token data.

A permissive schema does **not** establish flex scaling on its own. Confirm that the per-submission Jinja template actually renders `user_endpoint_config` into the provider's queue, walltime, node count, block limits, account and scheduler options. A deliberately minimal manager `config.yaml` with its engine defined in that template is an expected flex-scaling architecture, not a defect. Follow the selected System reference to check live queue constraints.

Record the verified manager UUID in campaign YAML as `execution.hpc.endpoint_id`; never hard-code a user endpoint UUID or site-specific resource defaults into this skill. A directory on disk does not prove an endpoint is healthy—prefer current logs/process state over historical registrations.

## Deployment and verification

1. On the endpoint host, inspect the live endpoint user-config schema/template and queue limits. Confirm that the requested allocation keys are accepted and rendered, the intended environment is activated, `HPC_BASE_DIR` is importable, and accelerator placement is correct. Two checks that are cheap here and expensive later:
   - Check maintenance reservations and queue availability using the selected System reference; a healthy endpoint alone does not establish that a block can start.
   - `HPC_BASE_DIR` must already EXIST on the shared filesystem. `HpcClusterAgent.__init__` passes `ckpt_dir` and `log_dir` under it straight to EL with no `mkdir`, and nothing else creates them.
2. Create/reuse a local uv environment containing `fastmcp`, `globus-compute-sdk`, `academy-py`, and `ensemble-launcher`. The last is needed even though the launcher only ever runs on a worker: `HpcClusterAgent.__init__` builds EL's `SystemConfig` and `LauncherConfig` from the generated environment, and Academy constructs that agent LOCALLY before shipping it, so without EL the server dies with a `NameError` before submitting anything. The agent finds its nodes later, on the worker, the way `HPC_NODE_DISCOVERY` says (`PBS_NODEFILE` on PBS Systems). The base install suffices; `mpi4py` lives in EL's `hpc` extra.
3. Put the reviewed managed-HPC inputs in the campaign YAML and run `python -m loam CONFIG.yaml`. Loam generates the server's internal environment and starts `server_script`; do not source a campaign environment file or launch a separately configured server.
4. The server validates its generated environment and imports its dependencies before submitting anything, so bad configuration fails without costing a job. Its Academy launch manager uses the Globus Compute `Executor` created with the supplied UUID and endpoint configuration. A local `ThreadPoolExecutor` is invalid: it launches agents locally.
   A multi-user endpoint forks the per-config user endpoint on demand, and the first submit against a cold one is answered `422 ENDPOINT_NOT_ONLINE`, which the Globus SDK does not retry. `ClusterPool` therefore retries just that code, bounded by the YAML's `cold_endpoint_timeout` (default 300s) starting at `cold_endpoint_backoff` (default 5s); every other error still aborts on the first attempt. Set these in the campaign YAML like every other input — loam renders them into the generated environment, and no `HPC_*` variable is ever a user-facing knob. Do not widen this into a general retry — it would hide a misconfiguration behind minutes of silence. Note the symptom when it is NOT handled: the launch dies, but the user endpoint keeps starting and queues a block, stranding an allocation with no client attached.
5. Launch the agents. Direct mode launches exactly `HPC_POOL_SIZE` `HpcAgent`s, each reporting hostname, `HPC_CAMPAIGN_ID`, and a unique device pin. Cluster mode launches the `HpcClusterAgent` first, then `HPC_POOL_SIZE` `HpcClientAgent`s against that cluster's checkpoint dir; clients report an empty device pin by design. All handles are rebound to their exchange client before serving MCP calls — the pool probes them for the life of the server.
6. Set `HPC_MCP_URL` to the running server's URL and `HPC_BASE_DIR` to the campaign's configured base directory, then run `scripts/smoke_test_client.py`. This operator check requires the nine-tool cluster-mode contract, makes one cheap `run_code` call, and checks bounded filesystem and `pool_status` metadata without submitting science.
7. Submit concurrent cheap calls and confirm they **overlap** rather than running one at a time. In direct mode also verify unique `(hostname, device pin)` pairs; if pins duplicate or an agent is absent, stop and repair placement. Pins are empty and identical in cluster mode, so they are not a valid acceptance signal there — EL owns placement.
8. Agents do not outlive the server. `Manager.close()` shuts down everything it launched, so a restarted server builds a fresh pool rather than adopting one. Within a run, cluster mode repairs itself (below).

The skill supplies `scripts/hpc_agent.py`, `scripts/pool_manager.py`, `scripts/mcp_server.py`, and `scripts/smoke_test_client.py`. A deployment is not considered ready until steps 6–7 succeed against the live endpoint.

### Shutting the server down

Stop the server with `SIGINT` (`Ctrl-C`, or `kill -INT`). A clean exit is what
runs `Manager.close()`, which asks every launched agent to shut down; for
`HpcClusterAgent` that is what calls `el.stop()` and releases the cluster.
`mcp_server.py` installs `signal.default_int_handler` for `SIGTERM` so that
signal unwinds the same way — without it uvicorn re-raises `SIGTERM` after
`serve()` returns, the default handler kills the process before the FastMCP
lifespan resumes past its `yield`, and the EL cluster is orphaned until
walltime. Never stop the server with `kill -9`: no agent is told to shut down
and the allocation leaks.

In cluster mode the lifespan calls `pool.stop()` before `Manager.close()`. That
ordering is load-bearing: the health-check task must stop before the manager
starts shutting agents down, or it sees them die during teardown and rebuilds
the cluster on the way out.

## Self-healing (cluster mode)

`ClusterPool` owns the cluster agent and the client agents, and re-probes them
every `HPC_POOL_PROBE_INTERVAL`. Repair runs on that timer, never inside a tool
call — reconciling per call is what serializes concurrent `run_code` requests.

- **A client agent stops answering** → after `HPC_CLIENT_PROBE_STRIKES`
  consecutive failed probes it is dropped from rotation, shut down, and replaced
  on its own. The cluster is untouched. Until then it stays in rotation: a probe
  can fail with the agent healthy (a message lost on the exchange, a 5xx burst),
  and relaunching on every such blip piled load onto an exchange already in
  trouble. Any successful probe resets the count.
- **The cluster agent dies** → the EL cluster died with it, so every client is
  holding a `ClusterClient` bound to nothing. After `HPC_CLUSTER_PROBE_STRIKES`
  consecutive failed probes the pool shuts the stale agents down (freeing their
  worker slots), mints a **fresh `ckpt_dir`**, and rebuilds the cluster and all
  clients. A new generation never reuses the dead master's checkpoint record.
- **During a rebuild** a tool call waits rather than failing, bounded by the MCP
  client's own timeout unless `HPC_POOL_WAIT_TIMEOUT` is set.
- **After `HPC_CLUSTER_MAX_RESTARTS` consecutive failed rebuilds** healing stops
  for good. The MCP server stays up and every tool call reports the terminal
  error; restart the server to try again.

The MCP server, its client sessions, and the Globus/Academy plumbing survive all
of this — a failed cluster is no longer a restart-everything event.

Known gap: `HpcClusterAgent.probe` reports only that the *agent* is alive. An EL
master that dies under a live agent reads as healthy, and tasks hang until their
own timeout instead of triggering a rebuild.

## Timeouts and failures

Distinguish:

- **Submit rejection:** no endpoint task was accepted—inspect UUID, auth, and endpoint configuration.
- **Scheduler wait:** the batch job (block) is queued—do not blindly resubmit. Scheduler-wait allowance is independent of payload timeout.
- **Payload failure:** an agent returned a normal result with nonzero `returncode`—use stderr to fix/retry only that payload.
- **Cluster loss (cluster mode):** the cluster agent stopped answering—the pool rebuilds it automatically; see "Self-healing". Do not restart the server unless healing has reported itself unrecoverable.

Never alter queue or resource values outside the endpoint user-config schema. Never call scheduler commands from the harness.

## Agent capabilities and filesystem inspection

The cluster-mode MCP contract has nine public tools:

- `run_code(payload, workdir, timeout, venv_path)` — execute one Python payload; use it only for cheap probes, result harvesting, or failed-row diagnostics.
- `write_files(files)` — stage bounded text files strictly under `HPC_BASE_DIR`; `HPC_INSPECT_ROOTS` and `/tmp` remain read-only.
- `inspect_filesystem(operation, path, offset, limit)` — bounded read-only visibility without SSH.
- `register_tool`, `list_tools`, `run_batch(script_path | tool, inputs_path, timeout_each, gpus, venv_path, batch_id, labels)`, `batch_status`, and `list_batches` — the run-batch-first path for campaign work. Stage one script plus JSONL inputs, pass the environment required by the relevant skill as `venv_path`, and run the script once per row through EL. `venv_path` is the venv **ROOT**, not an interpreter: `_python()` appends `bin/python` to it, so a path ending in `/bin/python` becomes `.../bin/python/bin/python` and every row fails to launch. A skill that exports a `*_PYTHON` interpreter variable is giving you the value for direct shell use — strip `/bin/python` before passing it here. Declare the output contract with `register_tool(..., expects=[<glob>, ...])` — globs relative to each run's folder, naming the artifacts the script MUST write — then submit batches with `tool=<name>` instead of `script_path`. `expects` is the ONLY signal separating a row that did the work from one that exited 0 having written nothing: a row that exits clean without its outputs is counted as failed (L3), and a batch carrying no contract reports `ok: false` with `ok_withheld` and cannot be harvested. Register once, reuse across rounds; `list_tools` shows what exists. Globs are checked in the row's attempt directory (the script's cwd) before a successful attempt is promoted to the row top level, so match the path the script writes, not the promoted one. A submission receipt is not execution: poll the owned batch until `batch_status` reports both `complete: true` and `ok: true`. `complete` alone only means no rows remain unfinished; it may include failures.
- `pool_status()` — configured allocation capacity (`nodes`, tiles, queue, walltime) for sizing batch fan-out. It reports capacity, not current free/busy state; use batch status tools for in-flight work.

`gpus` is tiles per row, not per batch: use `gpus=1` for a script that touches
one device and raise it only when one row spans multiple devices. The campaign
skills specify the launcher, environment, and weights paths for each workload.

`inspect_filesystem` permits only:

| Operation | Semantics | Required bounds |
|---|---|---|
| `ls` | Sorted immediate directory entries with type and size metadata; no recursive traversal. | At most 200 entries. |
| `cat` | UTF-8 text read, with line numbers. | `offset ≥ 1`, at most 200 lines and 64 KiB. |
| `tail` | UTF-8 text read from end, with line numbers. | At most 200 lines and 64 KiB. |

The action resolves paths and restricts them to the explicit allowed roots: `HPC_BASE_DIR`, `HPC_INSPECT_ROOTS` (colon-separated optional additional roots), and `/tmp`. It rejects symlink escapes, binary data, directories passed to `cat`/`tail`, device files, and every write/mutation operation. Do not add a generic `shell`, `exec`, `find`, `grep`, environment dump, scheduler query, or recursive listing action. For programmatic analysis beyond bounded inspection, write it as a `run_code` payload.

## Size batches from observed allocation state

Call `pool_status` instead of assuming a node count. A `run_batch(gpus=1)` row
uses one accelerator (one tile on Aurora, one GPU on Polaris or Jupiter) and
`pool_status.concurrent_scientific_rows` is the configured one-wave capacity:
`total_tiles`, times `science_workers_per_gpu` when several science workers share
each GPU (Jupiter only). Under sharing, rows take `gpus=0` or `gpus=1`. This is not a live-free-tile count: inspect
`list_batches` and `batch_status` for work already in flight.

Code-generators share the pool across concurrently validated hypotheses. Size
a wave conservatively from the reported capacity and concurrency, poll its
owned batch until `complete: true` and `ok: true`, then submit another wave if
needed. A terminal batch with `ok: false` is observed failure, not successful
execution.

## Non-negotiable constraints

- Cluster mode exposes the nine tools listed above; the internal health probe is never an MCP tool.
- `write_files` writes only below `HPC_BASE_DIR`; inspection-only roots never become writable.
- Process timeouts terminate the payload's entire process group.
- Preserve endpoint UUID, endpoint config, campaign ID, agent IDs, pins, seeds, commands and outputs.
