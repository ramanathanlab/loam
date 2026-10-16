---
name: hpc-deployment
description: >-
  Deploy a local one-tool MCP code-execution server backed by a user-supplied
  Globus Compute endpoint. The endpoint's Parsl PBSProProvider provisions
  user-specified queue, walltime, node and accelerator resources; one Academy
  code-execution agent is pinned to each allocated GPU/tile.
license: Apache-2.0
---

# Globus Compute → PBSPro → pinned Academy-agent MCP deployment

## Sole responsibility

This skill does exactly this, and nothing else:

1. Start a **local FastMCP server** with two public tools: `run_code` and bounded, read-only `inspect_filesystem`. The second tool exists solely to provide `ls`/`cat`/`tail`-equivalent worker visibility without SSH.
2. The server uses a **Globus Compute `Executor`** targeting the **endpoint UUID supplied by the user**.
3. The submitted endpoint job uses its **Parsl `PBSProProvider`** to request the **user-supplied allocation** (`account`, `queue`, `walltime`, `nodes_per_block`, block count, filesystem/place settings, and worker environment).
4. Within that allocation, start **one Academy `HpcAgent` per pinned GPU/tile**. Every agent exposes `run_code` plus one constrained, read-only `inspect_filesystem` action. `run_code` executes submitted Python in a subprocess and returns `{stdout, stderr, returncode}`; `inspect_filesystem` supports `ls`, `cat`, and `tail` without giving the harness SSH.

The harness never uses SSH, `qsub`, `qstat`, `sbatch`, or a campaign-local scheduler wrapper to execute campaign work. The endpoint and its Parsl provider own scheduling. The agent has no domain-specific actions and no arbitrary shell action.

## Required inputs

Never guess, hard-code, or inherit these from an earlier campaign:

| Input | Meaning |
|---|---|
| `HPC_GLOBUS_ENDPOINT_ID` | Globus Compute endpoint/manager UUID addressed by the local server. |
| `HPC_POOL_SIZE` | Number of agents to launch; equal to requested addressable GPU/tile count. |
| `HPC_ENDPOINT_CONFIG_JSON` | JSON object accepted by the endpoint user-config schema; defines PBSPro resources. |
| `HPC_BASE_DIR` | Campaign directory visible to all workers. |
| `HPC_CAMPAIGN_ID` | Immutable ID used to isolate agent discovery/adoption. |
| `ACADEMY_EXCHANGE_URL` | Academy exchange the manager and agents rendezvous through. |

Optional: `HPC_INSPECT_ROOTS` (colon-separated extra `inspect_filesystem` roots)
and `HPC_MCP_TRANSPORT` (`http` or `stdio`, default `stdio`).

### Setting them: `set_env.sh`

Define all of them in `set_env.sh` and `source` it into the shell that runs
every later step:

```bash
source skills/hpc-deployment/set_env.sh
```

`mcp_server.py` reads the process environment and fails fast via `required()` on
anything missing, so make this file the single definition point rather than
exporting values by hand. A variable set in one shell and forgotten in the next
is the usual cause of a campaign that starts against the wrong endpoint or
adopts another campaign's agents.

`set_env.sh` is campaign-specific — edit it per deployment and keep the endpoint
UUID, queue, walltime, and `HPC_CAMPAIGN_ID` together in it so the whole
configuration is reviewable in one place.

Two couplings belong in the file itself, not in the operator's head:

- **Worker count.** The endpoint needs one worker slot per client agent *plus*
  one for the long-running `HpcClusterAgent`, which holds its slot for the life
  of the allocation. Derive it rather than typing a literal:
  `"workers": '"$((HPC_POOL_SIZE + 1))"'`.
- **Worker environment.** `worker_init` must activate the environment and put
  `HPC_BASE_DIR` on `PYTHONPATH`, or the agent modules will not import on the
  worker.

Example only—pass keys only when the supplied endpoint's user-config schema permits them:

```json
{
  "account": "PROJECT",
  "queue": "debug-scaling",
  "walltime": "01:00:00",
  "nodes_per_block": 1,
  "init_blocks": 1,
  "min_blocks": 1,
  "max_blocks": 1,
  "available_accelerators": 12,
  "max_workers_per_node": 12,
  "filesystems": "flare",
  "place": "scatter"
}
```

For Aurora, request 12 addressable tiles per node. Endpoint worker initialization must set `ZE_FLAT_DEVICE_HIERARCHY=FLAT` and a distinct `ZE_AFFINITY_MASK` for every pinned worker. `HPC_POOL_SIZE` is `requested nodes × 12` unless the endpoint explicitly exposes another count.

## Architecture

```text
MCP harness
  → local FastMCP server (one public tool: run_code)
  → Globus Compute Executor(endpoint_id=HPC_GLOBUS_ENDPOINT_ID,
                            user_endpoint_config=HPC_ENDPOINT_CONFIG_JSON)
  → endpoint manager / user endpoint
  → Parsl PBSProProvider allocation
  → one worker per pinned GPU/tile
  → one Academy HpcAgent per worker
  → run_code(payload)
```

The endpoint UUID routes the request; it does not select the queue. The endpoint's accepted user configuration is rendered by its Parsl `PBSProProvider` into the PBS request. In multi-user deployments, record both the manager UUID and the spawned user-endpoint UUID: they are different by design.

## Endpoint discovery and flex-scaling validation

When the user asks which Polaris endpoint to use or whether a single endpoint supports varying queues, node counts, and walltimes, SSH to the endpoint host and inspect `~/.globus_compute/` conservatively. Identify candidates from endpoint directories, read their `endpoint.json`, `config.yaml`, `user_config_schema.json`, and `user_config_template.yaml.j2`, then check recent endpoint logs/process state. Do not expose credential files, certificates, or token data.

A permissive schema does **not** establish flex scaling on its own. Confirm that the per-submission Jinja template actually renders `user_endpoint_config` into the Parsl PBSProProvider fields for `queue`, `walltime`, `nodes_per_block`, blocks, account, and PBS filesystem/placement options. A deliberately minimal manager `config.yaml` with its engine defined in that template is an expected flex-scaling architecture, not a defect. Validate live queue constraints with `qstat -Q` and, when selecting a queue, `qstat -Qf <queue>`; route queues can impose node-count minima that make an otherwise valid request fail.

Record the verified manager UUID in campaign configuration as `HPC_GLOBUS_ENDPOINT_ID`; never hard-code a user endpoint UUID or site-specific resource defaults into this skill. A directory on disk does not prove an endpoint is healthy—prefer current logs/process state over historical registrations.

See `references/polaris-globus-compute-flex-scaling.md` for the concise Polaris inspection procedure and session-specific observations.

## Deployment and verification

1. On the endpoint host, inspect the live endpoint user-config schema/template. Confirm that it accepts the requested allocation keys, activates the intended environment, makes `HPC_BASE_DIR` importable, and sets accelerator pinning.
2. Create/reuse a local uv environment containing `fastmcp`, `globus-compute-sdk`, and `academy-py`.
3. Edit `set_env.sh` for this campaign and `source` it. Run every later step in that same shell — the server inherits its configuration from the process environment and nowhere else.
4. Run `scripts/preflight.py`. It validates local dependencies/configuration and emits the exact worker-side import/pinning prerequisites without submitting a job.
5. Start `scripts/mcp_server.py`. Its Academy launch manager uses the Globus Compute `Executor` created with the supplied UUID and endpoint configuration. A local `ThreadPoolExecutor` is invalid: it launches agents locally.
6. Launch exactly `HPC_POOL_SIZE` `HpcAgent`s. Each bootstrap reports hostname, `HPC_CAMPAIGN_ID`, and device pin; returned handles are rebound to their exchange client before serving MCP calls.
7. Run `scripts/smoke_test_client.py` against the server. It requires both `run_code` and `inspect_filesystem`, a normal code result, and bounded worker filesystem output.
8. Submit `HPC_POOL_SIZE` concurrent cheap calls and verify unique `(hostname, device pin)` pairs. If pins duplicate or an agent is absent, stop and repair placement.
9. On server restart, discover and adopt only agents returning the exact campaign ID; revalidate, prune, and top up only the missing agents through the same Executor/configuration.

The skill supplies `scripts/hpc_agent.py`, `scripts/pool_manager.py`, `scripts/mcp_server.py`, `scripts/preflight.py`, and `scripts/smoke_test_client.py`, plus the campaign's `set_env.sh`. A deployment is not considered ready until steps 7–8 succeed against the live endpoint.

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

## Timeouts and failures

Distinguish:

- **Submit rejection:** no endpoint task was accepted—inspect UUID, auth, and endpoint configuration.
- **Scheduler wait:** PBS block is queued—do not blindly resubmit. Scheduler-wait allowance is independent of payload timeout.
- **Payload failure:** an agent returned a normal result with nonzero `returncode`—use stderr to fix/retry only that payload.

Never alter queue or resource values outside the endpoint user-config schema. Never call scheduler commands from the harness.

## Agent capabilities and filesystem inspection

The Academy agent has exactly two public capabilities:

- `run_code(payload, workdir, timeout)` — execute a Python payload on the agent's pinned accelerator.
- `inspect_filesystem(operation, path, offset, limit)` — read-only visibility into the worker filesystem without SSH.

`inspect_filesystem` permits only:

| Operation | Semantics | Required bounds |
|---|---|---|
| `ls` | Sorted immediate directory entries with type and size metadata; no recursive traversal. | At most 200 entries. |
| `cat` | UTF-8 text read, with line numbers. | `offset ≥ 1`, at most 200 lines and 64 KiB. |
| `tail` | UTF-8 text read from end, with line numbers. | At most 200 lines and 64 KiB. |

The action resolves paths and restricts them to the explicit allowed roots: `HPC_BASE_DIR`, `HPC_INSPECT_ROOTS` (colon-separated optional additional roots), and `/tmp`. It rejects symlink escapes, binary data, directories passed to `cat`/`tail`, device files, and every write/mutation operation. Do not add a generic `shell`, `exec`, `find`, `grep`, environment dump, scheduler query, or recursive listing action. For programmatic analysis beyond bounded inspection, write it as a `run_code` payload.

## Non-negotiable constraints

- `HpcAgent` exposes only `run_code` and bounded, read-only `inspect_filesystem`; an internal health probe is permitted but is never a public MCP tool.
- One agent is pinned to one addressable accelerator; subprocesses inherit its pinning environment.
- Preserve endpoint UUID, endpoint config, campaign ID, agent IDs, pins, seeds, commands and outputs.
- This skill contains no molecular-design logic and exposes no additional MCP capabilities.
