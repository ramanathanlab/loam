# Globus Compute + Academy/FastMCP incident pattern

## Layered acceptance checklist

| Layer | Required positive evidence |
|---|---|
| Globus endpoint | User endpoint is ready; endpoint status alone is not enough. |
| Scheduler/HTEx | Scheduler block is running **and** the HTEx manager registered workers. |
| Direct Compute | A bounded SDK task returns its result and remote provenance. |
| Academy | A launched agent responds to a health/probe action. |
| MCP | `run_code` returns stdout, stderr, and `returncode`, with remote host/device/job provenance in stdout. |

## Reproduction pattern

1. Use one supported Globus Compute endpoint with an endpoint-managed scheduler configuration.
2. Start with one persistent Academy agent; do not request a pool larger than registered worker capacity.
3. Confirm the manager log includes worker registration and each worker’s accelerator pin before launching the Academy smoke call.
4. Use a unique campaign/agent namespace per clean retry to prevent stale remote registrations from being mistaken for active agents.
5. Run an MCP payload that emits JSON with hostname, scheduler job ID, CUDA/ZE pin, campaign ID, and Python version.
6. Only then run the target workload such as deterministic `fractions.Fraction` arithmetic.

## Durable implementation guidance

- Pin compatible Academy package/protocol versions on client and worker. Protocol mismatch can present as an agent-health timeout rather than a clear compatibility error.
- `CampaignPool` should have two timeouts: a short timeout for probing discovered agents and a longer launch-health timeout for a newly submitted agent.
- Do not rediscover/adopt agents after the known in-memory healthy pool already satisfies the requested size. This avoids duplicate device pins from stale registrations.
- Stage all worker-imported deployment modules into the shared remote workspace and verify content hashes after every code change. A local patch does not affect a remote worker unless its import path sees the staged copy.
- Explicitly set campaign/provenance environment variables for child code subprocesses. A persistent agent’s constructor state is not automatically inherited by `subprocess.run`.

## Frequent false positives

- Endpoint API reports `online`.
- PBS/Slurm block is queued or running.
- MCP listener answers a raw GET with its transport-specific error.
- Academy exchange has a registration.
- Client process exits without a returned remote task result.

## Cleanup discipline

Terminate superseded local MCP servers and bounded smoke clients before retrying. Preserve endpoint, scheduler, and worker logs with the allocation ID. Avoid parallel agent launches while a limited debug allocation is being diagnosed.
