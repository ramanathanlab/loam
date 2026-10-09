---
name: remote-hpc-execution
description: Debug and verify stateful remote code-execution paths spanning clients, MCP services, Academy agents, Globus Compute, schedulers, and workers.
version: 1.0.0
---

# Remote HPC Execution

Use this skill when verifying or debugging a remote execution plane such as MCP → agent runtime → Globus Compute → scheduler → worker. This is for real execution, not scheduler-only diagnostics.

## Acceptance rule

Do not call the path working until a client receives a completed application result. Endpoint status, function registration, a submitted task, an allocated/queued/running job, a worker registration, or raw MCP listener reachability are intermediate evidence only.

Capture returned provenance appropriate to the workload: request/task/function IDs, endpoint/user-endpoint identity, scheduler job ID, worker hostname, device placement, runtime, stdout/stderr/return code, and application result.

## 1. Establish one tight, low-risk probe

1. Start with one deterministic application payload that asserts the desired behavior and emits structured JSON provenance.
2. Use exact arithmetic or other deterministic inputs; for rational arithmetic use `fractions.Fraction`, never floats.
3. Exercise layers in order:
   - direct executor task to establish endpoint/worker behavior;
   - agent-runtime launch and health/probe;
   - MCP tool invocation returning the same application result.
4. Maintain a concise evidence table: component, identifier, observed state, and returned result.

## 2. Treat endpoint configuration as stateful

A request configuration can create or select a user endpoint and scheduler pool. It is not a harmless per-test flag.

1. Pick one stable endpoint config for a debugging epoch.
2. Start one retained endpoint/pool and one retained MCP service.
3. Do not restart the service or alter user endpoint config between equivalent smoke retries.
4. Before a new launch, inspect active user endpoints, worker registrations, scheduler jobs, and per-user queue limits.
5. If queue limits or stale pools appear, stop parallel submissions and clean up only known, in-scope stale jobs/processes. Reuse the retained healthy allocation rather than multiplying registrations.

## 3. Instrument the actual boundaries

For each request, log or preserve:

1. MCP request and tool response.
2. Academy agent ID and lifecycle state.
3. Globus Compute task/future ID and submission exception, if any.
4. User endpoint registration and endpoint logs.
5. Scheduler job ID, state, queue comment, host, and walltime.
6. HTEx/worker registration and worker-side logs.
7. Application result and runtime provenance.

When an Academy manager reports an unhealthy agent, inspect its background launch task (diagnostic/private API if necessary). Distinguish submission/serialization failure, worker startup failure, exchange-connectivity failure, and action failure. A generic pool-health error is not root cause.

## 4. Version and capacity preflight

Before launching long-lived agents:

1. Compare agent-runtime package versions and protocol versions between client and worker. Align compatible versions before diagnosing transport behavior.
2. Confirm worker-side imports for the agent module and runtime are available.
3. Account for long-lived agents consuming workers. Pool size must not exceed worker capacity.
4. For GPU-pinned workers, validate that each adopted agent has a unique pin before accepting a pool.
5. Propagate campaign/request metadata deliberately into code subprocess environments; constructor state alone may not appear in child processes.

## 5. Pool discovery safety

Remote discovery may contain stale registrations from interrupted clients.

1. Bound health probes used during discovery separately from launch-health timeout.
2. Prefer existing verified in-memory handles; do not re-discover/adopt remote agents if the retained pool already satisfies its requested size.
3. Deduplicate by agent ID and validate campaign ID plus placement before adoption.
4. Fail closed on duplicate device pins, mismatched campaign IDs, missing task provenance, or missing application output.

## Approval and policy-gate discipline

If a command is denied by an approval or policy gate with an explicit prohibition on retrying, rephrasing, or finding an alternate route, stop the affected workflow immediately. Do not substitute another remote command, script, or transport to obtain the blocked state. Report only states already verified, identify the blocked check at a high level, and wait for renewed user direction or approval before performing dependent endpoint actions.

## 6. Verification and teardown

1. Run exactly one MCP smoke task after a stable pool is healthy.
2. Require complete structured result from the worker.
3. Save the result/provenance before starting any larger workflow.
4. Stop temporary MCP services and endpoint-managed pools when finished; cancel only jobs created by the investigation.
5. Record unresolved external scheduler constraints separately from application/runtime defects.

## Pitfalls

- `online` is manager availability, not proof that the particular user endpoint used for a request is ready.
- A scheduler allocation proves placement, not task completion.
- Repeated changed configurations can produce separate user endpoints and hit scheduler per-user queue/job limits.
- A raw executor can satisfy an agent manager's executor interface while the agent runtime still fails due to serialization, import, exchange connectivity, or lifecycle state.
- Do not present a direct executor success as proof that MCP/agent execution works; preserve the distinction.

## References

- `references/approval-denial-discipline.md`: mandatory stop behavior and reporting shape when a command approval/policy gate rejects remote-execution work.
- `references/globus-academy-polaris-debugging.md`: worked evidence model, version/capacity pitfalls, and diagnostic boundaries from a Polaris/Globus Compute/Academy investigation.
