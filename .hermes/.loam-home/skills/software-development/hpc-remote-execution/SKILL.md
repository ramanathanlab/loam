---
name: hpc-remote-execution
description: Deploy, debug, and verify remote code execution through an MCP/agent layer backed by Globus Compute or scheduler-managed HPC workers. Use for Academy/FastMCP/Globus/PBS execution planes, not direct scheduler substitutions.
---

# HPC Remote Execution

## Trigger
Use when a local agent or MCP server launches long-lived remote agents or code tasks through Globus Compute, PBS/Slurm, Parsl/HTEx, Academy, or equivalent systems.

## Evidence standard
Treat the execution plane as a chain and verify each boundary separately:

1. Endpoint readiness and user-endpoint startup.
2. Scheduler block allocation and worker registration.
3. A completed direct remote SDK task.
4. A completed MCP/agent `run_code` task.
5. Returned application provenance: task/function ID, host, scheduler job ID, GPU/device placement, interpreter/runtime, stdout/stderr, and return code.

Never accept endpoint `online`, a queued/running scheduler job, raw MCP HTTP reachability, an agent registration, or process exit alone as proof of remote computation.

## Debugging workflow

1. **Establish a lower-level control test.** Submit one bounded, deterministic task directly through the remote SDK. Record its task/function ID and returned result. Do not replace the supported endpoint with direct scheduler submission.
2. **Inspect compatibility on both sides.** Compare client and worker Python, serializer, SDK, agent-framework, and protocol versions. Align agent-framework protocol versions before diagnosing higher-level behavior.
3. **Inspect the live worker plane.** Confirm scheduler state, HTEx/worker-manager registration, worker count, accelerator pins, and task receipt in endpoint logs. A scheduler allocation without registered workers is insufficient.
4. **Match long-lived-agent demand to worker capacity.** A persistent agent consumes a worker for its lifetime. Request at least one registered worker per concurrently desired agent, plus capacity for any independent task execution requirements.
5. **Bound discovery separately from launch.** Discovery of stale remote-agent registrations must have a short timeout. Agent launch/health may need a longer timeout. Never let stale discovery consume the entire tool-call deadline.
6. **Keep pool ownership strict.** Prefer already-known healthy campaign agents. Only discover/adopt remote registrations when the in-memory pool is below the requested size. Deduplicate by agent ID and reject duplicate device pins.
7. **Stage code deterministically.** When workers import files from a shared remote workspace, copy modified sources before launching and compare hashes on local and remote copies. Compile both copies before test execution.
8. **Run one clean, bounded smoke call.** Use a unique campaign/agent namespace to avoid stale registrations. Capture the returned `run_code` object, not just client logs. Stop superseded local servers and test calls before retrying.
9. **Propagate provenance deliberately.** Constructor state in a persistent agent is not automatically present in subprocesses. Explicitly inject campaign IDs and required provenance variables into the subprocess environment.
10. **Validate application acceptance in the client.** Make the smoke client fail closed unless the MCP result has exactly `stdout`, `stderr`, and `returncode`; the return code is successful; stdout is parseable structured data; and every mandated calculation and provenance field is present, non-empty, and correct. A successful transport response alone is not application evidence.
11. **Treat endpoint-start timeouts as unknown state.** Before issuing another start request, inspect endpoint status, endpoint-manager process/logs, user-endpoint processes, and scheduler state. A start CLI may outlive the caller timeout and successfully bring up the daemon.
12. **Honor the declared target and active consent.** When the handoff identifies a named HPC system/endpoint, repeat that target in the execution record and do not substitute a similarly named environment. Once the user has granted real remote-execution permission for the current task, proceed with the bounded, documented operation rather than repeatedly asking for the same authorization.

## MCP smoke payload
Use a minimal payload first: print JSON containing hostname, scheduler job ID, accelerator pin, campaign ID, and runtime. Then run the target deterministic computation. Fail closed if the returned object lacks stdout, stderr, and a successful return code.

## Common pitfalls

- Treating a direct SDK success as MCP/agent success. They validate different layers.
- Mixing incompatible agent-framework protocol versions across client and worker.
- Launching a multi-agent pool into a one-worker allocation.
- Allowing stale exchange registrations to block discovery or be adopted into a new campaign pool.
- Running stale shared-workspace source after editing local deployment code.
- Assuming agent object state becomes subprocess environment state.
- Retrying in parallel: multiple long-lived launches confound queue state and consume scarce debug allocations.

## Cleanup and verification
After a failed attempt, terminate only the local MCP server/client and known campaign agents as appropriate; preserve logs and scheduler IDs. Recheck worker registration before retry. Before reporting success, require a completed returned MCP result and correlate it to the scheduler allocation.

See `references/globus-academy-debugging.md` for a compact incident pattern and verification checklist. See `references/globus-academy-polaris-lifecycle.md` for Globus Compute + Academy lifecycle constraints, scheduler-limit interpretation, and the single-stable-deployment verification order.
