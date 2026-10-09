---
name: distributed-agent-triage
description: Use when a scheduler-backed remote executor can run ordinary tasks but an agent, pool, exchange, or MCP worker is unhealthy. Isolate the failing boundary with read-only, evidence-backed correlation.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [distributed-systems, debugging, hpc, scheduler, agents, mcp]
    related_skills: [systematic-debugging]
---

# Distributed Agent Triage

## Overview

A running scheduler job and a completed direct task do **not** establish that an agent framework is healthy. Agent execution crosses distinct layers: client/launcher, exchange or control plane, endpoint/interchange, scheduler allocation, worker runtime, and the agent's own startup and health action.

Use this skill to establish the earliest layer that lacks evidence. Investigate read-only unless the user explicitly authorizes changes.

## When to Use

- A Globus Compute, Parsl, Slurm, PBS, or similar endpoint completes direct probes but agents never become healthy.
- An MCP service reports no workers although scheduler jobs are running.
- A distributed agent pool is stuck launching, discovering, or health-checking.

Do not use this for a single local process crash with a complete traceback; use ordinary systematic debugging instead.

## Evidence Boundaries

Maintain a timeline across these layers:

1. **Scheduler/allocation** — job state, node assignment, resources, stdout/stderr.
2. **Endpoint transport** — endpoint process, queue/control connection, interchange or manager registration.
3. **Executor worker** — worker starts, receives task IDs, completes task IDs, and emits runtime errors.
4. **Agent control plane** — launch request, agent discovery, health/probe action, and its response.
5. **Client/MCP server** — process lifecycle, client logs, exchange credentials, and tool-facing errors.

A direct-task success proves only layers 1–3 for that task. Do not infer layer 4 or 5 health from it.

## Read-Only Investigation

1. **Establish the live allocation.** Capture scheduler state, job ID, allocated node(s), resource usage, and the generated scheduler script. Read scheduler stdout and stderr. Completion: know whether the allocation launched and whether bootstrap failed.
2. **Verify endpoint-to-worker transport.** Read endpoint, interchange, manager, and worker logs. Record manager registration, worker startup, and the discovered transport address. Completion: identify whether a worker is actually connected.
3. **Correlate tasks by timestamp and task ID.** Classify each worker task as a known direct probe, an agent launch, a health action, or unknown. Completion: determine whether an agent launch/probe ever reached executor dispatch.
4. **Inspect the client-side agent boundary separately.** Find the MCP server/launcher process and logs; inspect agent-launch, discovery, and health/probe errors. Completion: capture the actual exception or establish that client evidence is unavailable.
5. **Report the narrowest supported failure location.** If endpoint logs contain no agent-specific work, report "upstream of endpoint task dispatch" rather than naming a speculative framework defect. If an agent task reaches the worker and fails, quote its task/runtime error and locate it at worker startup or action execution.

## Log-Reading Discipline

- Bound log output. Registration records can include huge package inventories; extract only lines around `registration`, `Received executor task`, `Completed executor task`, `HOLD_WORKER`, `Traceback`, `Exception`, and relevant task IDs.
- Preserve timestamps and exact paths in the report.
- Empty scheduler stderr plus worker registration and task completion is positive evidence against scheduler/bootstrap failure.
- Treat unavailable node SSH, restricted scheduler inspection, or shell-command formatting errors as **inspection limitations**, never as system-failure evidence.

## Common Pitfalls

1. **Equating a running PBS/Slurm job with a healthy agent.** Check the agent launch and probe path explicitly.
2. **Searching only endpoint logs.** The failure may occur before a task is submitted, in the launcher, MCP service, exchange client, discovery, or credential path.
3. **Calling absence of errors a success.** When no agent task exists in worker logs, the useful conclusion is an upstream dispatch boundary—not agent health.
4. **Flooding analysis with full package inventories.** Use narrow, timestamped log excerpts so true error signals remain visible.
5. **Changing endpoint or scheduler configuration during diagnosis.** First prove the boundary from existing evidence; changes destroy the state needed for correlation.

## Verification Checklist

- [ ] Scheduler job and generated script inspected without modification.
- [ ] Endpoint, manager, and worker registration evidence collected.
- [ ] Direct-task IDs/times distinguished from agent launch/probe work.
- [ ] Agent/MCP client process and logs inspected, or their absence explicitly recorded.
- [ ] Conclusion identifies an evidence-backed boundary and names remaining missing evidence.

See `references/hpc-endpoint-academy-boundary-triage.md` for a concise Globus Compute/Academy case pattern.
