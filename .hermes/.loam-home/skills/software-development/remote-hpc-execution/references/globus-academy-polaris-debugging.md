# Globus Compute + Academy + Polaris: diagnostic reference

Use as an evidence checklist, not as a copy-paste deployment configuration.

## Layered acceptance

| Layer | What it proves | What it does not prove |
|---|---|---|
| Endpoint API `online` | Endpoint manager is visible | Particular user endpoint is ready |
| User endpoint logs connected to task/result queues | User endpoint initialized | A PBS worker is registered |
| PBS job running | Allocation exists | Worker is registered or application ran |
| HTEx manager registration | Worker pool connected | Academy agent/exchange health |
| Academy `probe` response | Agent is callable | Payload execution/provenance correctness |
| MCP `run_code` completed result | End-to-end execution | Only then accept remote work |

## Durable troubleshooting sequence

1. Use one direct Globus Compute probe to validate endpoint/scheduler/worker baseline. Include structured result: hostname, PBS job ID, device pin, runtime, and exact application assertion.
2. Align Academy package/protocol versions on client and worker. A raw `globus_compute_sdk.Executor` satisfies Academy Manager's `concurrent.futures.Executor` contract, but that alone does not validate code serialization, remote imports, or exchange connectivity.
3. Configure the retained endpoint pool once. For long-lived agents, choose worker capacity at least equal to requested agent pool size. For GPU placement, ensure workers have distinct pins.
4. Start one MCP service and invoke a single simple `run_code` request. Preserve agent ID, task ID, scheduler job, and worker result.
5. Only after simple code succeeds, submit exact workload and require its complete returned provenance.

## Academy Manager diagnostic boundary

`Manager.launch()` returns a handle immediately and creates a background long-lived executor task. A later `Handle.action("probe")` timeout only says the handle did not respond; it does not identify why.

For a controlled debug run, inspect the manager's background task associated with the handle:

```python
acb = manager._acbs[handle.agent_id]  # diagnostic/private API
if acb.task.done():
    print(acb.task.exception())
```

Interpretation:
- background task exception: submission, serialization, import, or worker launch failure;
- task remains live but probe times out: agent/exchange connectivity or runtime health issue;
- probe succeeds but action fails: application/action boundary.

## Stateful-configuration pitfall

A Globus Compute `user_endpoint_config` may cause a distinct user endpoint and scheduler pool to be created. Repeatedly starting MCP servers with changed config can multiply user endpoints, queued PBS jobs, and stale Academy registrations.

Symptoms may include `ENDPOINT_NOT_ONLINE` even while the endpoint API reports online, or scheduler messages about per-user queued/running job limits. Stop retries, inspect registrations/jobs, clean only investigation-owned stale work, and reuse one retained configuration/pool.

## Pool-discovery pitfall

Exchange discovery may return stale agent registrations after client interruption. Health probes during discovery must have a short timeout distinct from agent-launch timeout. Do not discover/adopt agents if the current verified in-memory pool already reaches target size. Validate campaign ID and placement pin before adoption and fail closed on duplicate pins.

## Metadata propagation pitfall

Agent constructor fields are not automatically environment variables in a subprocess started by the agent. Explicitly pass campaign/request metadata into the subprocess environment when it is required in returned provenance.
