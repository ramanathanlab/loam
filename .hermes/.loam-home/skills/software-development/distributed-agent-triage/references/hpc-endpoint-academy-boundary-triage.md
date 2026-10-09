# Globus Compute / Academy Boundary Pattern

## Symptom

A Globus Compute endpoint can complete known direct work (such as fraction probes), while Academy agents never appear healthy.

## Interpretation

Direct work proves scheduler allocation, endpoint transport, Parsl manager/worker registration, and execution for those direct tasks. It does **not** prove that `Manager.launch(...)`, agent discovery, or a `Handle.action("probe")` request reached the endpoint.

## Read-only evidence to collect

- `qstat -fx <jobid>`: scheduler job state, host, resource usage, stdout/stderr locations.
- Active endpoint `endpoint.log`: endpoint startup, AMQP readiness, scale-out, activity/idle transitions.
- `interchange.log`: manager registration and registration-to-ready-queue events.
- `block-*/<manager-id>/manager.log`: discovered interchange URL and connection.
- `block-*/<manager-id>/worker_*.log`: worker start, received executor task IDs, completion timestamps.
- Generated submit script plus scheduler stdout/stderr: `TMPDIR`, proxies, environment activation, working directory, `PYTHONPATH`, and exact worker-pool command.
- The *client-side* Academy/MCP launcher: process state plus logs for launch, discover, and probe.

## Diagnostic conclusion rule

If endpoint/worker logs contain only known direct-task IDs and no task correlated to an Academy launch or probe, the defensible location is:

> The failure is upstream of endpoint task dispatch, in the client-side Academy/MCP launch, exchange, discovery, or health-probe path.

Do not name a specific Academy exception unless it appears in the client-side evidence.

## Helpful report form

1. **What is working:** scheduler job, worker registration, direct-task completions.
2. **What is absent:** Academy-specific dispatch/probe and any endpoint-side error.
3. **Exact boundary:** upstream of endpoint dispatch, or the specific worker action/runtime phase if task evidence exists.
4. **Remaining evidence required:** client/MCP process logs if no client process/logs were available.
5. **Safety:** explicitly confirm that no endpoint or scheduler changes were made.
