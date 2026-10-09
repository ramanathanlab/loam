# Globus Compute + Academy + Polaris lifecycle notes

Use this reference when a local FastMCP service launches long-lived Academy agents through a Globus Compute endpoint backed by PBS/HTEx.

## Evidence ladder

Require a returned `run_code` result. The following are insufficient on their own: endpoint `online`, function registration, raw `GET /mcp` HTTP 400, a submitted/queued/running PBS block, HTEx worker registration, or model prose.

For a provenance-bearing arithmetic smoke result, require stdout/stderr/returncode plus:

- exact `Fraction` output: `sum=10`, `count=4`, `mean=5/2`, `threshold=12/5`, `exceeds=True`;
- compute hostname;
- PBS job ID;
- accelerator pin (`CUDA_VISIBLE_DEVICES` or site equivalent);
- Python version;
- campaign ID propagated into the executed child process.

## Stable deployment rule

Treat an MCP deployment as persistent. Keep one stable `user_endpoint_config`, one campaign identifier, one endpoint user endpoint, one retained scheduler block, and one local MCP server for a smoke attempt. Do not restart the MCP server with altered endpoint config/campaign settings for every probe.

A Globus Compute executor created with `user_endpoint_config` can cause new user endpoint registrations and PBS block requests when configurations change. Scheduler per-user limits may then surface to the client as `ENDPOINT_NOT_ONLINE`; inspect user-endpoint logs and PBS state before retrying.

## Academy agent pool rules

- Match local and worker Academy protocol/package versions before debugging higher layers.
- Long-lived Academy agents consume HTEx workers for their lifetime. Set initial pool size to one, then raise it only after the single-agent smoke result returns.
- Ensure the HTEx worker count and accelerator pins can support the requested pool size.
- Bound probes of stale exchange registrations separately from the longer launch-health timeout.
- Do not discover/adopt additional agents when the healthy in-memory pool already meets target size; stale registrations can produce duplicate accelerator pins.
- Propagate campaign metadata explicitly into the child process launched by `run_code`; agent constructor state alone does not guarantee environment inheritance.

## Endpoint startup repairs observed on Polaris

When the endpoint manager runs a sanitized child environment, place needed executable and scheduler paths in the persistent user endpoint environment. Verify the deployed user endpoint can resolve both `globus-compute-endpoint` and `qsub` before diagnosing Academy.

## Minimal continuation order

1. Verify no stale local MCP process or duplicate queued PBS block remains.
2. Start the endpoint once and wait for its specific user endpoint to signal readiness.
3. Start one local MCP service with pool size one and stable configuration.
4. Send one exact-Fraction `run_code` request.
5. Capture the returned result and only then test multi-agent capacity or campaign integration.
