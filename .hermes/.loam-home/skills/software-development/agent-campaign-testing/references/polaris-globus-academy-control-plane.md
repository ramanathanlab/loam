# Polaris Globus Compute and Academy control-plane notes

Use this alongside the remote HPC gate when an ALCF Polaris campaign uses a local FastMCP server, Academy agents, and a Globus Compute flex-scaling endpoint.

## Proven preflight sequence

1. Inspect endpoint metadata and its live user-config schema/template under the submitting identity. Preserve only non-secret fields.
2. Use the smallest allowed debug request and reuse one *identical* `user_endpoint_config` for every probe, warm-up, and Academy executor. Changing config values causes distinct spawned user endpoints and can consume the per-user debug queue limit.
3. Stage the Academy agent module, pool manager, and MCP server into a campaign-specific directory on the shared project filesystem. Compile/import them with the worker environment before dispatch.
4. Probe first through Globus Compute with registered source code or an importable module-level function. A source-registered function avoids client/worker Python and dill module-resolution mismatches that can make otherwise healthy allocations appear broken.
5. Require all of the following before declaring the backend ready:
   - completed probe result with hostname, PBS job ID, and accelerator visibility;
   - corresponding PBS job actually running;
   - worker registration in the endpoint HTEx/interchange logs, not merely a bound listener;
   - an MCP `run_code` round trip returning stdout, stderr, return code, and remote task identity.

## Polaris template requirements

- A caller passes `walltime` as a normal `HH:MM:SS` string. The service may JSON-encode string values before Jinja rendering; templates must not add static outer YAML quotes around an already encoded expression.
- `GlobusComputeEngine.strategy` must be a plain string such as `"simple"`; a nested mapping reaches Parsl's strategy lookup and produces `TypeError: unhashable type: 'dict'`.
- Use `bond0` for the Polaris worker-to-login-node interchange interface, `TMPDIR=/tmp` before worker launch, and include `home:grand` in PBS filesystem options.
- Preserve the worker environment setup: CUDA module, intended Python environment, `PYTHONPATH` pointing at the staged campaign directory, and one GPU pin per worker.

## Failure classification

- `endpoint online` is control-plane status only.
- A spawned user endpoint plus a running PBS block is still insufficient if the HTEx listener has no registered workers; investigate compute-worker launch/network registration before retrying the same MCP call.
- If the debug queue reports a per-user running-job limit, do not repeatedly submit altered configurations. Reuse or drain the existing allocation, then retry with the exact same configuration.
- Treat serialization errors separately from endpoint failures. Repair function registration/shipping first, then re-run the minimal probe before escalating scheduler or endpoint configuration.
