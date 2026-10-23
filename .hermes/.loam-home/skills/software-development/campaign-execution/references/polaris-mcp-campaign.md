# Polaris MCP-backed campaign integration

Use this reference when an isolated Loam/Hermes campaign must force code-generator execution through a live remote MCP server.

## Known-good routing shape

- Start and retain the FastMCP server separately, then target it with an isolated `HERMES_HOME`.
- Add the remote server by URL (the successful endpoint was `http://127.0.0.1:8000/mcp`).
- Do not assume the config-facing tool name is the runtime identifier. After `hermes mcp add`, inspect the registered tool names in the isolated agent log or tool listing.
- In this run, server `hpc-execution`, tool `run_code` became runtime identifier:
  `mcp__hpc_execution__run_code`
  — not `mcp_hpc_execution_run_code`.
- Store that exact runtime identifier in the campaign execution profile and require it in code-generator briefs.

## Isolation and preflight

1. Create a temporary campaign root containing separate `hermes-home/` and `state/` directories; copy only the intended baseline config.
2. Set `HERMES_HOME`, `LOAM_CONFIG_PATH`, `LOAM_STATE_DIR`, and a stable `LOAM_RUN_ID` for every coordinator command.
3. Ensure nested delegation permits coordinator → validator → code-generator (`max_spawn_depth: 2` where needed).
4. Run `hermes mcp test <server>` under that isolated `HERMES_HOME` before launching a round.
5. Run the Loam orchestration unit tests before the live campaign.

## Fail-closed acceptance audit

Audit the generated state rather than trusting coordinator prose:

- `run-manifest.json`: confirms `mode: remote_mcp`, endpoint, function, and runtime tool name.
- `hypothesis-orchestration.jsonl`: count `code_generator_completed`, `validator_completed`, and `reasoner_completed`; confirm one reasoner per requested round.
- isolated `agent.log`: search for `tool mcp__hpc_execution__run_code completed` to prove tool routing.
- each code-generator evidence record must contain return code 0, empty stderr, and remote placement metadata.
- verify the MCP server remains reachable after the campaign; if relevant, query the scheduler to confirm the retained allocation is still running.

For a five-round, two-hypothesis campaign, the expected minimum completed-event counts are 10 code generators, 10 validators, and 5 reasoners across five distinct plan IDs.

## Evidence interpretation

Two validator reports that cite the same remote stdout, job ID, host, and placement are duplicate assessments of one execution. Do not call them independent corroboration. Report the arithmetic/technical result as supported by the execution evidence, while preserving any provenance or independence caveat.

## Common pitfall

A campaign may look configured correctly but fail before code execution solely because the configured MCP tool name differs from Hermes' runtime-normalized identifier. Detect the runtime identifier after registration and update the isolated config before launching; do not proceed on a failed first round as if it were acceptance evidence.
