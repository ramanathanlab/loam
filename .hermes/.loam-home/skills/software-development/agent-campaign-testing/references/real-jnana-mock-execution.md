# Real Jnana with mocked execution: evidence boundary

Use this pattern when validating Loam campaign infrastructure without submitting
real scientific/HPC work.

## Required topology

```text
real design prompt
  -> real model endpoint
  -> real Jnana hypothesis generation
  -> real pairwise tournament / Elo state
  -> real coordinator-validator-reasoner lifecycle
  -> code-generator
  -> explicitly named mock execution tool only
```

The mock belongs at the execution boundary, not at hypothesis generation or
ranking. The mock must have an unmistakable target such as:

```yaml
execution:
  mode: remote_mcp
  tool_name: mock_hpc_execution_run_code
  endpoint: mock://empty
  function_name: run_code
  input_schema: mock-empty-v1
  runtime: python3.11
```

## Acceptance and reporting

- Inspect the Jnana state file and tournament reports. Require real hypotheses
  and `matches_completed == matches_requested` before calling the tournament
  exercised.
- Inspect persisted steering/reasoner artifacts separately from execution
  outputs.
- Verify policy blocks validator direct execution and code-generator alternate
  tools, while allowing the configured mock tool.
- A mock return value is not scientific evidence and cannot satisfy a campaign's
  empirical execution criterion.
- If the real model/Jnana request is rejected, times out, or yields fewer than
  two hypotheses, stop at that gate. Preserve the provider error and report
  `tournament not exercised`; never create replacement hypotheses or mock Elo
  reports.

## Isolation

Use a fresh temporary campaign state and one stable run ID. Keep the design
prompt unchanged except for an explicit test annotation if required by the
provider. Do not reuse an earlier partial Jnana state, because it can make a
mock run appear to have real hypotheses or rankings.

## Minimal evidence table

| Layer | Evidence | Counts as real? |
|---|---|---:|
| Model request | HTTP response and provider/model identity | Yes |
| Hypothesis generation | persisted hypotheses in Jnana state | Yes |
| Tournament | immutable report with completed matches | Yes |
| Steering | reasoner brief persisted/injected into next turn | Yes |
| Execution | mock tool response | No; boundary-only |
| Scientific result | structures/metrics/artifacts | No, not run |
