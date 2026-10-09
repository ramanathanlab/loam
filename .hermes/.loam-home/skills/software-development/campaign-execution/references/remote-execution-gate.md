# Remote execution acceptance gate

Use this when a campaign requires an HPC, MCP, or other remote code-execution backend.

## Evidence hierarchy

Do not credit a campaign round from any of the following alone:
- endpoint status (`online`);
- a registered remote function;
- a queued or running scheduler allocation;
- worker/bootstrap logs; or
- model-reported arithmetic.

Require a completed returned task result for each required code-generator execution. Preserve the task/function ID, remote endpoint ID, returned stdout/stderr/return code, and placement provenance.

## Exact-arithmetic minimum

For Fraction-based arithmetic, require remote output showing exact—not floating-point—values:

- `sum: 10`
- `count: 4`
- `mean: 5/2`
- `threshold: 12/5`
- `exceeds: true`

Also require remote hostname, scheduler job ID, and accelerator placement where applicable (for example `CUDA_VISIBLE_DEVICES`).

## Layered readiness

Verify execution in this order:

1. Complete one direct remote probe and persist its returned result.
2. Complete one MCP/API call through the intended server path and verify the same provenance reaches the caller.
3. Ensure campaign code-generator contracts permit only the configured remote tool and explicitly forbid local fallback.
4. Start a fresh isolated campaign only after steps 1–3 pass.

Keep scheduler queue time, worker/bootstrap time, remote-agent health time, server call timeout, and client request timeout separate. The client call timeout must cover the server's agent-health window plus realistic allocation/bootstrap time; otherwise a client timeout is not proof that the backend failed.

## Retry discipline

Do not start duplicate endpoint pools/probes while a one-node allocation is queued or running. Inspect and reuse the active allocation first. A scheduler `not enough free nodes` queue comment is a capacity condition, not configuration evidence.
