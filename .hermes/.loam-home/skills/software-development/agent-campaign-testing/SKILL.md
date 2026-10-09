---
name: agent-campaign-testing
description: Run and debug isolated multi-round Hermes/Loam agent campaigns with nested delegation, lifecycle evidence, and fail-closed acceptance criteria.
---

# Agent campaign testing

Use for an end-to-end campaign that exercises a coordinator → validator → worker/code-generator → synthesis/reasoner loop over multiple turns.

## Isolation first

1. Read the campaign contract and relevant adapter, hook, and agent-prompt files before launching.
2. Create one temporary root containing the campaign profile/config and a single state directory.
3. Use one stable run ID throughout the campaign. Do not reuse a partial campaign state for a retry.
4. Preserve an explicitly supplied `HERMES_HOME` when the launcher supports isolated homes; verify the effective config path before dispatching.
5. Confirm the local model endpoint and the exact Python/runtime command required by the contract before starting a persistent coordinator.

## Nested delegation contract

For a batch `delegate_task(tasks=[...])`, role is per task. A top-level role is not enough.

```python
{
  "goal": "... [loam-agent-role=hypothesis-validator] [loam-plan-id=...] ...",
  "context": "full hypothesis and execution contract",
  "role": "orchestrator",
}
```

- Set each validator task to `role: "orchestrator"` when it must spawn a code-generator.
- Set each code-generator child to `role: "leaf"`.
- Require a tag/plan identifier on every lifecycle-relevant child goal.
- Add a pre-tool guard that rejects missing per-task roles and untagged substitute validator dispatches before partial work begins.

## Run protocol

1. Use a persistent coordinator session; one-shot sessions can exit after background dispatch and cannot perform validator → reasoner handoff.
2. Run one coordinator turn at a time.
3. Before advancing, inspect persisted state for: completed tournament, validator starts/stops, code-generator starts/stops, parsed scores, and reasoner completion.
4. Stop a specific campaign attempt at the first unmet acceptance condition, but do not stop the overall task after reporting it: diagnose, patch, verify, create a fresh isolated root, and retry until the requested multi-round acceptance path actually completes or an external blocker is concrete and reported.
5. Treat a user request to continue—or frustration about premature status reports—as a standing delivery constraint. Do not issue an intermediate final response while a viable acceptance path, an active coordinator/backend process, or a concrete recovery action remains; continue autonomously until verified completion or a concrete external blocker after the applicable recovery attempt.
6. Never infer lifecycle events, execution, evidence, or credit from summaries alone. A partial round, unit tests, or configuration checks are diagnostic progress only—not delivery.
7. Treat Jnana rankings and reasoner text as advisory; do not convert Elo or unreviewed summaries into accepted empirical evidence.

## Remote/HPC execution backend gate

Before replacing local code-generator execution with a remote backend (for example an HPC MCP/Globus service), treat backend readiness as a separate fail-closed campaign prerequisite:

1. Obtain endpoint metadata/status and validate the requested allocation shape against its published user configuration schema.
2. Submit one harmless, importable top-level probe function—not a lambda or inline-only closure—and retain the returned task identifier.
3. Do not call an allocation/executor ready merely because an endpoint is `online` or a submission was attempted. Require a completed probe result with placement evidence (hostname/job id) and the expected runtime/device evidence.
4. If endpoint-side validation/startup fails, preserve its returned error and repair/escalate the endpoint configuration. Do not route production code-generator credit through that backend until the probe succeeds.
5. Change the campaign execution contract only after the remote tool round trip is verified. Persist remote invocation target, task identifier, stdout/stderr/return code, and status as code-generator lifecycle evidence.

### Managed Polaris / Globus Compute repair ladder

For a managed multi-user Globus Compute endpoint, `online` is only a control-plane signal. Diagnose the manager and user endpoint separately before using it for campaign evidence:

1. Inspect the manager log and newest per-user endpoint log under `~/.globus_compute/`, preserving only non-secret facts: user-endpoint ID, scheduler job ID, queue, host, and error tail.
2. Require three milestones in order: the user endpoint starts; its Parsl provider successfully submits a PBS block; the HTEx interchange records worker registration. A bound interchange port or a running PBS allocation without worker registration is not ready.
3. Managed user endpoints may execute from a sanitized environment. Confirm the persistent managed environment exposes both the actual `globus-compute-endpoint` executable and PBS client commands such as `qsub`/`qstat`; an interactive login shell's PATH does not prove the forked child has the same PATH.
4. Apply durable endpoint environment/configuration fixes through the manager-supported configuration mechanism, restart the endpoint manager, and verify the newly started child before retrying. Do not rely on a temporary SSH export.
5. Keep the client SDK compatible with the version reported by the live endpoint unless compatibility has been independently proven. Hold the full user endpoint configuration stable during a probe series, because changed configurations can create separate dynamic user endpoints and consume debug-queue concurrency.
6. After every infrastructure repair, run one harmless exact-Fraction probe and require a completed remote result with placement evidence, device visibility, and the expected exact values before retrying Academy/MCP or campaign work.

When a live allocation is broken, use the supported Globus endpoint lifecycle to stop it rather than leaving it consuming queue time. Do not substitute raw scheduler submission for the required MCP/Academy/Globus execution path.

See `references/remote-hpc-execution-gate.md` for the Globus Compute/ALCF Polaris validation pattern and endpoint-template pitfalls.

## Persistent-session discipline

- Keep the coordinator process alive across all requested rounds. Do not substitute one-shot sessions or prematurely kill it after partial lifecycle evidence.
- Before issuing an intermediate status, check whether a coordinator or child remains active; continue driving the requested path unless the user explicitly asks to pause or stop.
- Require exactly one reasoner start/completion per plan. Once a reasoner has started, block further coordinator tool calls that could dispatch duplicate reasoners and wait for the terminal result before the next round.
- If a coordinator model rewrites or abbreviates a required delegation payload, canonicalize tagged validator, code-generator, and reasoner arguments at the tool-request boundary from immutable persisted plan state. Add tests that permit reasoner delegation after all validators complete while retaining rejection of untagged validator substitutions before completion.
- For any live remote probe, keep exactly one owner process and a recorded explicit timeout. Do not launch concurrent or delayed duplicate probes while the first is pending; wait for its result or timeout, inspect the endpoint state, then retry only after the prior task/process is terminal.

## Parsing and evidence

- Validator score parsing should accept the contract's common Markdown form, e.g. ``### Evidence Score\n`0.00` — rationale``.
- A code-generator must execute the prescribed calculation/tool action; a validator’s independent calculation does not substitute for required child execution.
- Persist child outputs as audit observations unless a separate source-validation/reconciliation layer accepts evidence.

## Verification

1. Add focused tests first for the exact role guard, lifecycle boundary, and output parser behavior.
2. Run those focused tests plus adjacent lifecycle/tournament tests.
3. Run a fresh isolated integration campaign only after checking the launcher actually honors the isolated `HERMES_HOME` and `max_spawn_depth` configuration.
4. Report focused/ad-hoc verification separately from a canonical project suite.

## References

- `references/loam-hermes-nested-delegation.md` records the reusable Loam/Hermes batch-role and isolated-launcher pitfalls.
- `references/polaris-globus-academy-control-plane.md` records the Polaris MCP → Academy → Globus Compute validation ladder, user-config consistency rule, and worker-registration diagnostics.
- `references/polaris-managed-endpoint-repair.md` records the managed-endpoint readiness ladder, sanitized-PATH diagnosis, and exact-Fraction revalidation contract.
