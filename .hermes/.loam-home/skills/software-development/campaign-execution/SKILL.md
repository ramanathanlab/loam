---
name: campaign-execution
description: Run strict multi-round agent campaigns end-to-end with isolated state, nested delegation, immutable artifacts, and fail-closed acceptance checks.
---

# Multi-Round Campaign Execution

Use for any request to run or validate a multi-round Hermes/Loam-style campaign where a partial round must not count as success.

## Core operating rule

Continue autonomously until the requested accepted round count has actually completed, or a concrete external blocker has been attempted and evidenced. Do not stop after source edits, unit tests, one successful child lifecycle, partial rounds, or a progress report.

## Workflow

1. Read the campaign acceptance specification and trace the relevant lifecycle hooks, agent-role handling, artifact writers, and policy guards before editing.
2. Create a new isolated campaign root for each retry. Keep `HERMES_HOME`, campaign state, configuration, logs, Python/runtime shim, and run ID inside that root. Never reuse a root containing an invalid or partial round.
3. Verify prerequisites before launching: endpoint reachability, actual config consumption, nested-spawn depth, plugin loading, and local execution runtime.
4. Run exactly one round first. Inspect persisted artifacts rather than trusting model claims:
   - tournament completed;
   - selected hypotheses persisted;
   - validators have the required role;
   - each validator produces its required child lifecycle;
   - local execution evidence exists and matches the required runtime/math;
   - evidence scores are valid;
   - exactly one reasoner completes after validators.
5. If any criterion fails, stop that campaign root immediately, diagnose the first unmet criterion, patch the root cause, run focused regression tests, and retry from a fresh root.
6. Once one round is valid, continue rounds sequentially in the same isolated campaign state. Before accepting each successor, prove the predecessor's reasoner synthesis/steering was persisted and injected into the next round.
7. Finish only after auditing campaign-only artifacts against every acceptance criterion and the requested exact round count.

## Nested delegation reliability

When the coordinator model must reproduce an exact `delegate_task` shape, prose alone is unreliable. Prefer canonicalizing recognized campaign delegation at tool-request middleware from persisted state:

- coordinator calls are rebuilt as the plan's complete validator batch;
- validator calls are rebuilt as the exact tagged leaf execution task;
- post-validator calls are rebuilt as the one plan-specific reasoner task.

Keep canonical reconstruction narrowly scoped to opted-in campaign workflows. Retain fail-closed policy checks for malformed calls, but sequence them correctly: validator-only guards apply before validators finish; after they finish, permit exactly one reasoner; once reasoner start is recorded, block duplicate dispatch until its terminal result.

## Evidence and provenance rules

- Rankings and agent summaries are advisory, never empirical evidence by themselves.
- Require concrete execution evidence from the mandated local/remote runtime.
- Persist append-only lifecycle and campaign records.
- Treat missing lifecycle, missing output, invalid score, duplicate reasoner, or missing steering as a non-crediting failure.

## Verification

After code changes, run focused lifecycle/policy tests, syntax/type checks appropriate to touched files, and an isolated ad-hoc probe of the changed runtime integration. Label focused/ad-hoc results accurately; they are not a completed campaign.

Before final reporting, programmatically count distinct completed plans and inspect their role/lifecycle events. State the artifact root and the first unmet acceptance condition if the run fails.

## User workflow preference

For campaign requests, do not pause for an intermediate summary or ask whether to continue after partial progress. Continue through retries and the requested multi-round acceptance path unless an actual external blocker prevents further execution.

## References

- `references/canonical-nested-delegation.md` — persisted-state canonicalization and policy sequencing for nested campaign delegation.
- `references/remote-execution-gate.md` — fail-closed acceptance gates, remote provenance requirements, timeout layering, and retry discipline for HPC/MCP-backed campaigns.
