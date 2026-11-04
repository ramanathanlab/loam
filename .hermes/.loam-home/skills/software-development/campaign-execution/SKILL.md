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
- Keep prerequisite/context evidence separate from hypothesis-bearing evidence. A verified target sequence or monomer fold does not support a binder, selectivity, stability, or affinity claim unless the discriminating comparison actually ran.
- Treat validator reports as unvalidated orchestration input until independent evidence review reconciles each claim against its authoritative source or hashed artifact. In particular, do not accept a child-reported batch ID, score, or artifact path without read-back verification.
- Match every claim to its own source. Do not cite a target's UniProt URL as provenance for a campaign execution failure or computed result; use the batch manifest, scheduler record, or output artifact that produced that claim.
- Distinguish blocked/inconclusive from refuted: absent candidates, failed tools, or an unexecuted gate produce no directional evidence and should retain an evidence score of 0.00 unless a completed experiment contradicts the hypothesis.

## Loam round sequencing

Use the canonical `loam_dispatch_round` for a pending plan rather than hand-composing validator or reviewer batches. After validators finish, dispatch the independent evidence-reviewer phase when the dispatcher requires it; after all review threads are terminal, dispatch exactly one reasoner. Do not dispatch a reasoner early, duplicate it, or report the round as synthesized before its returned brief is available.

Each round's report should state: what was actually executed, which artifacts were independently verified, the evidence score, the first unmet acceptance criterion, and the single next discriminating action. A successful prerequisite phase is not a successful campaign round when the campaign's acceptance bar requires a complete funnel.

## Verification

After code changes, run focused lifecycle/policy tests, syntax/type checks appropriate to touched files, and an isolated ad-hoc probe of the changed runtime integration. Label focused/ad-hoc results accurately; they are not a completed campaign.

Before final reporting, programmatically count distinct completed plans and inspect their role/lifecycle events. State the artifact root and the first unmet acceptance condition if the run fails.

## User workflow preference

For campaign requests, do not pause for an intermediate summary or ask whether to continue after partial progress. Continue through retries and the requested multi-round acceptance path unless an actual external blocker prevents further execution.

## Dispatcher no-op gate

A campaign-start request is not proof that a dispatch is pending. If the authoritative campaign dispatcher returns `nothing_to_dispatch`, do not fabricate a validator batch, substitute direct work, or repeatedly reissue the same dispatch blindly. Report the exact fail-closed state: either children are already running or the tournament selected nothing. If campaign-state inspection is available, inspect it before retrying; otherwise wait for a new pending plan or terminal child state. A repeated user request does not override the dispatcher’s prohibition on substitute work.

## References

- `references/canonical-nested-delegation.md` — persisted-state canonicalization and policy sequencing for nested campaign delegation.
- `references/remote-execution-gate.md` — fail-closed acceptance gates, remote provenance requirements, timeout layering, and retry discipline for HPC/MCP-backed campaigns.
