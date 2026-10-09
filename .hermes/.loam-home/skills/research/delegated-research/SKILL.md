---
name: delegated-research
description: Run controlled multi-agent web-research batches and report their provenance, lifecycle state, and findings precisely.
---

# Delegated Research

Use for research tasks that divide into independent source-verification workstreams, especially when the user specifies agent count, role, source restrictions, or lifecycle-observability requirements.

## Procedure

1. Translate each workstream into a self-contained child brief: exact deliverables, permitted sources/tools, required source URLs, and stated uncertainty boundaries.
2. When a fixed parallel count is requested, dispatch exactly that many independent `leaf` agents in one batch. Do not add follow-up agents or use non-leaf roles.
3. Keep parent work separate from child work. Do not create, edit, upload, or publish files if prohibited. Do not make scientific claims stronger than the cited deposited/curated metadata supports.
4. After the batch completes, provide the requested compact parent synthesis. Identify each child's completion status and its reported identifiers/URLs.
5. If asked whether a lifecycle ledger was written, verify the actual ledger/state record before answering. State a direct yes/no first, then give the minimum relevant evidence (for example, count and event/status types). Do not answer a narrower question about *who* wrote it when the user asked only whether it was written.

## Nested Delegation and Role-Gated Workflows

For campaigns that require a parent → validator → executor chain, validate the *effective* delegation capability before dispatching the validator batch. A declared configuration value is not sufficient: confirm that validator children receive `delegate_task` and that the configured spawn depth permits the executor layer.

1. Dispatch validators in the requested batch with their required role/plan/hypothesis markers and state the exact executor contract in each brief.
2. Require each validator to delegate its tagged executor child before interpreting results. The validator must not substitute parent-side execution or plausible arithmetic/reasoning for the child’s observed output.
3. Treat a terminal validator that could not create its executor child as blocked/inconclusive, not as empirical support or contradiction. Preserve the reported score/status exactly; do not manufacture evidence from the hypothesis or ranking.
4. Only after every validator is terminal, dispatch the single requested synthesis/reasoner agent. Give it the structured reports and distinguish comparative ranking signals (such as Elo) from empirical evidence scores.
5. Do not state a campaign conclusion as execution-backed until lifecycle records show the required nested executor completions and their actual outputs.

## Reporting conventions

- Label source-derived fields as reported metadata where applicable.
- Preserve explicit caveats such as an “apo” title coexisting with a deposited metal ion.
- For direct operational questions, lead with the answer; do not substitute process explanation or an unverified conditional.

## Pitfalls

- **Do not infer ledger status merely because a delegation completed.** Inspect the recorded lifecycle data when verification is requested.
- **Do not over-answer.** “Did it write?” calls for a verified yes/no, not an attribution narrative.
- **Do not treat child outputs as final scientific truth.** Summarize their reported findings and source URLs, with material uncertainties retained.
