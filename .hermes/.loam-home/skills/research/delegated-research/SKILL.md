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

## Reporting conventions

- Label source-derived fields as reported metadata where applicable.
- Preserve explicit caveats such as an “apo” title coexisting with a deposited metal ion.
- For direct operational questions, lead with the answer; do not substitute process explanation or an unverified conditional.

## Pitfalls

- **Do not infer ledger status merely because a delegation completed.** Inspect the recorded lifecycle data when verification is requested.
- **Do not over-answer.** “Did it write?” calls for a verified yes/no, not an attribution narrative.
- **Do not treat child outputs as final scientific truth.** Summarize their reported findings and source URLs, with material uncertainties retained.
