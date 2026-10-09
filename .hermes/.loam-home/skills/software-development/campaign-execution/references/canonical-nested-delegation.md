# Canonical nested delegation from campaign state

## Problem

A coordinator model may rewrite or truncate an injected `delegate_task` payload. This can drop per-task orchestrator roles, markers, nested-child requirements, or the reasoner context. Prompting it repeatedly is not sufficient for a strict campaign contract.

## Pattern

Install narrowly scoped `tool_request` middleware for an opted-in campaign workflow. On `delegate_task`, derive the authoritative task arguments from immutable/persisted campaign state rather than model-supplied arguments:

1. If the active plan has not started validators, rebuild the complete selected-validator batch. Every validator task must carry task-level `role: "orchestrator"` and the full contract context.
2. For a validator session, rebuild its required code-generator task as `role: "leaf"`, including exact execution instructions and stable role/plan/hypothesis markers.
3. When every validator in the active plan has completed, rebuild exactly one reasoner task from the validator completion records.

## Policy sequencing

Keep policy checks fail-closed, but distinguish phases:

- Before all validators complete: reject substitute/untagged coordinator work and malformed validator dispatch.
- After all validators complete: allow the reasoner dispatch; do not keep applying the validator-only block.
- After a reasoner `agent_started` record exists: block all additional parent tool calls or at least duplicate reasoner delegation until the terminal reasoner event arrives.
- Count a round only on the terminal reasoner record that writes a campaign-round-completed artifact.

## Verification checklist

- Unit test canonical payload reconstruction for all three phases.
- Unit test malformed validator calls remain blocked before completion.
- Unit test a reasoner call is allowed only after every validator is terminal.
- In an isolated live root, inspect append-only lifecycle records for coordinator → orchestrator validator → leaf code-generator and one reasoner.
- Treat model text as untrusted; inspect actual child role, local execution output, evidence score, and completed-round records.
