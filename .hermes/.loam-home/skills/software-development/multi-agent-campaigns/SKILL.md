---
name: multi-agent-campaigns
description: Run and debug isolated multi-round agent campaigns that require a persistent coordinator, nested delegation, durable lifecycle evidence, and per-round acceptance checks.
---

# Multi-Agent Campaigns

Use this for end-to-end campaigns where a coordinator must repeatedly delegate specialist agents, collect their terminal outputs, synthesize a round, and feed the synthesis into a subsequent round. Treat the campaign specification as an acceptance contract, not a loose prompt.

## Core workflow

1. Read the campaign specification and enumerate its required artifacts, exact lifecycle roles, stable identifiers, and stop-on-first-failure rule.
2. Run in a new isolated directory containing the profile/config, a dedicated Hermes home, and the only runtime state directory. Use one stable run ID for the complete campaign.
3. Verify the launcher preserves the caller-provided `HERMES_HOME`; wrappers must not silently replace the isolated config.
4. Verify nested-delegation configuration before execution: the coordinator must spawn validators as orchestrators and validators must spawn code generators as leaves.
5. Start one persistent coordinator session. Do not use one-shot invocation; do not interrupt the initial turn by sending a second message before it has accepted the first.
6. For each round, inspect lifecycle artifacts before proceeding: selected validators, code-generator starts/stops, validator terminal reports and scores, exactly one reasoner, then round synthesis.
7. Only start the next coordinator turn after the reasoner has completed. Confirm the previous reasoner brief appears in the next coordinator prompt or persisted steering material.
8. Stop at the first unmet criterion. Preserve the isolated state for diagnosis; never fabricate lifecycle events, evidence, scores, or synthesis.

## Enforce contracts in code, not prose alone

Model instructions alone are insufficient for mandatory delegation topology.

- Render a ready-to-run `delegate_task` payload for the coordinator. Each batch task must include its own `role: "orchestrator"`; a top-level role does not grant nested delegation to batch children.
- Put lifecycle identity markers in the actual child `goal`, not merely in surrounding text or context.
- Include full selected work item, execution contract, required child role, and expected child markers in each task context.
- Use a pre-tool policy to fail closed when a pending plan attempts untagged substitute work or direct coordinator execution.
- Parse both single-task and batch (`tasks=[...]`) delegation forms. A validator may correctly create its code-generator using a batch even if there is only one child.
- Require all selected hypothesis IDs exactly once before accepting a validator dispatch; reject duplicates, omissions, and extra validator work.

## Isolation preflight

Before the live run, verify:

```text
HERMES_HOME points at the campaign-local directory
LOAM_CONFIG_PATH points at campaign-local loam.yaml
LOAM_STATE_DIR points at campaign-local state/
LOAM_RUN_ID is stable for all rounds
custom provider endpoint responds and supports the intended request shape
python3.11 is available to code-generator children
```

Do not rely on a launcher setting environment variables correctly without checking its effective config path. A wrapper that replaces `HERMES_HOME` can make nested delegation appear configured while the coordinator actually uses defaults.

## Lifecycle audit

For every round, derive pass/fail only from persisted campaign state:

- tournament report is completed and all requested matches completed;
- validator starts match all selected candidates;
- validator children are orchestrators;
- each validator has exactly one code-generator child with leaf role;
- code generators terminate completed and include the required local execution observation;
- validator evidence scores parse, are non-null, and are in range;
- exactly one reasoner terminates completed;
- campaign-round record includes terminal validator results and reasoner synthesis;
- next round has evidence that the prior synthesis informed its input.

Keep advisory hypothesis rankings distinct from accepted empirical evidence or credit.

## Troubleshooting order

1. Confirm effective isolated config, especially `HERMES_HOME` and nested delegation depth.
2. Inspect actual child roles in the lifecycle ledger, not the coordinator's claimed tool arguments.
3. Inspect pre-tool blocks and hook output before changing prompts.
4. Check whether policy code supports the exact tool-call JSON shape the agent emitted (`goal` versus `tasks[*].goal`).
5. Only then refine directive text. Add a regression test for each discovered shape mismatch.

## Verification

- Add focused tests first for directive rendering and policy acceptance/rejection paths.
- Run the focused suite after each policy change.
- Use a fresh campaign directory for every live retry; partial rounds must never be resumed as if complete.
- Report focused/ad-hoc verification separately from a full campaign pass.

See `references/hermes-loam-campaign-pitfalls.md` for concrete Hermes/Loam failure signatures and checks.

For the repository-local five-round **local-execution** Loam contract, use `references/loam-local-campaign-launch-and-audit.md`. It records the isolated launcher invocation, persistent-session gate, artifact-to-acceptance mapping, and the boundary between local and remote-MCP campaign profiles.
