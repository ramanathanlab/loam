# Loam/Hermes nested-delegation notes

## Batch role semantics

Hermes treats `role` as a per-task field for `delegate_task(tasks=[...])`. A batch-level role does not reliably make each child an orchestrator. A validator that must launch a code-generator needs:

```json
{
  "goal": "... [loam-agent-role=hypothesis-validator] [loam-plan-id=...] ...",
  "context": "complete hypothesis plus execution contract",
  "role": "orchestrator"
}
```

Code-generator children should be tagged and use `role: "leaf"`.

## Isolated launcher pitfall

For a campaign-specific `HERMES_HOME`, the project launcher must preserve a caller-provided value:

```bash
export HERMES_HOME="${HERMES_HOME:-$ROOT/.hermes/.loam-home}"
```

An unconditional assignment to the repository default home silently defeats isolated config, including `delegation.max_spawn_depth: 2`.

## Minimum preflight

- Stable `LOAM_RUN_ID` and one temporary `LOAM_STATE_DIR` for all campaign turns.
- Persistent coordinator, not one-shot invocation.
- Isolated Hermes config enables `loam-hooks` and `delegation.max_spawn_depth: 2`.
- Confirm `HERMES_HOME=<temp> hermes config path` resolves to `<temp>/config.yaml` before launch; invoke `hermes` directly rather than through a launcher that would reset the home.
- Ensure the local endpoint/model works with the campaign request shape.

## Fail-closed lifecycle checks

After each coordinator turn, inspect only campaign state. Require completed tournament, validator starts and terminal records, code-generator child lifecycle records, parsed bounded evidence scores, and exactly one reasoner completion. Stop at the first failed criterion; do not count a partial round.

## Evidence boundary

A child summary is audit material, not accepted evidence. Jnana Elo and reasoner synthesis remain advisory until a separate reconciliation/validation path accepts source-backed evidence.
