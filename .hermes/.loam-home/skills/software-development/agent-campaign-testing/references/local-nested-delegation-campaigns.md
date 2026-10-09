# Local nested-delegation campaign testing

Use this reference when a campaign must prove coordinator → validator → code-generator lifecycle behavior in an isolated Hermes/Loam run.

## Fresh isolated run contract

- Create one fresh temporary root containing `hermes-home/`, `state/`, `bin/`, and `loam.yaml`.
- Keep a stable `LOAM_RUN_ID` for all rounds in that one root.
- Set `HERMES_HOME` to the campaign-local home before launching the repo launcher. The launcher must preserve a caller-supplied `HERMES_HOME`; otherwise campaign-local delegation settings are silently lost.
- Configure `delegation.max_spawn_depth: 2`; use validator task-level `role: "orchestrator"` and code-generator `role: "leaf"`.
- Never reuse state from a stopped campaign. A partial round is diagnostic evidence only and receives no completion credit.

## Batch delegation invariants

For `delegate_task(tasks=[...])`, the child role is taken from each task object. A top-level role or a natural-language instruction is insufficient.

Validator batch shape:

```python
{
  "goal": "... [loam-agent-role=hypothesis-validator] [loam-plan-id=...] [loam-hypothesis-id=...]",
  "context": "<complete injected contract>",
  "role": "orchestrator",
}
```

The complete contract should contain:

- an explicit version marker (for example `loam_validator_contract: v1`) so policy can reject abbreviated/reconstructed tasks;
- the exact tagged leaf code-generator goal;
- a complete execution contract;
- required validator report headings and a parseable evidence score requirement.

## Local exact-arithmetic probe

For arithmetic campaign tests, make the code-generator execute—not merely describe—the following kind of standard-library check:

```python
from fractions import Fraction
values = [Fraction(n) for n in (1, 2, 3, 4)]
mean = sum(values) / len(values)
threshold = Fraction("2.4")
print({
    "sum": str(sum(values)),
    "count": len(values),
    "mean": str(mean),
    "threshold": str(threshold),
    "exceeds": mean > threshold,
})
```

Expected result: sum `10`, count `4`, mean `5/2`, threshold `12/5`, and `exceeds=True`.

## Fail-closed policy checks

Before running five rounds, test policy behavior with a deterministic fixture:

1. Full injected validator batch is allowed.
2. Untagged substitute validator batch is blocked.
3. Validator batch without per-task orchestrator role is blocked.
4. Validator batch with abbreviated/missing versioned context is blocked.
5. Validator can delegate only a plan-tagged code-generator child.
6. Code-generator can execute locally; validator and reasoner cannot execute experimental work directly.
7. Evidence-score parsing accepts bare numbers and a single Markdown code span, but rejects absent/out-of-range values.

## Live-run inspection order

After round 1, inspect campaign-local files only:

1. `subagent-ledger.jsonl`: validator starts must be `orchestrator`; code-generator starts must be `leaf`; both must terminate normally.
2. `hypothesis-orchestration.jsonl`: matching validator completion records must have non-null in-range scores; code-generator completion must include execution output.
3. `hypothesis-campaign.jsonl`: only after all validators are terminal should exactly one reasoner complete and write the round-completed event.
4. `tournament-reports/`: coordinator reports must be complete and child sessions must not create tournament reports.

Stop immediately at the first missing lifecycle or invalid result. Do not treat model summaries, Elo, or repeated calculations as accepted empirical credit.

## Reusable verification pattern

Use a temporary `hermes-verify-*.py` script under the OS temporary directory to import the orchestration hook, build a tiny fixture with two hypotheses and one completed tournament report, render the directive, and assert:

- two validator tasks;
- task-level orchestrator roles;
- versioned full contracts;
- leaf code-generator requirements containing `Fraction`;
- policy allows the exact rendered payload and blocks a context-truncated variant.

Clean up the verifier after execution and report it as ad-hoc verification, not suite green.
