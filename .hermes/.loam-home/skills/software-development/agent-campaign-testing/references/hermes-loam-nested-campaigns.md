# Hermes/Loam nested local campaign notes

Use this when an isolated Loam campaign requires `coordinator → validator → code-generator` and repeated model attempts rewrite or omit structured delegation arguments.

## Isolate every attempt

- Use one fresh temporary root per acceptance attempt, containing `hermes-home/`, `state/`, `bin/`, and `loam.yaml`.
- Keep a stable `LOAM_RUN_ID` within that one root; never reuse a stopped root as an acceptance run.
- Export `HERMES_HOME` before invoking `.loam/run-hermes`. The launcher must preserve caller-supplied `HERMES_HOME`; otherwise campaign-local `delegation.max_spawn_depth: 2` is silently lost.
- Supply a Python 3.11 shim when the default `python3` is older and the campaign contract requires `python3.11`.

## Hermes batch role contract

For `delegate_task(tasks=[...])`, each validator task object—not merely the surrounding natural-language prompt—must contain:

```json
{"role": "orchestrator"}
```

A missing task-level role becomes `leaf`; leaf children cannot make the required nested code-generator delegation.

## Fail-closed lifecycle checks

Before a round may count, inspect only its temporary `state/` artifacts:

1. validators have `agent_role: orchestrator` in `subagent-ledger.jsonl`;
2. each validator has one code-generator child with `agent_role: leaf`;
3. code-generator stop records are completed and include observed local execution;
4. every validator record has a non-null, in-range score;
5. exactly one reasoner completes for the plan;
6. `hypothesis_campaign_round_completed` exists before the next coordinator turn.

Stop at the first unmet condition. Do not treat plausible validator prose, direct coordinator arithmetic, Elo, or missing lifecycle records as evidence or a completed round.

## Canonicalize structured delegation when the model copies it unreliably

Natural-language directives and pasted JSON are insufficient when the coordinator/validator rewrites task objects. Use Hermes `tool_request` middleware to replace the effective `delegate_task` arguments from persisted campaign state before pre-tool policy and dispatch.

Reconstruct three stages mechanically:

- coordinator: selected Jnana hypotheses → tagged validator batch with each task `role: orchestrator`;
- validator: its persisted plan/hypothesis markers → exactly one tagged code-generator `role: leaf` task;
- coordinator after all validators complete: persisted validator results → exactly one tagged reasoner task.

Keep pre-tool policy aligned with lifecycle phase: reject untagged validator batches only while validators are pending. Once all selected validators are terminal, permit the canonical reasoner delegation; otherwise a stale coordinator guard rejects the legitimate reasoner as an "untagged validator" batch.

## Exact local arithmetic contract

For the bounded mean test, code-generator context should direct Python 3.11 to use `fractions.Fraction`, print exact sum/count/mean/threshold/comparison output, and return stdout plus execution status. Validator reports must include `### Evidence Score`; parser should tolerate a numeric Markdown code span but reject malformed or out-of-range values.

## Verification pattern

- Unit-test state reconstruction and both allow/block phases of policy.
- In an isolated ad-hoc verifier, confirm middleware replaces arbitrary delegate arguments with canonical coordinator and validator task payloads.
- Run a live round only after those checks. Verify nested lifecycle at round 1 before attempting rounds 2–5.
