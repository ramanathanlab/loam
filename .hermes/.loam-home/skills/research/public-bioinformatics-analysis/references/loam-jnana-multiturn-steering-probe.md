# Live multi-turn Loam/Jnana steering probe

Use this recipe to validate advisory steering behavior without mutating an active research workflow.

## Isolated campaign boundary

Create a unique temporary campaign root and place under it:

- Jnana `state_file`
- tournament `reports_dir`
- Loam hypothesis/evidence/steering state directory
- a distinct workflow ID and session ID

Do not copy generated hypotheses, reports, evidence, or credit into a production workflow.

## Probe sequence

1. On turn 1, provide a biologically bounded `user_message`, seed at least three hypotheses, and run one pairwise match.
2. Explicitly accept selected hypotheses into the *sandbox* ledger only. Acceptance is not an empirical endorsement.
3. For Elo-only steering, accept no evidence. The expected evidence rating remains the prior (1200) and the recommended action is `seek_independent_source_verification`.
4. Project steering from the accepted sandbox hypotheses, accepted evidence, and latest immutable tournament report.
5. On each later turn, inject the prior steering snapshot and run exactly one further live tournament match.
6. Re-project after the tournament so the following turn receives updated Elo-derived steering.

## Ordering invariant

`pre-llm.py` assembles recall context before tournament context. Therefore it exposes:

1. the latest persisted steering projection, then
2. the newly completed tournament result.

The steering portion is necessarily stale by one tournament iteration unless another lifecycle action writes a new projection between turns.

## Repeated-turn invariant

A normal pre-LLM payload can omit `user_message`. A nonempty Jnana state must still execute ranking:

- when fewer than two hypotheses exist, missing `user_message` yields `generation.status = "skipped"` with `reason = "missing_research_goal"`;
- when two or more hypotheses exist, missing `user_message` must yield `generation.status = "not_needed"` and continue tournament ranking.

The implementation must load/count state before requiring the generation goal.

## Verification

For each immutable report, assert:

- schema version is `loam_jnana_tournament_v1`;
- the isolated workflow ID matches;
- status is `completed`;
- requested and completed match counts equal the configured one-match budget;
- hypothesis count stays stable after seeding;
- only the initial report records generated hypotheses;
- later reports record `generation.status = "not_needed"`.

Also inspect Jnana state for the persistent hypothesis count and cumulative completed matches. Confirm each later composite context contains both `Hypothesis steering state` and that turn's `Trigger: Loam turn N` tournament update.

When there is no canonical integration target, create and run a temporary `hermes-verify-*.py` script using `tempfile`; ensure it exercises the no-`user_message`, existing-hypotheses path and removes itself afterward. Report this as ad-hoc verification, distinct from suite results.
