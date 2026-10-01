# Loam/Jnana hypothesis tournament: generation then ranking

Use this only for an opt-in Loam research profile. It is an advisory discovery layer, not evidence validation.

## Required state flow

1. Hermes invokes Loam `pre_llm.py` with the original `user_message`.
2. `hypothesis_tournament.context_for()` locks the workflow-scoped Jnana state.
3. If fewer than two hypotheses exist, instantiate Jnana `CoScientist`, call `set_research_goal(user_message)`, call `generate_hypotheses(count=initial_hypotheses)`, wait, and persist state.
4. Run Jnana's `run_incremental_tournament()` with the configured match count and top-k.
5. Write an immutable Loam report containing `result.generation`, `hypothesis_count`, and `matches_completed`.
6. Inject only compact Elo-ranked summaries into Hermes context.

## Completion-budget pitfall

Jnana's structured generation schema includes a mechanism, novelty items, predictions, explanation, and strategy. It can exceed a compact-ranking JSON budget. A 1,200-token cap produced `finish_reason: length` with empty JSON content during live testing. Keep generation `json_max_tokens` at least 4,096 while allowing compact tournament ranking to use the profile-configured lower value.

## Verification

After a real run, inspect the newest immutable report and assert:

- `result.status == "completed"`
- `result.generation.generated_hypotheses >= 2`
- `result.hypothesis_count >= 2`
- `result.matches_completed` equals the requested completed tournament work

Also confirm Jnana state contains the matching number of hypotheses and tournament matches. Use a temporary `hermes-verify-*.py` script outside the repository for this check when no canonical test target exists.

## Evidence boundary

Generated hypotheses and Elo ordering are model-mediated prioritization, not empirical support. Do not create evidence or credit records from tournament output. Source-backed evidence must pass explicit operator reconciliation before Loam credit assignment.
