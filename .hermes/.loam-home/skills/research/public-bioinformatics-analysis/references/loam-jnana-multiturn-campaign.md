# Loam/Jnana Multi-turn Campaign Notes

## Campaign stages

`completed Jnana tournament → persistent top-3 resolution → completed exploration for all 3 → bounded synthesis → next-turn injection`

Keep each stage independently inspectable. Tournament rankings are advisory; raw exploration output is not evidence.

## Operational credit boundary

When a sandbox needs automatic completion accounting without human source review, use a dedicated append-only operational event such as:

- `event: unverified_exploration_credit_assigned`
- `epistemic_status: unverified`
- `eligible_for_evidence_rating: false`
- `eligible_for_contribution_credit: false`

Do not write accepted-evidence or contribution-credit events from a model-generated summary.

## Required guards

- Reject incomplete matches (`matches_completed != matches_requested`).
- Reject rankings whose IDs are missing from persistent Jnana state.
- Reject a partial top-three pass before synthesis or credit persistence.
- Reject an empty synthesis before appending any credit records.
- Bound injected prior synthesis and exploration excerpts; prompt crowding can cause an OpenAI-compatible structured endpoint to emit an empty response.

## Verification design

Use two evidence tiers and label them accurately:

1. A deterministic fixture with two rounds proves record counts, eligibility flags, fail-closed behavior, and that round 1 synthesis appears in round 2 tournament input.
2. A live Jnana probe proves the configured endpoint can persist hypotheses and immutable tournament reports. An empty exploration completion is a runtime failure, not an empty biological result; record the blocker and do not mint credit.

Use a `tempfile`-created `hermes-verify-*` script outside the repository for focused ad-hoc verification, clean it up, and do not call it canonical suite coverage.
