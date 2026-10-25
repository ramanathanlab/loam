# Jnana model-compatibility benchmarking

Use this reference when the question is whether a model can successfully drive Jnana generation, rather than whether it is scientifically the best model.

## Required protocol

1. Define the candidate ordering explicitly. A user-provided ordinal such as “lower than or equal to model X” is not a public benchmark claim; record it as the test ordering.
2. Run candidates sequentially, not concurrently, against one local endpoint.
3. Request the same hypothesis count for each candidate and stop at the first candidate meeting the requested persisted count.
4. Use a separate state file per model and a fresh Jnana object/state namespace.
5. Capture model ID, Jnana repository branch and commit, prompt class, requested count, persisted count, schema-valid count, unique count, elapsed time, state path, and error class.
6. Verify the on-disk schema before interpreting counts. Jnana may persist dictionaries such as `{hypothesis_id, content}` rather than strings.

## Acceptance

A model passes only if the requested number of records is persisted, each record has a non-empty identifier and content, identifiers are unique, and the generation path reports no provider/parser failure. HTTP 200, a non-empty text answer, or a completed worker task is not sufficient.

## Failure taxonomy

- `policy_rejection`: provider rejects the scientific prompt; do not downgrade to synthetic hypotheses.
- `invalid_json`: provider returns prose, Markdown, or fenced JSON where Jnana expects strict JSON.
- `storage_failure`: Jnana cannot persist or reload generated records.
- `partial_generation`: fewer than requested records persisted; this is not evidence of a model or Jnana capacity ceiling.
- `success`: requested count and schema/uniqueness gates pass.

## Prompt interpretation

A sanitized abstract computational prompt can isolate protocol compatibility from biological-policy filtering. This benchmark result must not be presented as proof that the model will handle the full scientific prompt. Conversely, a failure on the full prompt must not be described as a Jnana capacity limit.

## Example benchmark outcome

A sequential benchmark stopped at `argo:gpt-5.6-sol` after it persisted 50/50 unique hypothesis dictionaries. The checker initially expected strings and incorrectly reported zero valid records; re-checking the actual persisted schema corrected the result. Always validate against the observed Jnana schema before reporting.
