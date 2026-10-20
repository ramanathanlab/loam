# Loam/Jnana live-orchestration with mocked execution

## Correct test boundary

For an infrastructure test of a scientific Loam campaign, keep these components real:

```text
local model → Jnana hypothesis generation → tournament/ranking
→ coordinator steering → validator/reasoner lifecycle
```

Mock only the specific downstream execution tool used by a code-generator, for example:

```yaml
mode: remote_mcp
tool_name: mock_hpc_execution_run_code
endpoint: mock://empty
function_name: run_code
input_schema: mock-empty-v1
```

An empty mock result is useful for testing tool-policy routing and lifecycle handling, but is not scientific evidence.

## Acceptance and failure classification

For a request for N hypotheses, record:

- exact model ID and provider endpoint;
- Jnana repository path, branch, and commit;
- requested count;
- persisted count;
- unique/non-empty hypothesis IDs and content;
- tournament status and completed/requested matches;
- exact failure class if incomplete.

Do not fabricate or pad missing hypotheses. Distinguish:

- provider biological-safety/policy rejection (`400`, often `bio_policy`);
- valid HTTP response but invalid JSON/schema for Jnana;
- transport/authentication failures;
- actual capacity/resource limits.

A provider response with HTTP 200 can still fail generation if the model emits prose or fenced Markdown instead of the JSON schema expected by Jnana. This was observed with a Haiku model: zero persisted hypotheses despite successful HTTP responses.

## Provenance example from one investigation

The Jnana checkout used for generation was on branch `laya`, commit `dfe146b7`. The Luna model persisted 18/100 before repeated `bio_policy` responses; this was not evidence of an 18-hypothesis capacity limit. A Haiku trial persisted 0/100 because responses were not parseable JSON. These are examples of reporting shape, not permanent model capabilities.

## Anti-patterns

- Do not call synthetic lifecycle hooks and report that as a real Jnana campaign.
- Do not replace real hypotheses/tournament results with mock records just to reach an acceptance count.
- Do not interpret a configured execution endpoint or mock tool-policy pass as evidence that the scientific workload ran.
- Do not switch models or rewrite the prompt without preserving the preceding failure class and provenance.
