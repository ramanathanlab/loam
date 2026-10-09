# Hermes/Loam campaign pitfalls

## Isolated home can be silently lost

A repository launcher may reset `HERMES_HOME` to a project default. This makes an isolated campaign appear to use its temporary config while Hermes actually reads the default config, including its default delegation depth.

Preflight proof:

```bash
HERMES_HOME="$campaign/hermes-home" .loam/run-hermes config path
# must print: $campaign/hermes-home/config.yaml
```

Preserve caller configuration in wrappers:

```bash
export HERMES_HOME="${HERMES_HOME:-$ROOT/.hermes/.loam-home}"
```

## Batch delegation roles are per task

For Hermes `delegate_task(tasks=[...])`, each validator object must carry:

```json
{"goal":"...", "context":"...", "role":"orchestrator"}
```

A top-level role is not sufficient for batch children. Verify actual `agent_role` in the subagent ledger, not the coordinator's narrative.

## Policy must parse actual tool-call shapes

Validators may create one code-generator via `tasks=[{"goal": ...}]` rather than a top-level `goal`. Any policy that only validates `arguments.goal` rejects a legitimate nested dispatch. Validate `tasks[*].goal` and single-goal forms.

## Pre-LLM injection timing

Hermes calls `pre_llm_call` after the user turn enters the loop. If a prompt is submitted before the hook context is visible to the model, it may perform direct work or report that it lacks injected data. Inspect the session's actual plugin context and persisted hook output before attributing failure to model compliance.

## Failure interpretation

Do not count any round when:

- a coordinator computes directly instead of dispatching;
- a validator is a leaf when nesting is required;
- a validator cannot create its tagged code generator;
- a code generator is blocked/interrupted;
- validator score/synthesis/lifecycle events are absent.

Retain the isolated state directory for diagnosis, then create a new directory for the next live attempt.
