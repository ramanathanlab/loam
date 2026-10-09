# Loam/Hermes Local Campaign Preflight

Use before a real nested campaign (`coordinator → hypothesis-validator → code-generator`), including non-biology integration tests.

## Isolated launch contract

Create one temporary campaign root containing all of:

- `loam.yaml` with a unique workflow ID and `state_dir` beneath that root;
- `hermes-home/config.yaml` with `plugins.enabled: [loam-hooks]` and `delegation.max_spawn_depth: 2`;
- a stable `LOAM_RUN_ID` shared by every coordinator turn;
- a `PATH` where the configured local runtime (for example `python3.11`) resolves.

Launch Hermes with `HERMES_HOME`, `LOAM_CONFIG_PATH`, `LOAM_STATE_DIR`, `LOAM_RUN_ID`, `HERMES_ENABLE_PROJECT_PLUGINS=1`, and `LOAM_ENABLED=1` set for the persistent coordinator process.

## Mandatory one-round gate

Do not begin a five-round campaign until round 1 proves nested delegation:

1. The coordinator's first tournament report is `completed` with all requested matches completed.
2. The selected validator start records show the intended orchestrator capability, not a leaf-only child.
3. Each validator delegates a tagged `code-generator` child.
4. Each code generator records a completed local execution with actual stdout.
5. The validator reports parseable `### Evidence Score` text: place the numeric value as the first bare token after the heading (for example `0.75 — rationale`, not backticked).
6. Exactly one reasoner completes and the campaign ledger contains one completed round.

Inspect `subagent-ledger.jsonl` and `hypothesis-orchestration.jsonl` after round 1. If a validator says delegation is unavailable, stop immediately: a configured `max_spawn_depth: 2` is insufficient unless the actual spawned validator is granted the orchestrator role/toolset. Do not count the round, manufacture execution output, or proceed to later rounds.

## Persistent-session input discipline

Submit each coordinator turn as one complete message. Do not send a follow-up while Hermes is still initializing or processing the same round unless using the session's explicit queue/steer mechanism; an interrupt can split campaign instructions and invalidate the run.

## Acceptance audit

After a successful preflight, run the remaining turns in the same coordinator session, preserving the same state directory and run ID. Require every reasoner brief to appear in the following coordinator goal before treating the five-round campaign as complete.
