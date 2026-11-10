# Hermes gateway parent-agent fix

**Historical:** the active installation is now pinned to Hermes 0.20.6
(`v2026.8.27`). This patch targets the newer development build and must not be
applied to 0.20.6. Its patched checkout is retained under
`software/hermes-agent-newer-backup`. See `hermes-version.json` for the active pin.

The installed Hermes base is `5c7381a61e` (see the patch's deployment context in
`JUPITER.md`). Loam's `loam_dispatch_round` calls `ctx.dispatch_tool("delegate_task", ...)`.
Gateway plugin managers have no `_cli_ref`, so unpatched Hermes loses the parent
and returns `delegate_task requires a parent agent context`.

`hermes-parent-agent.patch` binds the executing agent in a ContextVar around both
sequential and concurrent registry tool calls. Plugin dispatch prefers that
binding, then its legacy CLI fallback; explicit `parent_agent` arguments win.
Bindings restore on exceptions and nesting. Native delegation remains unchanged,
including background top-level execution and joined orchestrator children.

Apply from the workspace with:

```bash
bash loam/scripts/apply_hermes_jupiter_patch.sh
```

The script checks for an already-applied patch and refuses incompatible source.
It does not restart services. Start a fresh campaign gateway after applying;
MCP and the Globus endpoint do not need restarting. Do not update Hermes without
revalidating or reapplying this patch against the new source.

Tests live in the patch at `tests/agent/test_plugin_tool_parent_context.py` and
run with Hermes' standard `scripts/run_tests.sh`. They cover overlapping sessions
on sequential and concurrent execution paths, nested exception cleanup, explicit
parent override, and a discovered plugin using native delegation with no CLI.
The isolation regressions fail on unmodified Hermes and pass with the patch.

The campaign Hermes config also sets `plugins.hook_callback_timeout: 600` to
allow the 480-second tournament hook. That setting is independent of the fix.

Validation on Jupiter: 8 focused tests passed, and the two overlap regressions
failed on the unmodified base. A live discovered-plugin delegation using
`argo:gpt-5.6-sol` with no CLI reference completed one child API call and returned
`LOAM_PARENT_CONTEXT_OK`. Receipt:
`software/hermes-agent/state/parent-context-live/result.json` in the workspace.
Patch application was checked against a clean base and repeated to verify
idempotence. These checks do not establish a completed EGFR campaign.
