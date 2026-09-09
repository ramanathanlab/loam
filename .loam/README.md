# Loam harness integration

`.loam/` is Loam's harness-neutral integration directory. It contains the
canonical lifecycle-event contract and its adapters; it is not Hermes-specific.

## Layout

- `manifest.yaml` — portable event-to-script mapping and JSON wire contract.
- `hooks/` — executable hook implementations. Each receives one JSON object on
  standard input and may emit one JSON object on standard output.
- `adapters/hermes/plugin/` — thin Hermes plugin that converts Hermes callback
  kwargs into the Loam payload and executes the canonical scripts.
- `run-hermes` — launches Hermes with project-plugin discovery enabled.

## Hermes setup (one-time, per user)

Hermes discovers project plugins only at `.hermes/plugins/`, so this repository
contains a tiny tracked discovery shim at `.hermes/plugins/loam-hooks/`. The
actual adapter lives here under `.loam/`.

To activate the adapter only while working in this repository:

```bash
.loam/run-hermes
```

Hermes additionally requires an explicit plugin allow-list. Add `loam-hooks`
to the `plugins.enabled` list in the Hermes configuration used by this
invocation. With the normal Hermes home, that is `~/.hermes/config.yaml`:

```yaml
plugins:
  enabled:
    - loam-hooks
```

This allow-list entry alone does not activate Loam anywhere else: discovery is
limited to this repository and only when `HERMES_ENABLE_PROJECT_PLUGINS=1`,
which `run-hermes` sets.

To avoid any global Hermes configuration at all, set `HERMES_HOME` to a
Loam-specific Hermes home before invoking `run-hermes` and place the same
`plugins.enabled` entry in `$HERMES_HOME/config.yaml`. That isolated home needs
its own model/credential configuration.

## Hook contract

See `manifest.yaml`. Current hook scripts are deliberately no-ops, so enabling
the adapter is safe. Replace their bodies with Loam policy as it is designed.

`pre-llm.py` runs every **N** turns, independently for each Hermes session.
The interval defaults to 5 and is configured at launch:

```bash
LOAM_PRE_LLM_EVERY=3 .loam/run-hermes
```

The adapter supplies `loam_turn` and `loam_interval` in that script's JSON
payload. It runs on turns N, 2N, 3N, and so on; set the interval to `1` to run
on every turn. The count resets when the Hermes process restarts.

- `pre-tool.py` may return `{"action":"block","message":"..."}` to veto a
  Hermes tool call.
- `pre-llm.py` may return `{"context":"..."}` to add turn-local context.
- Observer hooks (`post-tool.py`, `subagent-start.py`, `subagent-stop.py`) have
  no behavior-changing return value in Hermes.

All hooks should fail closed with respect to themselves: malformed output or a
non-zero exit means the adapter logs the failure and lets Hermes proceed. Do
not put secrets in stdout; payloads can contain tool arguments and subagent
summaries.
