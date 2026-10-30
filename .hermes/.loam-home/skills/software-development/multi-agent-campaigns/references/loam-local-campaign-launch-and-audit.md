# Loam local campaign: isolated launch and audit

Use this reference when the repository has the Loam local five-round campaign contract (`CAMPAIGN_TEST.md` or equivalent).

## Launch boundary

The campaign contract is local-execution scoped when its profile specifies:

```yaml
execution:
  mode: local
  runtime: python3.11
```

Do not describe it as an accepted remote-MCP campaign or mix its artifacts with remote provenance. A remote-MCP campaign requires a separate complete profile (`tool_name`, `endpoint`, `function_name`, `input_schema`, runtime) and separate acceptance criteria.

## Isolated invocation pattern

There is no launcher wrapper for this pattern. `run-hermes` was retired, and `run-campaign` is the wrong tool here: it assigns `HERMES_HOME`, `LOAM_STATE_DIR` and `LOAM_RUN_ID` itself and discards inherited values, which is exactly the isolation this acceptance run has to control. Invoke `hermes` directly from the repository root — that is where Hermes finds `.hermes/plugins` — with every variable set explicitly.

```bash
REPO=/path/to/loam
ROOT_CAMPAIGN="$(mktemp -d /tmp/loam-five-round.XXXXXX)"
RUN_ID="local-arithmetic-five-round-$(uuidgen | tr '[:upper:]' '[:lower:]')"
mkdir -p "$ROOT_CAMPAIGN"/{hermes-home,state}

# Create campaign-local loam.yaml and hermes-home/config.yaml first.
# config.yaml must enable loam-hooks and delegation.max_spawn_depth: 2.
cd "$REPO"
env \
  HERMES_HOME="$ROOT_CAMPAIGN/hermes-home" \
  LOAM_CONFIG_PATH="$ROOT_CAMPAIGN/loam.yaml" \
  LOAM_STATE_DIR="$ROOT_CAMPAIGN/state" \
  LOAM_RUN_ID="$RUN_ID" \
  LOAM_ENABLED=1 \
  HERMES_ENABLE_PROJECT_PLUGINS=1 \
  LOAM_PRE_LLM_EVERY=1 \
  hermes --cli
```

Preflight the effective config path without launching a campaign:

```bash
HERMES_HOME="$ROOT_CAMPAIGN/hermes-home" hermes config path
```

It must print `$ROOT_CAMPAIGN/hermes-home/config.yaml`.

## Persistent-session sequence

Run the coordinator in one persistent interactive session. Submit one bounded task per turn and wait for the prior plan's reasoner to finish before the next turn. Do not substitute `hermes chat -q` or another one-shot mode: it cannot reliably complete the validator-to-reasoner handoff.

For each accepted round, require task-level—not merely top-level—roles:

- coordinator batch validator task: `role: "orchestrator"`;
- validator's code-generator child: `role: "leaf"`;
- one tagged reasoner after all validator completions.

## Audit map

Audit only the isolated campaign `state/` directory:

| Artifact | What establishes |
| --- | --- |
| `run-manifest.json` | Stable run identity, workflow/profile, and execution contract; a changed identity is a hard failure. |
| `jnana-memory.json` | Persisted Jnana hypotheses. |
| `tournament-reports/tournament-*.json` | One completed coordinator tournament per accepted round; inspect `result.status`, requested/completed matches, turn, and workflow. |
| `subagent-ledger.jsonl` | Hermes lifecycle observation for validator and code-generator parent/child sessions, task goals, roles, and terminal statuses. Child summaries are audit material, not accepted evidence by themselves. |
| `hypothesis-orchestration.jsonl` | Tagged `agent_started`, `code_generator_completed`, `validator_completed` with parseable in-range evidence scores, and `reasoner_completed`. |
| `hypothesis-campaign.jsonl` | One `hypothesis_campaign_round_completed` per accepted round, including terminal validator results and reasoner synthesis. |

Never count a tournament report or partial child graph as a campaign round. Five accepted campaign-round records plus the predecessor-reasoner-brief steering chain are required for a five-round pass.
