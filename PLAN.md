# loam — hypothesis-driven scientific campaigns on pydantic-ai

## Why

Loam's science is sound; its plumbing is the problem. ~9k lines of hooks exist
mostly to coerce Hermes into behaving: an LLM orchestrator that must be nudged
to call `loam_dispatch_round`, tool-name policing, delegation refusal/repair,
`run_code` budgets, dead-child replacement, and parsing fenced JSON out of
prose reports. Each fix spawned the next failure mode.

loam keeps the *contracts*: Jnana remains the hypothesis generator and Elo
ranker, and its top-k hypotheses are validated each round. Validation and code
generation remain separate agents; independent evidence review and
accept-only-what-was-checked reconciliation follow. The coercion is dropped by
changing two things:

1. **The orchestrator is plain Python, not an LLM.** A round is a fixed
   sequence — tournament → validators → reviewers → reconcile → reasoner. There
   is nothing for a model to decide about scheduling, so nothing to police.
2. **Every agent returns a strict typed pydantic object.** Unknown output keys
   are rejected, and pydantic-ai asks the model to retry malformed output.
   Review outputs additionally require one ordered verdict per claim and
   complete acceptance evidence (`verification`, `source_ref`, and what was
   checked).

Tool restrictions become *which tools an agent is given*, not runtime policy.

## Shape of a round

```
            goal.md + last brief
                    │
          ┌─────────▼──────────┐
          │ Tournament (Jnana) │  generate/refresh hypotheses, Elo-rank, top_k
          └─────────┬──────────┘
        ┌───────────┼───────────┐        (asyncio.gather, bounded concurrency)
  ┌─────▼─────┐┌────▼──────┐┌───▼───────┐
  │ Validator ││ Validator ││ Validator │  owns one hypothesis; plans; delegates
  └─────┬─────┘└───────────┘└───────────┘
        │ tool: code_generator(instruction)  ──►  Code-generator agent
        │                                          (skills + execution tools)
        ▼
  ValidationReport {outcome, evidence_score, claims[]}
        │  claims only (never the prose report)
  ┌─────▼─────┐
  │ Reviewer  │  read-only; verdict per claim: accepted/rejected/unverifiable
  └─────┬─────┘
  ┌─────▼──────┐
  │ Reconcile  │  deterministic: accepted+verified claims → ledger.jsonl,
  └─────┬──────┘  each weighted by verification mode, grouped for independence
  ┌─────▼──────┐
  │ Steering   │  deterministic: Elo ⊕ evidence → priority + recommended action,
  └─────┬──────┘  over the whole Jnana pool
  ┌─────▼─────┐
  │ Reasoner  │  no tools; reads threads + steering table, writes the brief
  └─────┬─────┘
        ▼
   Brief → appended to the next tournament's goal
```

## Agents

| Role | Model (configurable) | Tools | Output |
| --- | --- | --- | --- |
| orchestrator | — (Python) | — | round JSON on disk |
| tournament | Jnana `generation` + `ranking` endpoints | — | ranked hypotheses |
| validator | `models.validator` | `code_generator`, `read_skill`, `fetch_url`, read-only exec | `ValidationReport` |
| code_generator | `models.code_generator` | `read_skill`, `fetch_url`, full exec | `CodeResult` |
| reviewer | `models.reviewer` | `read_skill`, `fetch_url`, read-only exec | `Review` |
| reasoner | `models.reasoner` | none | `Brief` |

Each role has its own `ModelSpec {model, base_url, api_key_env}`. With a
`base_url`, it is any OpenAI-compatible endpoint (vLLM, ALCF proxy, ...);
without one, `model` is a pydantic-ai model string (`anthropic:...`, `test`).

## Execution layer

Three modes, picked in the sole campaign config:

- `local` — trusted, unsandboxed execution for dry runs and laptops. The
  code-generator gets `write_files`, `run_shell`, `read_file`, and `list_dir`;
  file-tool paths are confined to a per-hypothesis workdir, but `run_shell` can
  access anything the loam process can. Do not use it for untrusted code.
- `mcp` — connect to an externally started MCP server at `mcp_url`.
- `hpc` — loam converts `execution.hpc` into the deployment server's
  internal environment, starts that server, waits for readiness, and stops it
  with the campaign. Users do not maintain or source another settings file.

Both MCP-backed modes attach a pydantic-ai `MCPToolset` filtered per role. The
code-generator gets all seven tools: `write_files`, `run_batch`, `batch_status`,
`run_code`, `inspect_filesystem`, `list_batches`, and `pool_status`;
validator/reviewer get the four read-only/status tools `inspect_filesystem`,
`batch_status`, `list_batches`, and `pool_status`. HPC science is
`run_batch`-first: read `pool_status`, stage one script and a JSONL input file
with `write_files`, then submit the sweep in one `run_batch` call with the
skill-required `venv_path`. Execution is observed only when the owned batch
reports both `complete: true` and `ok: true`. `run_code` is reserved for result
harvesting and failed-row diagnostics, never the scientific run. There is no
campaign scheduler wrapper.

## Skills

Kept in `skills/` (configurable list of dirs). Every
`SKILL.md` frontmatter becomes one line in a catalog injected into the
validator/code-generator instructions; `read_skill(name, file=None)` reads the
body or a bundled script/reference. No install step, no Hermes skill hub.

## Per-hypothesis workspace

One workspace per Jnana hypothesis id, **kept across rounds** (loam's was per
plan), so a hypothesis re-ranked into a later round starts from its own prior
runs, outcomes and still-pending batches, which are put into its validator's
prompt. IDs are identity and path components, not display text: they must match
`[A-Za-z0-9][A-Za-z0-9_.-]*`; unsafe IDs and an existing workspace whose stored
ID differs are rejected.

```
state/hypotheses/<id>/            records: orchestrator only
  experiment.json                 statement + one entry per round validated
  runs/<rNNN-KK>.json             instruction, state, result, batch receipts
<execution.workdir>/<id>/         files: agents read all of it
  outputs/<rNNN-KK>/              where that run writes
  data/                           workspace_inputs copied at creation (local only)
```

Records and files are separate trees so the reviewer's file tools can't reach
the validator's prior outcomes. Each `code_generator` call opens a numbered
run whose `write_files` are confined to that run's dir (local mode; `run_shell`
is not sandboxed). Reconcile groups an attested claim by the run named in its
path, so many claims from one run count as one independent observation.

## Failure handling

A large campaign runs many agents for hours; no single failure may end it.

| failure | handled by | effect |
| --- | --- | --- |
| tool error or tool timeout (`timeouts.tool`) | `Resilient` toolset wrapper | returned to the model as a failed tool result; no retry budget spent |
| shell command timeout | `run_shell` | whole process group killed |
| 429 / 5xx / dropped connection to an LLM | provider SDK, `model_retries` | retried with backoff; each request bounded by `timeouts.model_request` |
| 429 / 529 / 503+Retry-After from an OpenAI-compatible endpoint | `rate_limits` gate shared per endpoint | every agent and Jnana caller waits out one cooldown, then calls are spaced; `rate_limits.requests_per_minute` caps calls throughout |
| dropped MCP session | one connection per agent run | fails only that run's calls; the next run reconnects |
| code-generator crash, timeout, or request limit | `delegate` | reported to the validator, run recorded as `crashed` |
| validator or reviewer crash/timeout | `Campaign.validate` | that hypothesis is untested this round |
| reasoner failure | `Campaign.synthesize` | previous brief carried forward |
| Jnana generation timeout | `Campaign.rank` | generation is stopped and the round aborts before ranking |
| other tournament failure | `Campaign.rank` | validate the pool's current top_k by Elo |
| process killed mid-round | resume | ledger rows and experiment entries for the unfinished round are replaced, not duplicated; JSON written atomically |

Apart from a Jnana generation timeout, which deliberately aborts before ranking,
only an empty hypothesis pool stops a campaign.

## Steering score

Ported from loam's `project_hypothesis_steering.py` + `WEIGHT_POLICY`:

| verification | strength·reliability·relevance | independence group |
| --- | --- | --- |
| source_confirmed | 0.8·0.9·0.8 = 0.576 | the source URL |
| artifact_attested | 0.7·0.7·0.8 = 0.392 | the run named in the path (else the validation thread) |

Each claim's delta is ±weight (0 for `context`). Within a group, repeats
decay 1 / 0.25 / 0.10, so one run's five failed gates are one contradiction,
not five. Then

```
evidence_rating = 1200 + (400/ln10)·Σ deltas      # Elo scale; 1200 = no net evidence
uncertainty     = 1/√(1 + independent groups)     # 1.0 = never tested
priority        = 0.35·z(Elo) + 0.50·z(evidence) + 0.15·z(uncertainty)
```

`recommended_action` comes from the same rules as loam:
`resolve_rank_evidence_disagreement` if |Elo − evidence| ≥ 100, else
contradiction resolution, independent verification, or refine. The table
covers every hypothesis in the Jnana pool, not just this round's. It is stored
in the round file and given to the reasoner, whose prompt says how to turn it
into the `tournament_brief`. Jnana's generator only sees the brief, so the
reasoner has to state the table's conclusions in words.

## Integrity rules kept from Loam (in prompts + data flow)

- Reviewer sees only the claims and their sources, never the validator's prose.
- Default verdict is `unverifiable`; `accepted` requires `verification` and
  `source_ref`, enforced by reconcile, not trust.
- Failed predeclared gate = `contradicts`; passed = `supports`; `context` is
  for records merely used.
- A wait is not a finding; untested ≠ contradicted.
- Reference data is fetched by identifier at run time and verified, never
  recalled.
- `fetch_url` permits only HTTPS destinations whose resolved addresses are
  public; redirects are checked by the same rule.
- Elo never substitutes for evidence.

## Dropped on purpose (add back only if a run proves it necessary)

Hermes adapter/plugin, tool-call policy, delegation repair, run_code budgets
(replaced by a simple per-validator `max_code_calls`), ledger parsing,
remote (HPC-side) workspace hashing/verifier, batch-to-run binding,
knowledge-graph pre-pass, W&B hooks, preflight gate, backlog review,
statement-endorsement keys, artifact hashing and hash-chained logs. Hashes and
hash chains are deliberately absent; do not infer them as an integrity layer.

## Layout

```
loam/
  config.py        Config + ModelSpec (YAML → pydantic)
  skills.py        discover SKILL.md, catalog, read_skill
  execution.py     local tools / filtered MCP toolset, fetch_url
  schemas.py       typed outputs
  agents.py        build the four agents
  tournament.py    Jnana wrapper
  workspace.py     per-hypothesis records and runs
  reconcile.py     accepted claims → weighted ledger rows
  steering.py      Elo ⊕ evidence → priority table
  orchestrator.py  the round loop
  __main__.py      CLI
  prompts/*.md     condensed from loam/.loam/agents/*.md
examples/          example config + toy goal
```

State on disk (`state_dir/`) is exactly `jnana.json`, `ledger.jsonl`,
`hypotheses/<id>/experiment.json`, `hypotheses/<id>/runs/<run>.json`, and
`rounds/round-NNN.json` (`round`, `rankings`, `threads`, `steering`, `brief`).
There is no duplicate `steering.json` or `log.jsonl`. Resume parses only numeric
`round-<digits>.json` names and continues after the highest number; gaps and
unrelated filenames cannot cause a completed round to be overwritten.

## Configuration contract

The user supplies exactly one campaign YAML. It contains `rounds`, every agent
and Jnana model, and the execution choice: trusted `local`, an external server
in `mcp`, or a loam-managed deployment in `hpc`. Managed HPC keeps the
endpoint, allocation, queue, shared paths, pool, and server inputs under
`execution.hpc`; loam translates those values into the internal environment
consumed by the deployment scripts. Users do not edit or source another file.
Model API keys and proxy tokens are named with `api_key_env`, never stored in
YAML. Globus and Academy credentials likewise remain in their external login or
credential stores.

Configuration is strict: unknown keys and missing required role/model sections
are errors rather than silently ignored settings. Numeric limits, `rounds`, and
timeouts must be positive. External MCP mode requires an HTTP(S) `mcp_url`;
managed HPC requires its complete `hpc` block. Both require an absolute POSIX
`workdir` without `..` and disallow `workspace_inputs`. Jnana retains separate
`generation` and `ranking` model specs; `top_k` selects hypotheses for the
separate validator/code-generator agents and does not replace Jnana's ranker.

## Status and verification gaps (2026-11-10)

Implemented in source: the fixed Python round sequence, Jnana generation and
top-k ranking adapter, distinct validator/code-generator agents, per-hypothesis
state, independent review/reconciliation, steering, local execution tools, and
role-filtered MCP tools. The retained design intentionally has no artifact
hashes, hash chains, or scheduler wrapper.

Still requiring end-to-end verification: a live LLM endpoint; generation and
ranking through a real Jnana installation; a real hpc-deployment MCP connection;
and an HPC sweep proving `write_files` staging, one-script/JSONL `run_batch`,
owned `batch_status` completion with `complete: true` and `ok: true`, output
harvesting, and failed-row `run_code` diagnostics. No live local or HPC campaign
result is claimed here.

Run: `python -m loam examples/loam.yaml`

## Open questions

- Should the reasoner be able to inject explicit seed hypotheses into Jnana
  (vs. only steering via the goal text)? Currently: goal text only, as in loam.
- The steering weights are loam's, which were never calibrated against
  outcomes. Worth revisiting after a few real rounds.
