# Cairn tool reputation

## What it does

Before delegating, Loam appends a short reputation brief (composite score and confidence) for the
configured tools to each code-generator and hypothesis-validator context (never the reasoner's, whose
synthesis becomes Jnana steering). When a child stops, Loam parses its `### Tool Ratings`
report section, queues one rating event per backed tool, and submits the queue to Cairn in batches
from a detached process. Every Cairn failure is recorded and swallowed; it never changes delegation
or fails a campaign.

## Project silo: nothing to set

Every Cairn request (the brief, the flusher, the key mint) carries `X-Cairn-Project: <project>`, so the
run reads and writes its own private silo instead of production scores. By default the project is
derived from the profile's `workflow_id` (`loam-<workflow-id>`, lower-cased, non-alphanumerics as `-`,
at most 63 characters; `loam-<UTC date>` without one), so every runner of one campaign shares a silo
with no setup. To choose it yourself, export it in the shell that launches Hermes:

```bash
export CAIRN_PROJECT=<experiment-name>    # ^[a-z0-9][a-z0-9-]{2,62}$; use an unguessable name for sensitive runs
```

It comes only from the launch environment or the workflow id: profiles cannot set it and agents never see
or type it.
The value is bound when the profile loads and written into every queued rating; a flush sends only
rows queued under its own project (others are counted as `other_project`).

If `CAIRN_PROJECT` is invalid, Loam logs `Loam Cairn: WARNING CAIRN_PROJECT=... is invalid ...: using
<default>` once and uses the default project. A request never goes out without the header. Every response must echo the header; a missing or different echo logs a warning,
discards a fetched brief, and sets `project_echo` in `cairn-status.json`. Keys are not bound to
projects, so the same key works in every silo.

Check a host and the project before a run (one unauthenticated read; mints nothing):

```bash
CAIRN_PROJECT=<experiment-name> python scripts/cairn_mint_key.py probe
# project ok: <experiment-name> (echoed); mcp://hpc#run_code composite=0.5 confidence=0.0
```

Agents are told the scores come from the experiment's silo, so mostly empty scores early in a run
are expected.

## Enable (on by default)

Cairn is on by default. The repo ships `.loam/cairn.json` (same shape as the profile block below,
with the Loam tool entities), and it is used whenever the profile has no `cairn` key. Anyone who
checks out this branch and runs a campaign gets Cairn with no setup. If `.loam/cairn.json` is
unreadable or invalid, Cairn stays off and the profile still loads.

Opt out in either way:

- profile: `cairn: {enabled: false}` (a profile `cairn` block always wins over the default file), or
- environment: `LOAM_CAIRN=off` (also `0`, `false`, `no`, any case); this wins over everything. Set it
  before launching Hermes: the setting is read from the environment of the process that starts the hooks.

All Cairn traffic (the reputation fetch, the flusher and the key mint) comes from the host running
Hermes, never from compute nodes: code-generator payloads run on compute nodes, but Cairn calls are made
by Loam's hooks. That host needs outbound HTTPS to `api.cairnscore.ai`:

- a laptop or workstation: usually nothing to do;
- an Aurora login node (`aurora-uan-*`): outbound access is direct. Do **not** set the proxy variables
  there; ALCF documents that the proxy does not function on login nodes;
- an Aurora compute node (`x4…c…s…b…n…`): nothing to do. With no `https_proxy`/`HTTPS_PROXY` set,
  Loam's hooks send their Cairn (and FALDA) requests through `http://proxy.alcf.anl.gov:3128`
  (`.loam/hooks/loam_http.py`), and `run-hermes` exports `https_proxy`/`HTTPS_PROXY` for Hermes (its
  FALDA MCP connection) with loopback, the node itself and `.alcf.anl.gov` in `no_proxy`. Plain-HTTP
  proxy variables are not set;
- another ALCF host without direct access: export `https_proxy`/`HTTPS_PROXY` before launching Hermes;
  an exported proxy always wins.

Without outbound HTTPS, ratings stay queued (`unsent`, retried on the next flush) and the read side costs
at most 2 s of wall clock per plan when it fetches the brief (`timeout_seconds` is capped at 2 s for the
brief; the flusher uses it as set). A busy reputation-cache lock never makes a delegation wait. `cairn_mint_key.py probe` (above) checks
both reachability and the project echo.

To change settings or tools, add a `cairn` block to the Loam profile in `loam.yaml`; it replaces the
default block entirely. It carries paths and switches only, never secrets.

```yaml
cairn:
  enabled: true                               # required; anything but true turns Cairn off
  base_url: https://api.cairnscore.ai         # default
  key_file: ~/.cairn/keys/agent-loam.key      # default
  auto_mint: true                             # default; mint the key on first flush if none is on disk
  identity_prefix: agent://anl-loam/loam-     # default; must contain ://; random hex is appended
  # identity: agent://anl-loam/loam-me        # optional; used as-is instead of a generated one
  read_side: true                             # default; append the reputation brief (changes agent behaviour)
  rate_agent_reports: true                    # default; queue ratings from agent reports
  rate_hook_fallback: true                    # default; hook-derived executor rating for failed code-generators
  auto_flush: true                            # default; spawn a detached flusher at each child stop
  rate_roles: [code-generator, hypothesis-validator]   # default; allowed: code-generator, hypothesis-validator, reasoner
  timeout_seconds: 5                          # default; capped at 10
  executor: mcp://hpc#run_code                # default; entity used for the fallback rating
  entities:                                   # required, non-empty; ids start with mcp:// or tool://
    mcp://hpc#run_code: {task_tags: [code_exec]}
    tool://lammps: {patterns: [lmp, lammps], task_tags: [md_simulation]}
```

`patterns` are lowercase substrings matched against execution-call arguments. `task_tags` (at most
10 per entity) are the only tags ever sent; Cairn accepts only snake_case tags, so each is
normalised (`MD-Simulation` becomes `md_simulation`) and a tag with nothing usable left (`42`, `--`)
makes the block invalid. The executor entity is added automatically with the tag
`code_exec` if you omit it. `timeout_seconds` must be greater than 0; otherwise the whole block is
invalid and Cairn stays off. See `examples/loam.yaml.example`.

## The key is minted automatically

On the first flush with no key on disk, Loam mints a reviewer key under a random
`agent://anl-loam/loam-<8 hex>` identity into `~/.cairn/keys/agent-loam.key` and writes the
(non-secret) identity to `~/.cairn/keys/agent-loam.key.identity`. It uses the same safeguards as the
manual tool below: reserve the file first, never overwrite, `0600` key, `0700` directory. If another
process is already minting, or the mint fails, rows stay pending and the next flush tries again
(`cairn_flush.py --report` never mints). Back the key up: it cannot be re-minted.

Check `cairn-status.json` in the state dir (`$LOAM_STATE_DIR`): it is rewritten by every flush with
`updated_at`, `key_file`, `key` (`present`, `minted`, `missing`, `mint_failed`, `mint_busy`),
`identity`, `base_url`, `project`, `project_echo` (`ok`, `missing`, `mismatch`, or empty when nothing
was sent), `counts` and `error`, plus `mint_failed_at` (UTC time of the last failed mint,
carried through later flushes until a mint succeeds or a key is found). It never contains the key. The flusher is detached and silent, so this file is where mint problems
appear (loose `~/.cairn/keys` permissions, a read-only `$HOME`, a failed or refused mint). After a
mint that may have claimed an identity without a reply (`mint_failed`), minting is skipped for 10
minutes (from `mint_failed_at`, even across flushes that do not mint) so Cairn is not littered with orphaned identities.

An empty key file left by a killed minter (older than 10 minutes, regular file) is recovered
automatically: it is renamed to `<key>.abandoned.<pid>.<time>` and minting proceeds; a fresh one is
treated as another process minting (`mint_busy`).

To find tools to add to `.loam/cairn.json`, look at `unknown_entities` in `--report`: ratings for an
entity that is not configured are recorded locally as `invalid` (`unknown_entity`) with the id the
agent wrote, and counted as `"unknown_entities": {"tool://gromacs": 2}` (shown only when non-empty,
also in `cairn-status.json`).

## Mint the key manually

```bash
python scripts/cairn_mint_key.py mint     # --identity (default agent://anl-loam/loam-<random hex>), --key-file, --base-url, --timeout
python scripts/cairn_mint_key.py check    # --key-file
python scripts/cairn_mint_key.py probe    # --project (default $CAIRN_PROJECT), --entity, --base-url
```

`mint` also needs a valid `CAIRN_PROJECT` (or `--project`): every request carries the header, although
the key itself works in any project.

An identity can be claimed once, ever. `mint` first refuses if a key file exists, prepares the key
directory (created `0700`; an existing directory is never chmod-ed, and one with group/other
permission bits is refused), and reserves the `0600` key file, and only then claims the identity.
`mint` defaults to a 10 s timeout; the flusher defaults to 5 s. Back the key up: it cannot be re-minted. Keep it under `$HOME`, never on shared storage such as
`/flare`. `check` verifies the file is present, non-empty, and not group/other accessible, and so
does its directory.

If `mint` gets no response at all (a timeout or dropped connection), it removes the empty
reservation and exits 1, but the identity may already have been claimed: do not retry blindly; ask
Cairn first. Once the key is fully written, a failing `fsync` or `close` keeps the file and is
reported (exit 1); run `check` and back the key up.

## During a run

- Each code-generator and hypothesis-validator delegation gets the reputation brief, and each role
  in `rate_roles` gets the rating-block instruction, appended to its built context (no `[loam-`
  markers are added). A reasoner gets nothing unless `reasoner` is in `rate_roles`, and then only the
  rating instruction. Headings inside fenced code blocks are ignored, so a quoted report cannot
  stand in for the agent's own `### Tool Ratings`. The brief is cached per plan in `cairn-reputation.jsonl`.
  The appended block is delimited by the lines `## Cairn tool reputation (added by Loam)` and
  `## End of Cairn tool reputation`, so it can be recognised or stripped; nothing is added when it
  would be empty.
- At each child stop the ratings step runs and appends immutable rows to `cairn-ratings.jsonl`.
  Transitions go to `cairn-submissions.jsonl`; locks are `cairn-reputation.lock` and
  `cairn-flush.lock`, all under `LOAM_STATE_DIR`.
- With `auto_flush`, a detached process submits pending rows. Only one flusher holds the lock.
- The `cairn_ratings` import in `subagent-stop.py` is guarded, so a Cairn failure never stops the
  lifecycle records.

## After a run

Once the campaign has finished, do both steps:

1. Send anything still pending (the last child's detached flusher may not have run, or may have
   found the network down). Use the same `CAIRN_PROJECT` as the run (or pass `--project`); rows
   queued under another project are never sent:

   ```bash
   python .loam/hooks/cairn_flush.py --state-dir "$LOAM_STATE_DIR" \
     --base-url https://api.cairnscore.ai --key-file ~/.cairn/keys/agent-loam.key
   ```

2. Report, which sends nothing:

   ```bash
   python .loam/hooks/cairn_flush.py --state-dir "$LOAM_STATE_DIR" \
     --base-url https://api.cairnscore.ai --key-file ~/.cairn/keys/agent-loam.key --report
   ```

`--report` prints JSON counts of queued events by state (`pending`, `in_flight`, `submitted`,
`unsent`, `failed`, `unknown`, `invalid`) and, under `recorded`, the rows that were never meant to be
sent, by status (`unbacked`, `suppressed_agent_error`, `invalid`, `missing_block`, `no_tool_used`).
Add `--resubmit-failed` to step 1 to resend `failed` batches (optional `--timeout`, default 5).

Delivery is at-most-once for anything that may have reached Cairn. A batch is classified by HTTP
status:

| Response | State | Resent? |
|---|---|---|
| 2xx | `submitted` | no |
| 400-499, 500, 501 | `failed` (definitely not applied) | only with `--resubmit-failed` |
| no response, and the request provably never left the host (DNS failure, connection refused, network or host unreachable, request not built) | `unsent` | automatically, on the next flush (never twice in one) |
| 502, 503, 504, any other status, any other no-response failure (timeout, reset) | `unknown` (may have been applied) | never |
| crash mid-send | `in_flight` | never |

`base_url` must start with `http://` or `https://`; otherwise the block is invalid and Cairn stays off.

## What is sent to Cairn

One event per rated tool: `context: "general"`, `reviewee.type: "capability"` with the entity id,
`score`, `weight`, `rationale`, `task_tags` (from the entity config), `failure_modes`, optional
numeric `metrics`, and `observed_at`. No `task` is sent. Weights are set by the hook from the role,
never by the agent: code-generator 1.0, hypothesis-validator 0.5, reasoner 0.2, hook fallback 0.2.

- The rationale starts with `[<role>, agent-authored] ` or `[code-generator, loam-hook fallback] `
  and is at most 2000 chars. The agent's note is stripped of `<...>` tags and cut to 1500 chars.
- A note that looks like a sequence (30+ amino-acid letters, either case) or a secret (`.cairn`, `api_key`,
  `x-api-key`, `{"`) is dropped whole; the rating is still sent without it.
- Failure modes are normalised to snake_case and prefixed with the self-reported origin
  (`infrastructure_walltime`, `software_oom`; `<origin>_unspecified` when none is named), at most 10.
  Metric names must be snake_case (`^[a-z][a-z0-9_]{0,31}$`); others are dropped, at most 10 kept.
  Agent-origin failures are recorded locally as `suppressed_agent_error` and not sent.
- A rating is sent only if backed (`_backed` in `cairn_ratings.py`); otherwise it is recorded
  locally as `unbacked` with the claim, and not sent. Backing depends on role and entity:
  - code-generator, executor entity (`mcp://hpc#run_code` by default): any execution call
    (`run_code` or `run_batch`) by the same child; no pattern match is needed.
  - code-generator, other entities: an execution call by the same child whose arguments contain one
    of the entity's `patterns`. Calls whose arguments were truncated by the tool audit never match,
    and inspection calls such as `inspect_filesystem` do not count.
  - hypothesis-validator: a completed code-generator record for the same plan and hypothesis, not
    its own calls.
  - reasoner (only if enabled in `rate_roles`): any completed code-generator record in the plan.
- Code-generator lifecycle rows (only with `rate_hook_fallback` on, and only when the report gave no
  rating for a configured entity; suppressed or unbacked ratings still count as ratings, since they
  are recorded locally):
  - interrupted or cancelled children: no fallback and no lifecycle row (rows from its own report
    are still recorded).
  - failed, with an execution call: one hook-derived executor rating (score 0.2, weight 0.2,
    failure mode `lifecycle_<status>`).
  - failed, no execution call: recorded locally as `no_tool_used`.
  - completed: recorded locally as `missing_block`, but only when `rate_agent_reports` is on and
    `code-generator` is in `rate_roles`.
- Batches hold at most 100 events; events failing local validation (including Cairn's tag and
  metric-key rules) are marked `invalid` and excluded so they cannot fail a whole batch.

## Known limits

- Ratings decay with a 3-day half-life, so reputation reflects recent runs.
- The hosted service is shared: other users' ratings affect the scores you see.
- The reasoner is not asked to rate by default. Its report becomes the next round's Jnana steering
  brief, so a ratings section would leak into hypothesis steering. Add `reasoner` to `rate_roles`
  to turn it on.
- Deviations from the design, made to keep the branch confined:
  1. No `cairn_status` in `run-manifest.json` and no preflight check; use `cairn_mint_key.py check`
     and `cairn_flush.py --report`.
  2. The rating-block instruction is injected into the built contexts, not `.loam/agents/*.md`.
  3. The reasoner is off by default (above).
  4. The brief shows composite and confidence only, not failure modes.
  5. A completed code-generator without a rating block is recorded locally, not rated; only a failed
     one gets the executor fallback.
  6. The infrastructure-origin override is omitted; origin is self-reported and suppressions stay
     auditable in the queue.
  7. Minting uses a stdlib client rather than Cairn's `mint-key.sh`, which is not installed on run
     hosts, with the same guarantees (exclusive create, `0600` key, `0700` directory, no overwrite).
