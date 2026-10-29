# develop-memory: FALDA and Cairn together, best effort

*Design, 2026-10-30. Branch `develop-memory`, cut from `develop-cairn` (itself based on the current
`develop`, which is what gets deployed).*

## 1. Intent

One branch with both integrations on by default, ready for an Aurora experiment on 2026-10-31
(~2,000 nodes, ~200 agents). Two outcomes, in order:

1. A run of the branch transmits memories to hosted FALDA and publishes tool ratings to Cairn.
2. Both are strictly best effort: a slow, overloaded, erroring or unreachable FALDA or Cairn never fails
   an agent step and adds no meaningful time to it. FALDA and Cairn stay enabled throughout; a call that
   takes too long is simply skipped.

The experiment will produce memories far faster than FALDA can distil them. Losing or never recalling
memories is acceptable; slowing agents is not. Recall is expected to return little or nothing.

The experiment will be launched by someone else, so it must be hands-free: no credentials to paste, no
FALDA/Cairn setup steps. Getting this test to transmit matters more than FALDA/Cairn security hygiene.

The addition must be minimally invasive to `develop`: no change to the biology / campaign behaviour
(agent prompts, validator rules, adapter behaviour) beyond what FALDA and Cairn need.

## 2. Goals, non-goals, anti-goals

Goals
- `develop-memory` = `develop-cairn` + only the FALDA pieces of `ryan/alcf-local-campaign`, ported onto
  `develop-cairn`'s code (not a whole-branch merge).
- FALDA enabled by default (as Cairn already is), including the reasoner's FALDA tools, with an opt-out.
- Loam obtains its own FALDA key unattended (mint on first use, cache under `$HOME`), as Cairn does.
- Every FALDA/Cairn write off the agent's critical path.
- Every FALDA/Cairn read on the critical path capped at 2 s wall clock.

Non-goals
- Making FALDA keep up, retrying until delivered, or guaranteeing delivery.
- FALDA server changes. Unattended machine keys are already deployed (§5.5).
- Stream capture for the FALDA distiller.
- The FALDA branch's non-FALDA changes: validator outcome-keyword rule (`d26409b`), hypothesis
  emphasis stripping (`7ff6402`), reasoner claim/duplicate prompt rules (`5c3f18a`, `2cf22b0`,
  `431a7dd`), Hermes adapter changes (`1a4aa41`, `d0c984d`), the `run-hermes` move, orchestration-loop
  commits predating the FALDA work, and the memory-backend-interface plan doc.

Anti-goals (a failure even if everything "works")
- Any agent step waits more than ~2 s on a single FALDA or Cairn read, or at all on a write.
- A FALDA/Cairn exception, hang, bad response or full disk surfaces as an agent or hook failure.
- Hermes runtime state, `.pyc` files or secrets (keys) end up in the branch.

## 3. Starting point (as found)

- Merge base of the two branches: `b9b1ad3`. `develop-cairn` is 39 commits past it (including upstream
  develop work); the FALDA branch is 23. Files that conflict when merging: the Hermes plugin
  `__init__.py`, `loam_config.py`, `hypothesis_orchestration.py`, `pre-tool.py`.
- All Loam hooks (so all FALDA and Cairn traffic) run on the host running Hermes. Today's kits run
  Hermes on a laptop or an Aurora login node (direct outbound HTTPS; the ALCF proxy must not be set
  there). The experiment may instead run agents (Hermes and its hooks) on compute nodes, which reach the
  internet only through the ALCF proxy `http://proxy.alcf.anl.gov:3128`. The design must work in both
  placements.
- Cairn (`develop-cairn`): ratings are queued to a local JSONL under `LOAM_STATE_DIR` at child stop and
  sent by a detached, lock-guarded flusher (`cairn_flush.py`) — already off the critical path; reviewer
  key minted automatically on first flush. The one synchronous call is the reputation brief
  (`cairn_brief.reputation`) at canonical delegation: one `score_batch` per plan, timeout default 5 s
  (cap 10), cached per plan including failures, fail-open.
- FALDA (`ryan/alcf-local-campaign`): `falda_memory._post` is a synchronous urllib call, default timeout
  10 s, never raises. Three hook call sites, all on the critical path, in `hypothesis_orchestration.py`:
  - `write_validator_outcome` → `upsert`, at every validator stop (`record_stop`).
  - `generation_memory` → `recall`, before Jnana generation (only when Jnana will generate).
  - `reasoner_memory_block` → `recall`, once per plan (cached per plan).
  Plus the `reasoner_tools` switch: the reasoner is told to call `mcp__falda__falda_remember` /
  `falda_recall` itself through Hermes's MCP client, guarded fail-closed by `pre-tool.py`.
- FALDA is enabled only by a `falda` block in the Loam profile, requires a pool UUID, and reads
  `FALDA_TOKEN` / `FALDA_TENANT` from the environment.
- FALDA now issues machine keys without login (§5.5), tested 2026-10-30 against a throwaway pool:
  mint 0.9 s; `/atoms/upsert` of a validator-outcome atom ~2.6 s; `/recall` with `require_tags`
  0.9–1.9 s and returns the atom immediately (explicit atoms do not wait for distillation); a key reaches
  only the pool it enrolled in (403 `not_a_member` elsewhere); MCP `initialize` with key + pool header
  works. Timings are from New Zealand; Aurora is likely faster, but some 2 s recalls will time out.

## 4. Decisions

| Decision | Choice |
|---|---|
| Branch | `develop-memory` from `develop-cairn`. Port the FALDA modules (`falda_memory.py`, `falda_campaign.py`, the `falda` config block, the three call sites, the reasoner tools and fail-closed tool guard, the `falda` manifest summary) and their tests onto `develop-cairn`'s versions of the touched files. If a FALDA helper relied on a non-FALDA change (e.g. outcome-keyword parsing), carry it inside the FALDA module rather than changing validator or reasoner behaviour |
| Artefacts | Nothing from `.hermes/`, no `.pyc`, no `.swp`, no keys |
| FALDA default | On: ship `.loam/falda.json` (same shape as the profile block) mirroring `.loam/cairn.json`; a profile `falda` block wins; `falda: {enabled: false}` opts out. All switches on by default, including `reasoner_tools` |
| FALDA server / pool | `https://falda.cairnscore.ai`, pool `efb99ee4-c412-11f1-8af6-0affe8db6a05` (the existing Loam pool, which machine keys may join). One pool for every runner so memories are shared; campaigns are separated by the workflow tag. Pool stays a UUID in config |
| FALDA headers | `Authorization: Bearer <key>` and `X-Falda-Pool: <pool>` (or `pool` in the body, as the client already does). `X-Falda-Tenant` is no longer sent |
| FALDA writes | Spool + detached flusher, the Cairn pattern: the hook appends the atom to a local spool and returns; a detached, lock-guarded process sends pending atoms in batches (FALDA `upsert` takes a list) |
| Read cap | 2 s hard wall-clock timeout for FALDA recalls (generation, reasoner) and the Cairn reputation brief (Cairn default `timeout_seconds` 5 → 2). A read that fails or times out returns the existing "no memory" / "no scores" result; the next read tries again. No circuit breaker |
| Reasoner tool calls | Enabled. Hermes's MCP connection to FALDA gets a short tool-call timeout so a slow FALDA fails the call quickly and the reasoner carries on; a FALDA MCP server that cannot connect at Hermes start must not stop Hermes |
| Reasoner prompt | No edits to agent prompt files beyond what the reasoner tools need (the FALDA branch's "Shared campaign memory" guidance in `.loam/agents/reasoner.md` is FALDA's, so it comes across; nothing else in that file does). The injected recalled-memory text is added only when recall returned content; the "Memory interface" lines describing the tools are always present when tools are on |
| FALDA credentials | Resolution order: `FALDA_TOKEN` env, then a cached key file (default `~/.falda/keys/loam-<pool>.json`, `0600`, outside the repo, keyed by pool because a key reaches only its own pool), then an unattended mint (§5.5) whose result is cached to that file. One key per user per pool: `$HOME` is shared across Aurora nodes, so all runners reuse it. With no key, FALDA operations log `missing_token` and the run continues |
| FALDA campaign tag | `workflow:<id>` where id is the run's `workflow_id` unless it is the generic environment fallback (`environment-workflow`), then `CAIRN_PROJECT` if set, then `loam-<UTC date>` |
| Proxy | FALDA and Cairn HTTP clients pick their proxy automatically: an explicit `https_proxy`/`HTTPS_PROXY` in the environment wins; otherwise, on an Aurora compute node (detected from the hostname, not a `aurora-uan-*` login node), use `http://proxy.alcf.anl.gov:3128` for FALDA/Cairn requests only, set on their own urllib opener, never exported into the agent's environment (so biology tooling is unaffected); otherwise direct. Applies to the hooks, the detached flushers and both key mints. The launcher also exports the proxy for Hermes's FALDA MCP connection when Hermes starts on a compute node |
| Locks on the agent's path | If agents run on compute nodes the state dir may be on a shared parallel filesystem. Any lock taken on the critical path (e.g. the Cairn brief cache lock) is non-blocking: if busy or unsupported, skip the cache and continue. Detached flushers already use non-blocking locks |
| Failure surface | Every new code path (spool, flusher spawn, key mint, cache file) catches everything and fails open, matching the existing "client never raises" contract. The tool guard stays fail-closed (it blocks an unverifiable FALDA tool call; it never blocks other tools) |

## 5. Behaviour

### 5.1 FALDA writes
- At validator stop, `write_validator_outcome` builds the atom exactly as today, appends one row (atom,
  plan/hypothesis ids, `pending`) to a FALDA spool in the state dir, and spawns the detached flusher if
  none holds the lock. No network I/O in the hook.
- The flusher (which may also do the key mint, §5.5) sends pending rows in batches, marks them
  sent/failed, and exits. Writes may keep a longer timeout (off-path). Failed batches stay for a later
  flush; no retry loops. Spool growth is bounded (builder's call on the cap); oldest unsent rows are
  dropped first and counted.
- Every outcome still goes to `falda-calls.jsonl` via `log_event`, so the run manifest stays truthful.
- At the end of a campaign, a final flush is attempted the same way Cairn's is (best effort).

### 5.2 FALDA recalls
- `generation_memory` and `reasoner_memory_block` recall with a 2 s wall-clock cap. On failure or
  timeout they return the existing "unavailable" result immediately. Recall never mints a key: with no key
  cached, it returns "unavailable" (`missing_token`).
- The reasoner's per-plan cache is unchanged.

### 5.3 Cairn
- Reputation brief: default timeout 2 s (config may set lower; cap stays 10).
- Rating queue, flusher and key mint: unchanged.

### 5.4 Launch
- No launch-time steps. Cairn mints its own key on first flush. FALDA mints its own key on first flush.
- Hermes's FALDA MCP server needs the key at Hermes start (it is passed as a header). The Hermes
  launcher (or the Loam adapter at load) resolves the key per §4 without minting if absent, best
  effort, ≤ 2 s, and exports it for the MCP config. If no key exists yet, the reasoner tools are
  unavailable for that Hermes session (their calls fail fast); hook-side writes and recalls are
  unaffected. The FALDA MCP server entry must be present in the Hermes config the run uses.
- Traffic leaves from the host running Hermes, through the ALCF proxy when that is a compute node
  (§4 Proxy).

### 5.5 FALDA key minting
- Contract (deployed): `POST /auth/machine`, no auth, JSON `{"pool": "<uuid>", "label": "<name>"}`
  (omit `pool` to create a new pool). Response
  `{"api_key": "falda_…", "tenant": "machine-…", "pool": "<uuid>", "pool_created": bool, "user": {…}}`;
  the key is shown once. Errors: 404 `no_such_pool`, 429 `rate_limited`. Limits: 50 mints/hour per IP,
  500/hour total.
- Loam always passes the configured pool (never creates pools). Label `loam-<hostname>-<user>`.
- Only the detached flusher mints; recalls and hooks never do. Mirrors `cairn_keys.py`: a cool-down
  after a failed mint (longer after 429), status recorded in a state file, no key material in logs or
  state.
- Mint exclusivity must hold across nodes (all Aurora compute nodes egress through the one proxy IP and
  share `$HOME`): take the mint lock by atomic create (`O_EXCL`) of a lock file beside the key file, with
  a staleness timeout, not `flock`. A process that does not get the lock skips minting and leaves its
  rows pending; it picks up the key file on its next flush. Write the key file atomically (temp file +
  rename) so readers never see a partial key.

## 6. Verification

- Existing test suites from `develop-cairn` and the ported FALDA tests pass on `develop-memory`.
- New tests, using the existing fake FALDA transport (`tests/falda_fake.py`) and fake Cairn client:
  - a hanging transport costs a recall ≤ ~2 s;
  - a validator stop performs no network I/O and returns immediately when FALDA hangs;
  - the flusher drains the spool against the fake and leaves rows pending when the fake fails;
  - key resolution: env wins; cached file used; mint called once with the configured pool, cached
    atomically, cooled down after failure and after 429; a held `O_EXCL` lock skips minting;
  - transport exceptions, unwritable state dir and malformed spool rows never raise out of a hook;
  - FALDA is on with no profile block, and off with `falda: {enabled: false}`;
  - the Cairn brief default timeout is 2 s;
  - proxy selection: env proxy wins; compute-node hostname without env → ALCF proxy on the
    FALDA/Cairn opener only; login-node or other hostname → direct; `os.environ` unchanged;
  - a busy Cairn brief lock does not block delegation.
- Once the FALDA endpoint is deployed: a local smoke run against the real services shows successful
  upserts in `falda-calls.jsonl` and submitted batches in `cairn-status.json`. An Aurora debug-queue
  (1 node) run if time allows.

## 7. Left to the builder
- Spool cap, flusher batch size and spawn trigger; mint cool-down length.
- How to fold the FALDA call sites into `develop-cairn`'s `hypothesis_orchestration.py`, `pre-tool.py`
  and adapter, provided Cairn behaviour is untouched.
- The exact compute-node hostname test (Aurora compute nodes are named like `x4xxxc*s*b*n*`).
- How Hermes's MCP tool-call timeout is set (Hermes config field or adapter), once checked against the
  Hermes version in use.
- Whether FALDA and Cairn share a small spool/flusher/key helper or each keep their own.
