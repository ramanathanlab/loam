# develop-memory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the FALDA campaign memory from `ryan/alcf-local-campaign` onto `develop-cairn` as branch
`develop-memory`, with FALDA and Cairn both on by default and strictly best effort (no failures, ≤ 2 s on
any read, no time on writes).

**Architecture:** A small shared HTTP helper (`loam_http.py`) gives both clients ALCF proxy selection
and a wall-clock cap. FALDA writes go to a local spool sent by a detached flusher (Cairn's pattern);
the flusher also mints the FALDA machine key via `POST /auth/machine`, cached per pool under `$HOME`.
Recalls and the Cairn brief are capped at 2 s and return "no memory" on any failure.

**Tech Stack:** Python 3.11+ stdlib only in hooks (urllib, threading, fcntl, subprocess); pytest; Hermes
plugin adapter; bash launcher.

**Spec:** `docs/superpowers/specs/2026-10-30-develop-memory-design.md`

## Global Constraints

- Work on branch `develop-memory` (already cut from `develop-cairn`). Never merge `ryan/alcf-local-campaign`
  wholesale; copy files from it with `git show 'ryan/alcf-local-campaign:<path>'` and adapt. (In zsh,
  quote the `branch:path` argument or use `${F}:path`; `$F:t…` triggers a zsh modifier.)
- Do not port: validator directive / outcome-keyword changes, `_prompt_body` / reasoner-directive
  rewrite, hypothesis emphasis stripping, Hermes adapter `LOAM_HOOK_PYTHON` / `parent_session_id` changes,
  `run-hermes` move, the memory-backend plan doc, anything under `.hermes/`, `.pyc`, `.swp`.
- Hooks import only the standard library. Every FALDA/Cairn code path catches everything and returns a
  neutral result; nothing raises out of a hook or the adapter.
- No keys in the repo, in logs, or in state files.
- Test command (baseline on `develop-cairn`: 357 passed, 2 skipped):
  `uv run --with pytest --with pyyaml python -m pytest -q`
- FALDA defaults: `base_url` `https://falda.cairnscore.ai`; `pool` `efb99ee4-c412-11f1-8af6-0affe8db6a05`;
  `token_env` `FALDA_TOKEN`; `key_dir` `~/.falda/keys`; `recall_budget_chars` 1500;
  `recall_timeout_seconds` 2; `timeout_seconds` (writes, mint) 10; all switches true
  (`recall_generation`, `recall_reasoner`, `write_validator_outcomes`, `reasoner_tools`, `auto_mint`,
  `auto_flush`).
- Cairn: `timeout_seconds` default 5 → 2 (cap stays 10). Everything else unchanged.
- ALCF proxy `http://proxy.alcf.anl.gov:3128`. Aurora compute hostnames match
  `^x\d{4}c\d+s\d+b\d+n\d+`; login nodes are `aurora-uan-*`.
- FALDA mint contract: `POST {base_url}/auth/machine`, JSON `{"pool", "label"}`, no auth → 200
  `{"api_key", "tenant", "pool", "pool_created", "user"}`; 404 `no_such_pool`; 429 `rate_limited`.
- FALDA request headers: `Authorization: Bearer <key>`, `X-Falda-Pool: <pool>`; also `pool` in the body.
  No `X-Falda-Tenant`.
- Commit after each task with the message given, ending with the attribution trailer
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Recall hangs past its socket timeout** (slow-drip response, DNS stall): urllib's `timeout` is per
   socket op, so the 2 s cap must be wall clock (`loam_http.bounded`). Test in Task 1 with a transport that
   sleeps 5 s.
2. **Many processes mint at once on a shared `$HOME`** (200 agents start together): exactly one mints; the
   others skip without waiting and never see a half-written key file. Test in Task 4.
3. **Hermes on a compute node with `https_proxy` exported by the launcher**: local services
   (`127.0.0.1` HPC MCP, the ALCF inference proxy) must still be reached directly, so `NO_PROXY` must
   include `localhost,127.0.0.1,::1`. Test in Task 8.
4. **Spool grows without bound while FALDA is down for hours**: capped at 5,000 pending rows, oldest
   dropped and counted. Test in Task 5.
5. **FALDA MCP server unreachable or `FALDA_TOKEN` empty at Hermes start**: Hermes must still start and
   other tools work. Verified manually in Task 8.

---

### Task 1: Shared HTTP helper (proxy + wall-clock cap)

**Files:**
- Create: `.loam/hooks/loam_http.py`
- Test: `tests/test_loam_http.py`

**Interfaces:**
- Produces:
  - `ALCF_PROXY: str = "http://proxy.alcf.anl.gov:3128"`
  - `is_aurora_compute(hostname: str) -> bool`
  - `proxy_for(environ: Mapping[str, str], hostname: str) -> str | None` — `None` means "use urllib's
    defaults" (env proxy or direct); a string means "force this proxy".
  - `urlopen(request: urllib.request.Request, timeout: float)` — like `urllib.request.urlopen`, but with an
    explicit `ProxyHandler` opener when `proxy_for(os.environ, socket.gethostname())` returns a proxy.
    Never modifies `os.environ`.
  - `bounded(fn: Callable[[], T], seconds: float, fallback: T) -> T` — runs `fn` in a daemon thread, returns
    its result, or `fallback` if it raises or has not finished within `seconds`.

- [ ] **Step 0: Commit the design and this plan**

```bash
git add docs/superpowers/specs/2026-10-30-develop-memory-design.md docs/superpowers/plans/2026-10-30-develop-memory.md
git commit -m "docs: develop-memory design and plan (FALDA + Cairn, best effort)"
```

- [ ] **Step 1: Write the failing tests**

```python
def test_compute_and_login_hostnames():
    assert loam_http.is_aurora_compute("x4102c5s3b0n0")
    assert not loam_http.is_aurora_compute("aurora-uan-0010")
    assert not loam_http.is_aurora_compute("ryans-macbook")

def test_env_proxy_wins():
    assert loam_http.proxy_for({"https_proxy": "http://p:1"}, "x4102c5s3b0n0") is None
    assert loam_http.proxy_for({"HTTPS_PROXY": "http://p:1"}, "x4102c5s3b0n0") is None

def test_compute_node_without_env_uses_alcf_proxy():
    assert loam_http.proxy_for({}, "x4102c5s3b0n0") == loam_http.ALCF_PROXY

def test_login_and_laptop_go_direct():
    assert loam_http.proxy_for({}, "aurora-uan-0010") is None
    assert loam_http.proxy_for({}, "laptop") is None

def test_urlopen_uses_alcf_opener_without_touching_environ(monkeypatch):
    monkeypatch.setattr(loam_http.socket, "gethostname", lambda: "x4102c5s3b0n0")
    # stub urllib.request.build_opener to capture its ProxyHandler and return a fake opener
    before = dict(os.environ)
    loam_http.urlopen(urllib.request.Request("https://example.invalid"), 1)
    assert captured_proxies == {"https": loam_http.ALCF_PROXY, "http": loam_http.ALCF_PROXY}
    assert dict(os.environ) == before

def test_bounded_returns_result_fallback_on_raise_and_on_timeout():
    assert loam_http.bounded(lambda: 7, 2, None) == 7
    assert loam_http.bounded(lambda: 1 / 0, 2, "fb") == "fb"
    start = time.monotonic()
    assert loam_http.bounded(lambda: time.sleep(5) or 1, 0.3, "fb") == "fb"
    assert time.monotonic() - start < 1.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run --with pytest --with pyyaml python -m pytest -q tests/test_loam_http.py`
Expected: FAIL (`ModuleNotFoundError: loam_http`).

- [ ] **Step 3: Implement `loam_http.py`** per the Interfaces block (stdlib only; hostname regex from
  Global Constraints).

- [ ] **Step 4: Run to verify they pass**, then the full suite (expect baseline + new tests).

- [ ] **Step 5: Commit** — `Loam hooks: shared HTTP helper with ALCF proxy selection and a wall-clock cap`

### Task 2: Cairn on the shared helper; 2 s brief; non-blocking brief lock

**Files:**
- Modify: `.loam/hooks/cairn_client.py` (`_urlopen`), `.loam/hooks/cairn_brief.py` (`_timeout`, `reputation`),
  `.loam/adapters/hermes/plugin/cairn_config.py` (default `timeout_seconds`), `docs/cairn.md` (timeout
  default line and proxy paragraph)
- Test: `tests/test_cairn_client.py`, `tests/test_cairn_brief.py`, `tests/test_cairn_config.py`

**Interfaces:**
- Consumes: `loam_http.urlopen`, `loam_http.bounded`.
- Produces: no new names. `cairn_client._urlopen` now defaults to `loam_http.urlopen` (tests that
  monkeypatch `_urlopen` keep working).

- [ ] **Step 1: Write the failing tests**
  - `test_default_timeout_is_two_seconds` (config with no `timeout_seconds` → 2.0; brief `_timeout({})` → 2.0).
  - `test_brief_is_wall_clock_bounded`: `score_batch` stubbed to sleep 5 s → `reputation(...)` returns
    `None` in < 3 s.
  - `test_busy_brief_lock_does_not_block`: hold `cairn-reputation.lock` with `fcntl.flock` in the test,
    call `reputation` with a fast stub → returns scores promptly, no cache row written.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement**: `_urlopen = loam_http.urlopen` (import with the same `sys.path` handling the
  hooks already use); default 2.0 in `_timeout` and in `cairn_config` defaults; wrap `_fetch` in
  `loam_http.bounded(..., _timeout(cfg), (None, "timeout"))`; open the brief lock with `blocking=False`.
- [ ] **Step 4: Run the Cairn tests and the full suite.**
- [ ] **Step 5: Commit** — `Cairn: 2 s wall-clock brief, non-blocking brief lock, ALCF proxy aware`

### Task 3: FALDA config block, on by default

**Files:**
- Create: `.loam/adapters/hermes/plugin/falda_config.py`, `.loam/falda.json`
- Modify: `.loam/adapters/hermes/plugin/loam_config.py` (`LoamProfile`, `load_profile`, `payload_values`),
  `.loam/hooks/run_manifest.py` (`_falda_summary`, campaign `falda`, artifact `falda_calls`)
- Test: `tests/test_falda_config.py` (port from the FALDA branch and adapt), `tests/falda_fake.py` (port;
  update `falda_block` to the new fields)

**Interfaces:**
- Produces:
  - `falda_config(value: Any) -> tuple[dict[str, Any] | None, str]` — mirrors `cairn_config`: `LOAM_FALDA`
    in `{0,false,off,no}` → `(None, "disabled")`; `value is None` → read `.loam/falda.json`; returns the
    validated block (all Global Constraints defaults filled) and status `"enabled"`, `"disabled"`, or
    `"rejected:<reason>"` (`missing_pool` for a non-UUID pool, `missing_base_url`, `bad_switch`).
  - `LoamProfile.falda: dict | None`, `LoamProfile.falda_status: str = "disabled"`.
  - Payload keys `loam_falda` (the block) and `loam_falda_status`.
  - `.loam/falda.json`: `{"enabled": true, "base_url": "https://falda.cairnscore.ai", "pool":
    "efb99ee4-c412-11f1-8af6-0affe8db6a05"}`.
- Populate `falda` wherever `cairn` is populated in `loam_config.py` (both the file-profile and
  environment-override paths if Cairn is in both; otherwise match Cairn exactly).

- [ ] **Step 1: Port and adapt tests**: keep the FALDA branch's validation tests that still apply;
  replace "absent block → disabled" with `test_absent_block_uses_repo_default` (→ enabled, pool
  `efb99ee4-…`, `recall_timeout_seconds == 2`, `reasoner_tools is True`); add
  `test_env_switch_disables` (`LOAM_FALDA=0`) and `test_profile_block_wins`
  (`{"enabled": false}` → disabled); drop `tenant_env` and workflow-id-length tests; add a run-manifest
  test that `campaign.falda.status == "enabled"` and `artifacts.falda_calls == "falda-calls.jsonl"`.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** per Interfaces; `_falda_summary` takes `workflow_tag` from Task 6's
  `falda_memory.workflow_tag` (until then, `f"workflow:{workflow_id}"`; Task 6 swaps it).
- [ ] **Step 4: Run the FALDA config tests and the full suite.**
- [ ] **Step 5: Commit** — `Loam config: FALDA block on by default from .loam/falda.json`

### Task 4: FALDA machine keys

**Files:**
- Create: `.loam/hooks/falda_keys.py`
- Test: `tests/test_falda_keys.py`

**Interfaces:**
- Consumes: `loam_http.urlopen`.
- Produces:
  - `key_path(cfg: Mapping) -> Path` — `<key_dir>/loam-<pool>.json` (expanded).
  - `resolve_key(cfg: Mapping) -> str` — `os.environ[cfg["token_env"]]` if non-empty, else `api_key`
    from the key file if its `pool` matches, else `""`. No network. Never raises.
  - `ensure_key(cfg: Mapping, timeout: float) -> tuple[str, str]` — `(key, status)`; status one of
    `env`, `cached`, `minted`, `minting_elsewhere`, `cooling_down`, `mint_failed:<reason>`, `disabled`
    (`auto_mint` false). Mint label `loam-<short hostname>-<user>`, body `{"pool", "label"}`.
  - Exclusivity: reserve `<key_path>.lock` with `os.open(O_CREAT|O_EXCL|O_WRONLY, 0o600)`; a lock older than
    600 s is stale and may be replaced. Key file written to a temp file in the same dir (`0600`) then
    `os.replace`. Directory created `0700`.
  - Cool-down file `<key_path>.mint-failed` (JSON `{"at", "reason"}`): skip minting for 600 s, or
    3,600 s after `rate_limited`. Never stores the key.
  - CLI: `python falda_keys.py --print-key --config-json <json>` prints `resolve_key` (used by Task 8).

- [ ] **Step 1: Write the failing tests** (fake transport via monkeypatched `loam_http.urlopen`):
  `test_env_wins`, `test_cached_file_used_when_pool_matches`, `test_cached_file_for_other_pool_ignored`,
  `test_mint_writes_0600_file_and_posts_pool_and_label`, `test_held_lock_skips_without_waiting`
  (pre-create the lock; `ensure_key` returns `minting_elsewhere` in < 0.5 s; no request made),
  `test_stale_lock_replaced`, `test_failure_sets_cooldown_and_429_sets_long_cooldown`,
  `test_concurrent_mints_one_request` (8 `multiprocessing` processes, fake counts requests via a file → 1),
  `test_no_key_material_in_cooldown_or_lock_files`.
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** per Interfaces.
- [ ] **Step 4: Run the key tests and the full suite.**
- [ ] **Step 5: Commit** — `Loam hooks: FALDA machine keys, minted once per pool under $HOME`

### Task 5: FALDA client, spool and flusher

**Files:**
- Create: `.loam/hooks/falda_memory.py` (port and adapt), `.loam/hooks/falda_flush.py`
- Test: `tests/test_falda_memory.py` (port and adapt), `tests/test_falda_flush.py`

**Interfaces:**
- Consumes: `falda_keys.resolve_key`, `falda_keys.ensure_key`, `loam_http.urlopen`, `loam_http.bounded`.
- Produces (`falda_memory`):
  - Keep the FALDA branch's `config`, `log_event`, `calls_path`, `recall`, `upsert`, `RecallResult`,
    `UpsertResult`, `select_sections`, `cut_to_lines`, `set_transport`.
  - `config(payload)` no longer requires `workflow_id`.
  - `workflow_tag(payload) -> str` — `workflow:<id>` from `payload["workflow_id"]` unless empty or
    `environment-workflow`; then `CAIRN_PROJECT` env; then `loam-<UTC YYYYMMDD>`. Truncate the id so the
    tag is ≤ 128 chars.
  - `_post` takes the key from `resolve_key(cfg)` (missing → `missing_token`), sends
    `X-Falda-Pool`, no tenant header, transport via `loam_http.urlopen`.
  - `recall(...)` is wrapped in `loam_http.bounded(..., cfg["recall_timeout_seconds"], RecallResult(ok=False,
    error="timeout"))`; the timeout is still logged.
- Produces (`falda_flush`):
  - `SPOOL = "falda-spool.jsonl"`, `SENT = "falda-sent.jsonl"`, `STATUS = "falda-status.json"`,
    `LOCK = "falda-flush.lock"`, `BATCH_SIZE = 50`, `MAX_BATCHES = 10`, `MAX_PENDING = 5000`.
  - `enqueue(payload: Mapping, atom: dict, *, plan_id: str, hypothesis_id: str) -> None` — appends
    `{"id": uuid4, "atom", "plan_id", "hypothesis_id", "workflow_id", "run_id", "queued_at"}`; no network.
  - `flush(state_dir: Path, cfg: Mapping) -> dict` — non-blocking `flock` on `LOCK` (busy or unsupported →
    return `{"skipped": "locked"}`); `ensure_key`; send pending rows oldest first in batches via
    `falda_memory.upsert`-equivalent POST `/atoms/upsert` with `cfg["timeout_seconds"]`; record sent ids in
    `SENT`; stop at the first failed batch; if pending > `MAX_PENDING`, record the oldest overflow as
    dropped (in `SENT` with `"dropped": true`) and count it; write `STATUS` (counts, last error, key
    status — never the key); log each batch to `falda-calls.jsonl`.
  - `spawn_flush(payload: Mapping) -> None` — copy of `cairn_flush.spawn_flush` (detached
    `Popen`, `start_new_session=True`, DEVNULL I/O), only when `auto_flush` is true; never raises.
  - CLI `python falda_flush.py --state-dir DIR --config-json JSON [--report]` (for the end-of-campaign
    flush and Task 9).

- [ ] **Step 1: Port `test_falda_memory.py`** (drop tenant-header asserts; add `X-Falda-Pool`
  assert) and add: `test_recall_wall_clock_capped` (transport sleeps 5 s → `ok False`,
  `error "timeout"`, < 3 s), `test_missing_key_is_missing_token`, `test_workflow_tag_fallbacks`
  (id; `environment-workflow` + `CAIRN_PROJECT=exp-1` → `workflow:exp-1`; neither → `workflow:loam-YYYYMMDD`).
- [ ] **Step 2: Write `test_falda_flush.py`**: `test_enqueue_does_no_network`,
  `test_flush_drains_in_batches_and_marks_sent`, `test_failed_batch_leaves_rows_pending`,
  `test_overflow_drops_oldest_and_counts`, `test_busy_lock_skips`, `test_malformed_spool_rows_skipped`,
  `test_status_file_has_no_key`.
- [ ] **Step 3: Run to verify they fail.**
- [ ] **Step 4: Implement** per Interfaces.
- [ ] **Step 5: Run the FALDA tests and the full suite.**
- [ ] **Step 6: Commit** — `Loam hooks: FALDA client with 2 s recalls and a spooled, detached writer`

### Task 6: Campaign wiring (validator outcomes, generation recall, reasoner memory, tool guard)

**Files:**
- Create: `.loam/hooks/falda_campaign.py` (port and adapt)
- Modify: `.loam/hooks/hypothesis_orchestration.py` (`tool_policy`, new `_falda_policy`, `prepare_payload`,
  `record_stop`, `_reasoner_directive` and its two callers), `.loam/hooks/pre-tool.py` (fail closed for
  FALDA tools), `.loam/hooks/run_manifest.py` (use `falda_memory.workflow_tag`),
  `.loam/agents/reasoner.md` (append only the `## Shared campaign memory` section from the FALDA branch)
- Test: port `tests/test_falda_campaign.py`, `tests/test_falda_generation_recall.py`,
  `tests/test_falda_reasoner_memory.py`, `tests/test_falda_tool_guard.py`

**Interfaces:**
- Consumes: `falda_flush.enqueue`, `falda_flush.spawn_flush`, `falda_memory.recall`,
  `falda_memory.workflow_tag`.
- Produces: the FALDA branch's `write_validator_outcome`, `generation_memory`, `reasoner_memory_block`,
  `falda_tool_policy`, `FALDA_TOOL_PREFIX = "mcp__falda__"`, with these changes:
  - `write_validator_outcome` → `enqueue` + `spawn_flush` instead of `upsert`.
  - `reasoner_memory_block` adds the recalled-memory lines only when recall returned non-empty content;
    on failure/timeout/empty it adds nothing (no "Recall was unavailable" line). The "Memory interface"
    tool lines stay whenever `reasoner_tools` is on.
  - `_reasoner_directive(plan_id, completions, registry, payload)` keeps `develop-cairn`'s inlined
    `agent_definition` and adds the memory block as one more part, skipped when empty. Apply the FALDA
    branch's `prepare_payload` and `record_stop` hunks as shown in its diff; do not apply its
    `_validator_directive` or `_prompt_body` hunks.

- [ ] **Step 1: Port the four test files**; adapt asserts that relied on non-ported changes (validator
  directive text, `_prompt_body`) to `develop-cairn`'s behaviour; change outcome-write asserts to read
  the spool instead of the fake's `/atoms/upsert` bodies. Add:
  `test_record_stop_returns_fast_when_falda_hangs` (transport sleeps 5 s; `record_stop` < 0.5 s, one spool
  row), `test_reasoner_block_omits_recall_on_timeout`, `test_cairn_delegation_decoration_unchanged`
  (existing Cairn brief still appended to a reasoner delegation alongside the FALDA block).
- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement** per Interfaces.
- [ ] **Step 4: Run all FALDA, Cairn and orchestration tests, then the full suite.**
- [ ] **Step 5: Commit** — `Loam hooks: FALDA campaign memory on develop (spooled outcomes, capped recalls, reasoner tools)`

### Task 7: Docs

**Files:**
- Create: `docs/falda.md` (operator runbook, same shape as `docs/cairn.md`: default on, opt-out
  `LOAM_FALDA=0` / profile block, pool, key file location and how to back it up / revoke, proxy behaviour,
  `falda-status.json`, end-of-campaign `falda_flush.py` command, what "best effort" means).
- Modify: `README.md` (one line pointing at `docs/falda.md` next to the Cairn line).

- [ ] **Step 1: Write the docs**; every command in them must be one that exists after Tasks 1–6.
- [ ] **Step 2: Commit** — `docs: FALDA operator runbook`

### Task 8: Launcher and Hermes MCP config

**Files:**
- Modify: `run-hermes` (before `exec hermes`), `examples/hermes-config.yaml.example`,
  `.hermes/.loam-home/config.yaml` (add the `falda` MCP entry only)
- Test: `tests/test_run_hermes_falda.py`

**Interfaces:**
- Consumes: `falda_keys.py --print-key`, `loam_http.is_aurora_compute` (via a tiny `python -c` or a
  `--is-compute` flag on `falda_keys.py`).
- `run-hermes` additions, all best effort (`|| true`, no network, no minting):
  - If `FALDA_TOKEN` unset: export it from `falda_keys.py --print-key` (default config from `.loam/falda.json`).
  - Export `FALDA_POOL` from the same config.
  - If the host is an Aurora compute node and none of `https_proxy`/`HTTPS_PROXY` is set: export
    `https_proxy`, `HTTPS_PROXY`, `http_proxy`, `HTTP_PROXY` = ALCF proxy, and prepend
    `localhost,127.0.0.1,::1` to `no_proxy` and `NO_PROXY`.
- MCP entry:

```yaml
  falda:
    url: https://falda.cairnscore.ai/mcp
    headers:
      Authorization: "Bearer ${FALDA_TOKEN}"
      X-Falda-Pool: "${FALDA_POOL}"
    timeout: 5
    connect_timeout: 5
```

- [ ] **Step 1: Write the failing test**: run `run-hermes` with `PATH` containing a fake `hermes` that dumps
  its environment, `HOSTNAME`/`hostname` stubbed to `x4102c5s3b0n0`, a temp `HOME` holding a key file,
  `ALCF_PROXY=0`; assert `FALDA_TOKEN`, `FALDA_POOL`, `https_proxy` and `NO_PROXY` containing
  `127.0.0.1`; repeat with hostname `aurora-uan-0010` → no proxy vars; with no key file → `FALDA_TOKEN`
  empty and Hermes still launched.
- [ ] **Step 2: Run to verify it fails.**
- [ ] **Step 3: Implement** the launcher block and the two config entries.
- [ ] **Step 4: Run the test and the full suite.**
- [ ] **Step 5: Manual check (Review Focus 5)**: with a local Hermes, start it via `run-hermes` once with
  `FALDA_TOKEN=` and once with the MCP URL pointed at `https://127.0.0.1:9/mcp`; confirm Hermes starts and
  a non-FALDA tool call works. Record the Hermes version and outcome in the commit message body.
- [ ] **Step 6: Commit** — `run-hermes: export the FALDA key, pool and (on compute nodes) the ALCF proxy; FALDA MCP entry`

### Task 9: Live smoke test and final verification

**Files:** none committed (results reported to the user).

- [ ] **Step 1: Full suite** — expect all tests pass (baseline 357 + new).
- [ ] **Step 2: Live mint into the Loam pool** (first live use of `efb99ee4-…` with a machine key):
  with a temp `HOME`, run `falda_flush.py` against a state dir holding one synthetic spooled atom tagged
  `workflow:loam-smoke-<date>`; expect `falda-status.json` to show `key: minted`, 1 sent; then
  `falda_memory.recall` for that tag returns the atom within 2 s (report the measured time).
- [ ] **Step 3: Cairn**: with a temp `HOME` and `CAIRN_PROJECT=loam-smoke-<date>`, queue one synthetic
  rating in a state dir, run `python .loam/hooks/cairn_flush.py --state-dir <dir> --base-url
  https://api.cairnscore.ai --key-file <tmp>/agent-loam.key --project loam-smoke-<date> --auto-mint`,
  then the same with `--report`; expect the row `submitted` and a submitted batch in `cairn-status.json`.
- [ ] **Step 4: Failure drill**: set `base_url` to `https://127.0.0.1:9` and repeat a recall and a
  validator `record_stop`; recall returns unavailable in ≤ 2 s; `record_stop` returns in < 0.5 s.
- [ ] **Step 5: Branch hygiene**: `git diff --stat develop-cairn..develop-memory` shows no `.hermes/`
  state other than the one config entry, no `.pyc`, no `.swp`, no keys
  (`git grep -n 'falda_[A-Za-z0-9]\{20,\}' develop-memory` finds nothing).
