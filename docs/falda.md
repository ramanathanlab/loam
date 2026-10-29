# FALDA campaign memory

## What it does

FALDA (`https://falda.cairnscore.ai`) is hosted long-term memory. During a hypothesis campaign Loam:

- writes each validator outcome (supported, contradicted, inconclusive, blocked, with its conditions) as a
  tagged memory;
- recalls the campaign's memories into Jnana's generation goal and into the reasoner's delegation;
- lets the tagged reasoner write its own syntheses with `falda_remember` (the "Shared campaign memory"
  section of `.loam/agents/reasoner.md`). The other roles may not use FALDA tools.

It is strictly best effort. A slow, overloaded or unreachable FALDA never fails a step and adds at most
2 s to it:

- **Writes** cost the agent nothing. A validator stop appends the memory to `falda-spool.jsonl` in the
  state dir and starts a detached flusher. The flusher sends batches of 10 and stops at the first failure.
  Each atom carries its spool row's id, so resending a batch FALDA stored after the client timed out
  updates it instead of adding a copy.
  Rows it could not send wait for the next flush. Past 5,000 unsent rows, the oldest are dropped and
  counted.
- **Recalls** are capped at 2 s of wall clock (`recall_timeout_seconds`). A recall that fails, times out
  or comes back empty adds nothing to the prompt; the next one tries again. The reasoner recalls at most
  once per plan.
- **Reasoner tools** go through Hermes's MCP connection (`timeout: 5` in the Hermes config), so a slow
  FALDA fails the call quickly and the reasoner carries on.

## On by default

The repo ships `.loam/falda.json`, which enables FALDA for every profile with no `falda` block. To turn it
off:

- for one launch: `LOAM_FALDA=0 ./run-hermes`;
- for a profile: `falda: {enabled: false}` in `loam.yaml` (a profile block replaces the default entirely).

A profile block takes these keys (defaults shown):

```yaml
falda:
  enabled: true
  base_url: https://falda.cairnscore.ai
  pool: efb99ee4-c412-11f1-8af6-0affe8db6a05   # required: the shared pool's UUID
  token_env: FALDA_TOKEN                       # env var that overrides the cached key
  key_dir: ~/.falda/keys                       # where minted keys are cached
  recall_generation: true
  recall_reasoner: true
  write_validator_outcomes: true
  reasoner_tools: true
  auto_mint: true                              # mint a machine key on first flush
  auto_flush: true                             # spawn a detached flusher at each validator stop
  recall_budget_chars: 1500
  recall_timeout_seconds: 2                    # wall-clock cap for recalls (agent's path)
  timeout_seconds: 60                          # writes (detached flusher; FALDA takes ~1 s/atom); mint capped at 15
```

## Keys: nothing to do

Loam gets its own key. `run-hermes` gets it at launch (Hermes's FALDA MCP server takes its key only
at start): the cached key if there is one, else it calls `POST /auth/machine` with the pool and a
`loam-<host>-<user>` label and caches the key at `~/.falda/keys/loam-<pool>.json` (0600). This takes
at most `LOAM_FALDA_LAUNCH_BUDGET` seconds (default 6; `0` means cached key only). If no key comes
back, Hermes starts anyway with `LOAM_FALDA_MCP=0`, so the reasoner is not offered FALDA tools; the
flusher mints the same way on its first send, so hook writes still go through. A key
reaches only the pool it enrolled in, so keys are cached per pool. `$HOME` is shared across Aurora nodes,
so every runner reuses one key.

Only one process mints at a time, across nodes: it holds `loam-<pool>.json.lock`, created atomically
(a lock older than 10 minutes is treated as abandoned). The others skip minting and keep their rows for a
later flush. After a failed mint Loam waits 10 minutes before trying again, or an hour after a
`rate_limited` answer (FALDA allows 50 mints/hour per IP, and all compute nodes share the proxy's IP).
Recalls never mint: until a key exists they return nothing. To mint ahead of a launch (optional):
`python .loam/hooks/falda_keys.py --launch-key --config-json "$(…same as below…)"`.

`FALDA_TOKEN` in the environment overrides the cached key. To retire a key, delete the cached file; the
next flush mints a new one.

## Campaign tag

Every memory carries `workflow:<id>` and every recall requires it, so campaigns sharing the pool do not
see each other's memories. The id is the profile's `workflow_id`. If that is missing or the generic
`environment-workflow`, Loam uses `CAIRN_PROJECT`, and failing that `loam-<UTC date>`.

## Network

All FALDA traffic comes from the host running Hermes. On an Aurora compute node with no
`https_proxy`/`HTTPS_PROXY` set, Loam sends its FALDA and Cairn requests through
`http://proxy.alcf.anl.gov:3128` without exporting it (`.loam/hooks/loam_http.py`). On a login node or a
laptop it connects directly. An exported proxy always wins.

`run-hermes` exports `FALDA_TOKEN` and `FALDA_POOL` for Hermes's FALDA MCP server. On a compute node
with no proxy set it also exports `https_proxy`/`HTTPS_PROXY` (the ALCF proxy; plain-HTTP variables
are left alone), and whenever a proxy is in force there it puts `localhost,127.0.0.1,::1`, the node's
own name and `.alcf.anl.gov` in `no_proxy`/`NO_PROXY`, so local and ALCF-internal services stay direct.

Keep `LOAM_STATE_DIR` on node-local disk or a filesystem that supports `flock`: the flushers take a
non-blocking `flock` there and, if it is unsupported, skip sending.

## Checking a run

In the state dir (`$LOAM_STATE_DIR`):

- `falda-status.json`: rewritten by every flush. Shows pending/sent/dropped counts, the last error and the
  key status (`env`, `cached`, `minted`, `minting_elsewhere`, `cooling_down`, `mint_failed:<reason>`).
  Never contains the key.
- `falda-calls.jsonl`: one row per recall, upsert, skipped write and guarded tool call.
- `falda-spool.jsonl` / `falda-sent.jsonl`: the write queue and what has left it (sent, dropped on overflow, or rejected by FALDA: a batch FALDA refuses with a 4xx other than 401/403/408/429 is not retried, so it cannot block the rows behind it).

At the end of a campaign, send anything still pending, or print counts with `--report`. The config JSON
is the run's `loam_falda` block; the defaults are enough when you use `.loam/falda.json`:

```bash
python .loam/hooks/falda_flush.py --state-dir "$LOAM_STATE_DIR" \
  --config-json "$(python -c 'import json,sys; sys.path.insert(0,".loam/adapters/hermes/plugin"); from falda_config import falda_config; print(json.dumps(falda_config(None)[0]))')"
```

## Measured on Aurora (2026-10-30)

Hook code on compute node `x4601c7s0b0n0` (ALCF proxy chosen automatically), via Globus Compute:

- 12 parallel tasks sharing one `$HOME` key dir: exactly one `POST /auth/machine`; every task had a key
  in ≤ 0.43 s.
- One flusher (the normal case: one per state dir): 3 atoms in 3–4 s; recalls of a new query
  0.9–1.6 s, repeated queries ~0.2 s. Cairn: rating submitted, project echo `ok`; brief ~0.1 s.
- 12 concurrent flushers: FALDA stored every atom but answered after 10 s and even 60 s, so atoms carry
  their spool id and a resend updates rather than duplicates. Under that load a new-query recall takes
  ~2.7 s, so most recalls hit the 2 s cap and the agent proceeds without memory.
- A blackholed FALDA costs exactly 2.0 s per recall; a validator-outcome write costs ~0.1 ms.

Full Hermes probes (Hermes and hooks on a laptop, code on Aurora through `mcp__hpc_aurora__run_code`,
launched through `run-hermes` with no FALDA credentials set): the launch-time key let Hermes register
all eight `mcp__falda__*` tools; each run sent both validator outcomes to FALDA (none left pending) and
submitted four Cairn ratings.

Hermes defers MCP tools behind `tool_search` unless `tools.tool_search.enabled: off`, and Loam's role
policies block `tool_search` for the code-generator. The shipped Hermes configs
(`.hermes/.loam-home/config.yaml`, `examples/hermes-config.yaml.example`) therefore turn it off; a Hermes
home without that setting can leave the code-generator unable to reach its HPC `run_code` tool.
