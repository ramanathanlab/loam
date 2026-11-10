# Jupiter integration of develop, 2026-11-10

Reviewed `bc351f0..416228d` (four incoming commits, 38 changed files).
The Jupiter work was checkpointed as `a05745d` before rebase. A recovery ref,
`backup/jupiter-before-develop-rebase-20261110`, preserves that complete snapshot.
Generated Hermes curator/library churn was excluded from the commit and retained
locally. The rebase was performed in a separate worktree before updating the
canonical checkout, avoiding intermediate source/config states in live services.

| Develop commit | Changes | Jupiter assessment |
|---|---|---|
| `89cffcb` | Aurora RFD4 resident-worker route; removes older Aurora launch wrappers and shim | Keep this for Aurora. Jupiter continues to select `rfd4-proteina-jupiter`, its native CUDA CLI, packaged binder/CIF references and `num_workers=0`. Do not substitute the generic Aurora skill into a Jupiter campaign. |
| `7bba5a4` | Scientific execution through `run_batch`; phased `run_code` budget; reviewer capacity | Adopted. Jupiter recipes and the EGFR example now stage with `write_files`, use `run_batch` even for one scientific job, and reserve `run_code` for fallback staging, harvesting and diagnosis. |
| `35f61ac` | Tournament `scheme`, `rounds`, `workers`; generation `batch_size`; config-to-hook forwarding | Adopted without dropping keywords. Requires the Jnana source update below; the prior installed runtime would raise `TypeError`. |
| `416228d` | `write_files`; guards against invented Aurora HPC roots; spilled-summary recovery; stricter provenance parents; inference/controller defaults | Retained the orchestration, provenance and environment-expansion features. Kept Jupiter Argo instead of the site-specific Tara model migration. Adapted controller env inheritance to Jupiter and separated writable staging from read-only inspection roots. |

## Concrete incompatibilities addressed

1. **Old Jnana API.** The installed `aa70a1ba` generation API lacked `batch_size`,
   and its tournament API lacked `scheme`, `rounds`, and `workers`. Updated the
   clean source by fast-forward to
   `4c4cdfb42bf042a462eaf091dfbdf24a2ce34598` from
   `acadev/Jnana`, branch `feature/periodic-hypothesis-tournament`. Its six-file
   update does not change requirements/setup metadata. The existing editable
   environment remains in `software/jnana-engine/.venv`.
   [Pin](jnana-version.json); reproducible update:
   `bash scripts/update_jupiter_jnana.sh` after the base install. No Hermes patch.
2. **Conflicting execution instructions.** Old Jupiter recipes and the EGFR
   example suggested scientific `run_code` calls. The new policy counts four
   calls before a batch and ten total after one has been submitted. Updated the
   instructions to stage scripts/JSONL with `write_files`, submit scientific work
   via `run_batch`, and poll to completion. Retained 6000-second scientific row
   deadlines, 6300-second MCP and 14400-second Hermes outer deadlines. Operator
   installation smoke tests remain separate; no checks were added to preflight.
3. **Writer authority.** Upstream passed additional inspection roots to the new
   writer. Jupiter includes the software tree among these readable roots.
   The public writer action now grants only campaign-tree staging authority;
   adding a readable model/venv tree does not permit overwriting it.
4. **Expected staging refusals.** Invalid paths/inputs on a healthy worker return
   explicit application status, like existing inspection refusals. They do not
   masquerade as an MCP outage and trigger Hermes' transport circuit breaker.
   Pool/worker availability failures still remain MCP errors.
5. **Allocation recovery.** Preserving all old Jupiter config verbatim would
   retain a hardcoded default env file and a 30-minute controller cap. Adopted
   the upstream inheritance idea using `${HPC_ENV_FILE}`, preserving a caller's
   campaign env file when sourcing Jupiter defaults, and allowing two-hour jobs
   within the existing one-node cap. Verified the expansion helper exists in
   stock Hermes 0.20.6. Launchers must export their campaign `hpc_env.sh` path.

## Preserved Jupiter behavior

- Argo base URL `http://localhost:50856/v1` and model/provider settings.
- Stock Hermes 0.20.6; existing delegation/parent identity adaptations in Loam.
- Authenticated Redis exchange and the authenticated login-node internet proxy.
- Slurm/GH200 placement, assigned GPU visibility, and unbounded initial queue
  wait when `HPC_CLUSTER_STARTUP_TIMEOUT=0`.
- Driver test settings: 20 rounds, 12 hours, 100 prompts, 6000-second HPC wait;
  one-node/two-hour test allocations are still explicit campaign settings.
- Isolated software environments, native dependency repairs, packaged Jupiter
  references and manual software audit evidence. RFDiffusion3 remains archived.

## Operational effects for the next campaign

**Restart/recreate MCP and its pool together before testing the new writer.**
Existing servers and workers retain their loaded action classes and advertised
tool schemas. Updating files does not retroactively add `write_files` to them.
This integration did not cancel jobs or restart those live services.

Use a fresh campaign, or review its saved `execution.setup` before reuse. An old
literal instruction to run science inside `run_code` conflicts with the new
batch policy. Keep the Jupiter-specific skill names and source the campaign env
before starting Hermes so allocation recovery inherits the intended resources.

Optional tournament settings under `hypothesis_tournament`:

```yaml
scheme: swiss
rounds: 2
workers: 2
batch_size: 2
```

Omitted fields preserve `random`, 1, 1, 1. Parallel match workers require Swiss
pairings; random matches remain single-worker. Generation batch size is a
separate option. No active campaign's state or scientific evidence was migrated.

The new provenance validator correctly rejects dangling parent hypothesis keys;
older producer summaries that put a global UUID in that local-key field can now
be rejected. The new role guidance clarifies the contract. Existing accepted
evidence was not rewritten. Spilled-summary recovery is best-effort when the
harness supplies `child_summary_full_path`; omission remains compatible.

The invented-path guard currently recognizes Aurora `/flare` spellings. It does
not provide an equivalent early diagnostic for invented Jupiter `/e/project*`
paths. Server-side staging/inspection containment still applies. Do not extend
that regex alone: legitimate read-only software paths must remain usable.

## Verification

- Pre-rebase Jupiter regression baseline: **43 passed**.
- Focused integrated suite: **174 passed, 35 subtests passed**, including new
  staging-boundary/refusal and tournament-keyword regressions.
- Repository suite excluding two installed-service fixtures: **576 passed,
  3 skipped, 101 subtests passed**. Final added runtime/recovery checks:
  **9 passed**. Upstream Jnana: **14 passed, 1 deselected**.
- The deselected upstream fake-pool fixture requests a match that never finishes
  and passes no deadline, so it hangs by the API's documented contract. It was
  replaced for integration verification by a regression against the actual
  Jnana API with Loam-style `max_seconds`; that test passes. Jnana production
  source and upstream tests were not patched. All test responses are fixtures;
  no live LLM tournament or scientific campaign was launched.
- Installed-service fixtures are separate: this rebase does not establish a
  fresh live campaign or a restarted server/pool. The existing UMA/GenSLM access
  and optional PyRosetta runtime blockers remain in the software audit.

Final suite after all integration changes: **576 passed, 3 skipped, 101 subtests passed**. See [structured validation](develop-integration-validation.json).
