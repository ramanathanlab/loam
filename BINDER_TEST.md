# Multi-round de-novo binder-design campaign test (remote Aurora execution)

## Objective

Run one real multi-round Loam/Jnana de-novo **binder-design** campaign that
validates the full remote-execution path end to end on ALCF Aurora:

```text
Jnana generation/ranking (binder hypotheses)
→ hypothesis validators
→ code-generator (fetch target at run time, propose binder, co-fold)
→ remote run_code on Aurora (Chai-1 co-fold)
→ interface metrics (ipTM / ipSAE / pDockQ / pLDDT)
→ validator evidence scores (backed by the real fold)
→ reasoner synthesis
→ next-round Jnana steering
```

This is the binder-design analogue of `CAMPAIGN_TEST.md` (the local arithmetic
integration test). It differs in three ways: execution is **remote** on Aurora
through the HPC MCP server, the science is a **real design task** (so success is
judged on pipeline *integrity* and *real execution*, not on hitting a specific
affinity), and it additionally asserts the pipeline **survives HPC node churn**.

The target used here is the human GLP-1 receptor extracellular domain (UniProt
`P43220`, residues 24-145); substitute any target by changing the goal and the
accession. **The target sequence is fetched at run time and is never hardcoded.**

Do not modify tracked repository files. Use one isolated campaign home/state for
all Loam runtime artifacts, and one stable `LOAM_RUN_ID` for the whole campaign.

## Prerequisites (remote infrastructure)

Unlike the local test, this needs live remote infrastructure. Before starting,
all of these must be green:

1. **HPC MCP server** listening on `:8000`, backed by a healthy EL cluster on an
   Aurora allocation (`skills/hpc-deployment/scripts/mcp_server.py` via
   `set_env_akv.sh`). A bare GET to `http://127.0.0.1:8000/mcp` returns `400`
   (alive), and a `run_code` round-trip returns `returncode 0`.
2. **Inference endpoints** for every tournament/agent role (generation, ranking,
   reasoner, subagents) reachable and answering.
3. **`scripts/campaign_preflight.py` reports `ready: true`** — it probes
   `run_code` on Aurora and every inference role. Run it first:

   ```bash
   python3 scripts/campaign_preflight.py \
     --repo <repo> --hermes-home <hermes-home> --loam-config <loam.yaml> \
     --hpc-env skills/hpc-deployment/set_env_akv.sh \
     --hpc-url http://127.0.0.1:8000/mcp --no-auto-start
   ```

A cold just-started block can report `ready: false` once while its worker
registers; re-run before concluding a failure.

### Node-churn resilience (do not use a toy allocation and expect a long run)

The endpoint provisions PBS blocks through a Parsl provider. On the `debug`
queue a block lives only 30 minutes, so a multi-round campaign *will* outlive its
first block. The allocation knobs in `skills/hpc-deployment/set_env_akv.sh` make
Loam request a replacement node by itself (`HPC_MIN_BLOCKS=1`), and the server
rides out the ALCF-proxy exchange-stream drops (`LOAM_EXCHANGE_GIVEUP_S`) instead
of tearing the cluster down. This test asserts that behavior (see acceptance
criterion 7); it is queue-agnostic — set `HPC_QUEUE=prod` (or `capacity`) with a
longer `HPC_WALLTIME` for a scaled run, `debug` for a short one.

## Required isolated configuration

### Hermes configuration

Enable `loam-hooks` and allow the nested delegation the campaign needs:

```yaml
delegation:
  max_spawn_depth: 2        # coordinator -> hypothesis-validator -> code-generator
```

### Loam profile

Use a remote-execution profile. The critical fields:

```yaml
enabled: true
active_profile: binder-campaign
profiles:
  binder-campaign:
    workflow_id: binder-campaign-v1
    state_dir: /absolute/path/to/state       # the sole LOAM_STATE_DIR
    pre_llm_every: 1
    duplicate_policy: allow_and_audit
    hypothesis_orchestration:
      enabled: true
    hypothesis_tournament:
      enabled: true
      jnana_root: /absolute/path/to/Jnana
      provider: openai
      inference_endpoint: <ranking/judge endpoint>
      model: <ranking model>
      generation:
        inference_endpoint: <generation endpoint>
        model: <generation model>
        omit_temperature: true               # if the model rejects temperature!=1
      initial_hypotheses: 25                  # enough for ranking to discriminate
      matches_per_run: 5
      top_k: 3                                # validators delegated per round
      json_max_tokens: 16384                  # reasoning models need headroom
      state_file: <state_dir>/jnana-memory.json
      reports_dir: <state_dir>/tournament-reports
    execution:
      mode: remote_mcp
      tool_name: mcp__hpc__run_code
      function_name: run_code
      endpoint: hpc-deployment
      input_schema: '{"payload": "str", "workdir": "str|None", "timeout": "float", "venv_path": "str|None"}'
      runtime: Aurora frameworks python 3.12 via EL (loam-aurora-flex endpoint)
      workdir_root: /flare/<project>/<user>/hpc-run/work
      setup: |
        Activate each tool's venv in a bash command inside run_code
        (subprocess.run(["bash","-lc", CMD])); absolute paths only. For the
        decisive fold use run_code SYNCHRONOUSLY (timeout ~600), not run_batch,
        so the .cif + scores.npz are in hand in the same call. Fetch the target
        sequence at run time from the UniProt REST API by accession (the Aurora
        node has http_proxy set); never paste it from memory.
```

## Campaign protocol

Run the coordinator in a **persistent session** (not a one-shot: one-shot exits
after dispatching background validators and cannot complete the
validator → reasoner transition).

For each round:

1. Submit the binder-design research task, e.g.:

   ```text
   Design a candidate peptide binder for the human GLP-1 receptor (GLP1R) and
   assess its interface confidence by co-folding it with the target on Aurora.
   Fetch the GLP1R sequence at run time from the UniProt REST API (accession
   P43220) — never hardcode it; verify the returned accession and length — then
   co-fold a proposed binder + GLP1R with Chai-1 via the HPC run_code endpoint
   and report ipTM and ipSAE for the interface.
   ```

2. Follow the injected Loam directive exactly:
   - delegate one `hypothesis-validator` for every selected Jnana hypothesis in
     one batch, with the required role/plan/hypothesis markers and the configured
     `remote_mcp` execution contract.

3. Each validator MUST delegate one tagged `code-generator` child.

4. Each code-generator MUST, on Aurora via `run_code`:
   - **fetch** the target sequence from UniProt at run time and verify the
     returned accession and length (never a remembered/pasted value);
   - propose or redesign a binder (a *designed* candidate may be carried inline —
     that is legitimate; a *reference* datum like the target must be fetched);
   - co-fold binder + target with Chai-1 **synchronously** and read the real
     interface metrics from the produced `.cif` / `scores.npz`.

5. Each validator MUST return the full report, including an `### Evidence Score`
   in `[-1.00, 1.00]` derived from the real metrics (not a guess).

6. Wait for every selected validator to reach a terminal lifecycle state.

7. Delegate exactly one tagged `reasoner` for the plan. It consumes the
   structured `evidence_score` / `evidence_score_status` and reports, keeps
   empirical scores separate from Jnana Elo, and produces the next-round brief.

8. Wait for the reasoner before the next coordinator turn.

Do not start a separate Jnana tournament inside validator/code-generator/reasoner
sessions; only coordinator turns advance the tournament.

## Round-specific steering requirement

The reasoner brief must evolve the next Jnana input while preserving the target:

- Round 1: establish a baseline — at least one folded binder + target complex
  with real interface metrics.
- Later rounds: keep the target, but use the prior brief to prioritize stronger
  epitope engagement, selectivity, or developability, and to drop designs with
  reliably poor interfaces. Repeated folds of the *same* sequence are
  corroboration, not independent evidence.

## Required artifacts

Inspect only the campaign state directory and the per-fold Aurora run folders:

- `run-manifest.json`
- `jnana-memory.json`
- `tournament-reports/tournament-*.json`
- `subagent-ledger.jsonl`
- `hypothesis-orchestration.jsonl`
- `hypothesis-campaign.jsonl`
- `provenance.jsonl`  — Layer-0 input lineage for every execution call
- on Aurora, per fold: `pred.model_idx_0.cif`, `scores.model_idx_0.npz`,
  `result.json` (and the batch's `status.json` / `done` when `run_batch` is used)

## Acceptance criteria

The campaign passes only when all hold:

1. `run-manifest.json` has `identity.execution.mode == "remote_mcp"`, the expected
   workflow id, and one stable run id.

2. Jnana persisted at least `initial_hypotheses` hypotheses and wrote at least
   one tournament report per round with
   `status == "completed"` and `matches_completed == matches_requested`.

3. For each round, `hypothesis-orchestration.jsonl` contains the selected
   validator starts, matching `validator_completed` records with **non-null,
   in-range `evidence_score`** values whose `evidence_score_status == "reported"`
   — i.e. **backed by at least one `code_generator_completed` execution** (an
   unbacked score is recorded as `null` / `unbacked_no_exec` and fails this
   criterion), and exactly one `reasoner_completed`.

4. **Input integrity (no hardcoding).** In `provenance.jsonl`, the **target**
   sequence is obtained by a `fetched` origin (a UniProt/RCSB/EBI-class source) —
   it must NOT appear as an `inline_only` literal. Designed binder candidates MAY
   appear inline (that is expected and allowed); a long reference datum appearing
   `inline_only` is a hardcoding violation and fails this criterion.

5. **Real execution.** At least one fold produced a real Chai complex
   (`pred.model_idx_0.cif` plus `scores.model_idx_0.npz` with an `iptm` value);
   a run that exits 0 writing no output does not count (the run must be `ok`,
   not merely `complete`).

6. `hypothesis-campaign.jsonl` contains one
   `hypothesis_campaign_round_completed` per round, each including the terminal
   validator results and the reasoner synthesis.

7. **Node-churn resilience.** If the campaign spans at least one PBS block
   turnover (always true on `debug` for a run longer than the walltime), the MCP
   server stayed up across it (`:8000` never went permanently down and the
   process did not exit), the endpoint provisioned a replacement block on its own
   (a new PBS job id), and `run_code` resumed on the new block. The campaign must
   not require a manual server restart to continue.

The scientific outcome (how good the binders are) is **reported honestly, not
required**: weak or neutral interface scores are a valid result. This test
validates that the pipeline produces *real, source-backed, reproducible* metrics
— not that a strong binder was found.

## Failure handling

Stop and report the first unmet criterion. Never manufacture scores, metrics,
lifecycle records, folds, or completed rounds. Classify common failures:

- `fewer_than_N_hypotheses` — Jnana generation failed; check model compatibility
  (e.g. a reasoning model rejecting `temperature != 1`; set `omit_temperature`).
- `incomplete` tournament — a scheduled ranking match did not persist; check the
  ranking model/token budget (`json_max_tokens`).
- `endpoint_unreachable` — the HPC MCP server or EL cluster is down; check `:8000`
  and the endpoint logs. A single transient drop is not an outage (the exchange
  listener rides them out); a sustained one is.
- `hardcoded_target` — the target appears `inline_only` in `provenance.jsonl`:
  the code-generator pasted a remembered sequence instead of fetching it.
- `unbacked_score` — a validator emitted an Evidence Score with no
  `code_generator_completed` execution behind it (recorded `unbacked_no_exec`).
- `empty_fold` — `run_code`/`run_batch` exited 0 but produced no `.cif`/scores.
- `server_died_on_turnover` — the MCP server exited when a block expired instead
  of rebuilding on a fresh node (a node-churn-resilience regression).
</content>
