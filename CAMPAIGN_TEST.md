# Five-round local-execution campaign test

## Objective

Run one real five-round Loam/Jnana campaign that validates the bounded claim:

> The arithmetic mean of `[1, 2, 3, 4]` exceeds `2.4`.

Use this as an integration test of the complete campaign path:

```text
Jnana generation/ranking
→ hypothesis validators
→ local code-generator execution
→ validator evidence scores
→ reasoner synthesis
→ next-round Jnana steering
```

Do not modify tracked repository files. Use one temporary campaign directory for all profile, Hermes-home, and runtime-state artifacts.

## Required isolated configuration

Create a temporary directory containing:

- `loam.yaml` — active Loam profile;
- `hermes-home/config.yaml` — isolated Hermes configuration;
- `state/` — the sole `LOAM_STATE_DIR` for every round.

Use one stable `LOAM_RUN_ID` for the entire campaign.

### Hermes configuration

Enable `loam-hooks` and set:

```yaml
delegation:
  max_spawn_depth: 2
```

The campaign requires nested delegation:

```text
coordinator → hypothesis-validator → code-generator
```

Use the available local OpenAI-compatible endpoint and a model that supports the configured request shape.

### Loam profile

Use this profile shape, substituting the temporary state directory:

```yaml
enabled: true
active_profile: local-campaign
profiles:
  local-campaign:
    workflow_id: local-arithmetic-five-round-v1
    state_dir: /absolute/path/to/state
    pre_llm_every: 1
    duplicate_policy: allow_and_audit
    hypothesis_orchestration:
      enabled: true
    execution:
      mode: local
      runtime: python3.11
    hypothesis_tournament:
      enabled: true
      jnana_root: /Users/msinclair/github/jnana
      inference_endpoint: http://localhost:61612/v1
      model: argo:gpt-5.6-terra
      initial_hypotheses: 2
      matches_per_run: 1
      top_k: 2
      timeout_seconds: 30
      omit_temperature: true
      json_max_tokens: 1024
```

`omit_temperature: true` is required for the configured local model.

## Campaign protocol

Run the coordinator in a persistent session. Do not use a one-shot invocation: one-shot exits after dispatching background validators and cannot complete the validator → reasoner transition.

For each round `1` through `5`:

1. Submit the same bounded research task:

   ```text
   Determine whether the arithmetic mean of [1, 2, 3, 4] exceeds 2.4.
   ```

2. Follow the injected Loam directive exactly:
   - delegate one `hypothesis-validator` for every selected Jnana hypothesis in one batch;
   - include the required role, plan, and hypothesis markers;
   - provide the configured `local` execution contract.

3. Each validator MUST delegate one tagged `code-generator` child.

4. Each code generator MUST perform the calculation locally with Python. Use exact arithmetic where practical:

   ```python
   from fractions import Fraction

   values = [Fraction(n) for n in (1, 2, 3, 4)]
   mean = sum(values) / len(values)
   threshold = Fraction("2.4")
   print({"sum": str(sum(values)), "count": len(values), "mean": str(mean), "threshold": str(threshold), "exceeds": mean > threshold})
   ```

5. Each validator MUST return the complete required report, including:
   - outcome;
   - evidence ledger;
   - credit assignment;
   - `### Evidence Score` in `[-1.00, 1.00]`;
   - next step.

6. Wait for every selected validator to reach a terminal lifecycle state.

7. Delegate exactly one tagged `reasoner` for the plan. The reasoner must consume the structured validator `evidence_score`, `evidence_score_status`, and reports, distinguish empirical scores from Jnana Elo, and produce a brief for the next round.

8. Wait for the reasoner to complete before starting the next coordinator turn.

Do not start a separate Jnana tournament in validator, code-generator, or reasoner sessions. Tagged campaign children are excluded from pre-LLM campaign scheduling; only coordinator turns advance the tournament.

## Round-specific steering requirement

The reasoner brief must evolve the next Jnana input while preserving the bounded task.

- Round 1: establish the exact arithmetic result.
- Rounds 2–5: retain the arithmetic task, but use the prior reasoner brief to prioritize independent checks, representation/threshold boundary checks, or explicit contradiction attempts.
- Do not claim independent support from duplicate calculations. Repeated execution of the same arithmetic expression is corroboration, not independent evidence.

## Required artifacts

Inspect only the campaign state directory after completion:

- `run-manifest.json`
- `jnana-memory.json`
- `tournament-reports/tournament-*.json`
- `subagent-ledger.jsonl`
- `hypothesis-orchestration.jsonl`
- `hypothesis-campaign.jsonl`

## Acceptance criteria

The campaign passes only when all conditions hold:

1. `run-manifest.json` has:
   - `identity.execution.mode == "local"`;
   - the expected workflow ID;
   - one stable run ID;
   - lifecycle records for the campaign.

2. Jnana has at least two persisted hypotheses and five coordinator tournament reports.

3. Every accepted coordinator tournament report has:

   ```text
   status == "completed"
   matches_completed == matches_requested
   ```

4. `hypothesis-orchestration.jsonl` contains, for each round:
   - selected validator starts;
   - matching `validator_completed` records;
   - non-null, in-range `evidence_score` values;
   - exactly one `reasoner_completed` record.

5. `hypothesis-campaign.jsonl` contains five:

   ```text
   event == "hypothesis_campaign_round_completed"
   ```

   Each record must include the terminal validator results and reasoner synthesis.

6. `subagent-ledger.jsonl` proves that each validator delegated a code-generator child and that each code-generator child completed local execution rather than being interrupted or blocked.

7. The reasoner brief from rounds 1–4 appears in the following coordinator turn's Jnana generation goal. Confirm this from the persisted state/reports or captured coordinator context.

8. The final conclusion is supported by the observed local calculation:

   ```text
   sum = 10
   count = 4
   mean = 5/2 = 2.5
   threshold = 12/5 = 2.4
   5/2 > 12/5
   ```

## Failure handling

Stop the campaign and report the first unmet condition. Do not manufacture scores, syntheses, lifecycle records, or completed rounds.

Classify common failures precisely:

- `fewer_than_two_hypotheses` — Jnana generation failed; inspect model compatibility and Jnana state.
- `incomplete` tournament — a scheduled Jnana ranking did not persist; inspect the ranking task/model response.
- validator or code-generator interruption — the persistent coordinator did not remain alive through nested delegation.
- missing score — validator report did not use the required `### Evidence Score` section.
- child-local tournament report — campaign-child session gating regressed.

Do not count a partial round toward the required five rounds.
