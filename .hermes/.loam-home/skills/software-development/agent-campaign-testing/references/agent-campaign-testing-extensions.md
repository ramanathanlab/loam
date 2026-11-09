# Campaign-testing extensions (session-derived, 2026-11-06)

Class-level umbrella: `agent-campaign-testing`. Session-level patterns from a 5-validator + 2-reviewer Loam campaign (plan_8fe194c3ea0757ca) that future campaigns should reuse.

## Placeholder-artifact audit format
When child outputs claim co-fold / trajectory / energy results but artifacts are skeletons, audit fields must include explicit boolean/state flags rather than paraphrased summaries. Example verified fields from this session: `cofold_inspected_real: 0`, `dual_domain_retention_confirmed: false`, `free_energy_computed: false`. Quote audit text directly in the evidence ledger rather than summarizing.

## Provenance contract (fenced JSON) template
Every validator report should include a `contract_version: "hypothesis_provenance_v1"` block with `hypotheses` array (`hypothesis_key`, `statement`, `parent_hypothesis_keys`) and `evidence_candidates` array per claim (`hypothesis_key`, `claim`, `source_url`, `stance`: supports/contradicts/neutral/context). A reconstructed/non-existent `plan_...` path is a contradiction finding on verification reliability.

## Workspace / plan-guard behavior
The active-plan guard blocks coordinator-side actions (`cronjob`, `memory`, `loam_record_decision`) with `"The active campaign plan is pending..."`. Treat as real workspace state; record blocked storage/decision in the validator response text; retry storage when workspace opens. The validator's `delegate_task` mechanism continues to work.

## Child-budget cap enforcement
Hard limit: at most 2 `delegate_task` leaves per validator. After cap, only read-only inspection (`inspect_filesystem`, `batch_status`, `list_batches`). User preference: single-tile (`gpus=1`) for Aurora/HPC folding tests.

## Evidence score calibration notes from this session
+0.35 / 0.00 / -0.65 / 0.00 (TASK 2 / 3 / 4 / 5). Report `verified non-result` rather than synthetic positive; never fabricate co-fold / trajectory / energy data; label reliability honestly when file content was not independently re-read.

## Coordinator synthesis order
After all validators finish: independent `evidence-reviewer` subagents; then exactly one `reasoner` (not manual coordinator synthesis before reviewers return). Once a reasoner has started, block further coordinator tool calls that could dispatch duplicates.
