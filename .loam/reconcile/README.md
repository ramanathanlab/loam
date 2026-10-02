# Protein structure metadata pilot

## Candidate contract

A delegated child can opt into the first scientific validation path by returning
exactly one fenced JSON object with `contract_version` set to
`protein_structure_metadata_v1`:

```json
{
  "contract_version": "protein_structure_metadata_v1",
  "source_kind": "uniprot",
  "source_accession": "P00918",
  "source_url": "https://www.uniprot.org/uniprotkb/P00918",
  "claims": {
    "recommended_name": "Carbonic anhydrase 2",
    "gene_name": "CA2",
    "sequence_length": 260
  },
  "uncertainties": []
}
```

Supported source kinds and claims:

| Source kind | Supported claims |
| --- | --- |
| `uniprot` | `recommended_name`, `gene_name`, `sequence_length` |
| `rcsb_pdb` | `experimental_method`, `resolution_angstrom`, `chain_ids`, `ligand_component_ids` |

A completed hook appends shape-valid opt-in payloads to
`$LOAM_STATE_DIR/candidate-results.jsonl` with `candidate_status: pending` when
`LOAM_STATE_DIR` is set. This is not acceptance: candidate fields and summaries
remain untrusted until the reconciler validates them against the authoritative source.

## Manual reconciliation

Run reconciliation explicitly after a controlled session:
Set `STATE_DIR` to the deployment's `LOAM_STATE_DIR` before reconciling so the
outputs remain with the run that produced the candidates.


```bash
python3 .loam/reconcile/protein_structure_metadata.py \
  --candidates "$STATE_DIR/candidate-results.jsonl" \
  --out "$STATE_DIR/validated-findings.jsonl"
```

The output is append-only and writes one decision per claimed field:

- `accepted` — matches the source response and receives a stable `finding_id`;
- `claim_mismatch` — source response differs;
- `authoritative_field_unavailable` — the current source adapter cannot resolve
  that field;
- `source_fetch_failed:<error>` — source could not be retrieved.

The RCSB adapter currently reconciles only entry-core experimental method and
resolution. It deliberately marks chain and ligand component claims unavailable
rather than infer them from a child summary. Those fields require a later
entity/assembly-level RCSB adapter.

## Parent memory projection

Project only accepted validation decisions into a compact per-parent snapshot:

```bash
python3 .loam/reconcile/project_parent_memory.py \
  --decisions "$STATE_DIR/validated-findings.jsonl" \
  --out "$STATE_DIR/parent-memory.jsonl"
```

This command deduplicates by stable `finding_id`, preserves the contributing unit
IDs, and records non-accepted validation statuses separately. It does not read
child summaries or pending candidates, so model prose cannot enter parent memory
through this path.

## Read-only recall and duplicate audit

Set `LOAM_PARENT_MEMORY_PATH` when the current Hermes session should recall a
specific projected snapshot. The existing pre-LLM hook injects only validated
findings for the matching parent session. The pre-tool hook observes
`delegate_task` calls and writes `duplicate-candidates.jsonl` only when a task
explicitly names the same source kind, accession, and supported field as an
accepted finding. It always allows the delegation in this phase; no task is
blocked or rewritten.
