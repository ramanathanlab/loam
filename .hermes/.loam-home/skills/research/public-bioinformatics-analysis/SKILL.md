---
name: public-bioinformatics-analysis
description: Design and execute small reproducible in-silico biology experiments using public sequence or genomics data.
---

# Public Bioinformatics Analysis

Use this skill when a user asks for an in-silico biology experiment, especially one the agent should actually run rather than merely propose.

## First: distinguish a proposal from an execution

1. If the user asks for an experiment *idea*, give a compact design and clearly label it as proposed.
2. If the user asks for an experiment for the agent to do, select or confirm a specific public-data analysis, retrieve the source data, run the calculations, and report real output.
3. Do not substitute a wet-lab experiment when the request implies computational biology. Ask for the analysis class if it materially changes the workflow; otherwise choose a simple, reproducible public-data study.

## A robust starter experiment: bacterial GC content and codon usage

Question: across a deliberately diverse set of annotated bacterial reference genomes, is overall genomic GC composition reflected at synonymous third codon positions (GC3) and in codon preference?

### Data selection

- Prefer stable NCBI RefSeq nucleotide accessions with annotated CDS features (GenBank `gbwithparts` retrieval).
- Pick 3--10 genomes for a starter analysis; for a longer exploratory survey, use roughly 10--20 purposively GC-diverse references and explicitly label the panel as non-random.
- A panel can mix complete genomes and chromosome records only when the exact analyzed accession and scope are reported; do not silently combine unlisted replicons.
- Record organism names, accession versions, database, retrieval timestamp, and the panel-selection rationale in a machine-readable provenance manifest and human-readable report.

### Calculation workflow

1. Fetch each annotated GenBank record from NCBI E-utilities or an equivalent authoritative public source. Cache exact retrieved flatfiles locally before analysis, preserving accession versions and retrieval metadata. For transient public-API throttling, retry politely with backoff rather than silently changing the panel or accepting a partial fetch.
2. Compute whole-genome GC% as `(G + C) / genome length * 100`.
3. Extract each CDS according to its annotated strand/location, not by assuming a forward-strand interval.
4. Skip CDS sequences that are not divisible by three or contain ambiguous bases. Separately report retained CDS and all exclusion categories (ambiguous bases, non-triplet length, missing/invalid translation table); non-triplet annotated CDS are valid QC exclusions and must not be presumed absent.
5. Read the feature's `transl_table`; do not assume genetic code 11. Use it to identify sense codons. This prevents treating recoded codons (for example, TGA under code 4) as stops.
6. Count sense codons across CDSs; exclude translation-specific stop codons from codon-usage and GC3 denominators.
7. Compute GC3 from all counted sense codons: fraction whose third base is G or C.
8. For selected synonymous families, report within-family percentages or RSCU. A codon's raw abundance alone is confounded by amino-acid composition.
9. Independently recompute the headline correlation from the persisted result artifact and assert that every displayed codon-family distribution sums to 100% within floating-point tolerance.
10. If reporting Pearson correlation, label it descriptive and report the number of genomes. Do not make a p-value-driven or general biological claim from a tiny panel.

## Interpretation guidance

- A correspondence between genome GC% and GC3 suggests compositional/mutational bias influences synonymous codon usage.
- It does not, by itself, establish selection, tRNA adaptation, or causation.
- Explain exceptions as potentially due to amino-acid composition, horizontal transfer, gene expression, translational selection, and lineage-specific genome evolution.

## Loam/Jnana hypothesis exploration (when the Loam research profile is active)

Use the tournament as an **advisory hypothesis-discovery and prioritization layer**, separate from empirical validation:

1. Set the Jnana generation goal from Hermes's current `user_message`; it must name the biological mechanism or comparison to explain.
2. If workflow-scoped Jnana state has fewer than two hypotheses, seed it with bounded Jnana internal generation agents (`initial_hypotheses`, default 3), then run the configured pairwise tournament.
3. Persist an immutable tournament report with generation count, hypothesis count, and completed-match count. Treat Elo as a model-mediated prioritization signal only.
4. Do not award evidence credit from generated or ranked hypotheses. Credit still requires explicit, source-backed operator review through Loam reconciliation.
5. Give structured hypothesis generation more completion budget than compact ranking. In the current Jnana integration, use at least 4096 JSON tokens for generation; a 1200-token cap can end in an empty, length-terminated JSON completion.
6. For a multi-turn steering probe, isolate Jnana `state_file`, tournament `reports_dir`, workflow ID, and Loam state directory under a temporary campaign root. Run one bounded live match per turn and inspect immutable reports plus persistent Jnana state independently.
7. The composite pre-LLM hook recalls steering before it runs the current tournament. Regenerate the read-only steering projection after each completed tournament if the next turn must see its Elo update; otherwise that steering snapshot is intentionally one iteration behind.
8. A later ranking turn with at least two persistent Jnana hypotheses must not require `user_message`; a current generation goal is necessary only for an underpopulated state. Verify later reports record `generation.status == "not_needed"`.

For the implementation boundary and live verification recipe, see `references/loam-jnana-hypothesis-tournament.md`, `references/loam-jnana-multiturn-steering-probe.md`, and `references/loam-jnana-multiturn-campaign.md`. For a real local Hermes nested-delegation campaign, run the one-round gate in `references/loam-hermes-local-campaign-preflight.md` before attempting all rounds.

### Multi-turn exploration campaign contract

When the requested biology exploration is explicitly iterative, make every transition observable and fail closed:

1. For each round, require a completed Jnana result (`status == "completed"` and `matches_completed == matches_requested`) before selecting its ranked top three.
2. Resolve every selected hypothesis ID against persistent Jnana state before launching exploration. A ranking report and a later state can otherwise drift or be incomplete.
3. Require a non-empty completed exploration for every selected hypothesis. Do not synthesize a partial top-three pass or award any completion credit when even one exploration is absent.
4. Persist a compact round synthesis, then inject that exact prior-round synthesis into the following tournament goal/context. Bound injected text and per-exploration synthesis excerpts so they cannot crowd out the tournament's structured JSON response.
5. If automatic accounting is explicitly requested without source review, name it `unverified_exploration_credit` and persist it separately from the evidence/credit ledgers. Set `epistemic_status: "unverified"`, `eligible_for_evidence_rating: false`, and `eligible_for_contribution_credit: false`.
6. Only append those unverified operational records after a non-empty round synthesis succeeds, so a failed synthesis leaves no misleading completed-credit trail.
7. Keep this campaign isolated from the active research profile unless the user requests production integration. A deterministic fixture is valid for exercising persistence, credit-label, and next-turn-injection invariants; it does not substitute for a live exploration claim.
8. For live model-backed campaigns, treat empty model completions or malformed tournament JSON as failed work, not as empty findings. Retry only according to a bounded, explicit policy; otherwise retain no operational credit and report the endpoint failure as the blocker.

A future production integration should use actual Hermes subagent completion records as the exploration input, rather than treating arbitrary model summaries as accepted evidence.

## Verification checklist

- Confirm accession versions and that records parsed successfully.
- Report raw scope: genome length, analyzed CDS count, and total sense codons.
- Validate that CDS extraction honors feature orientation and translation table.
- Sanity-check at least two codon-family percentages against their numerator and family total.
- Keep generated data/results distinct from interpretation; never present planned or simulated values as retrieved results.
- When using Loam/Jnana, verify the persisted report separately records generated hypotheses, hypothesis count, and completed matches before describing a tournament as having run.
- If the project has no canonical bioinformatics test target, create a temporary `hermes-verify-*.py` script with `tempfile` outside the repository. Use it to parse generated output, assert expected panel/QC invariants, recompute the key statistic, render a plot to a temporary directory when applicable, and delete the verifier afterward. Report this explicitly as ad-hoc verification, not suite green.

## Pitfalls

- Do not remove the final triplet of every CDS unconditionally: annotated records vary, and this can incorrectly discard a valid sense codon. Classify every triplet through the feature's actual genetic code instead.
- Do not count all G/C-ending triplets before excluding stops; GC3 is defined over coding/sense codons here.
- Do not call three species biological replicates or overstate their correlation.
- Do not rely on an organism-level default genetic code when the record's individual CDS annotation supplies a translation table.
