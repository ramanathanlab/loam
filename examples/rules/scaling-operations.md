# Operating rules for the frozen scaling assay

These are reusable execution instructions, not scientific results or proposal
texts. Preserve the registered numerical settings and report unsupported
mechanisms as untested.

- Use the operator-registered version-1 complete pipeline, summary and intact
  follow-up tools. Do not rewrite a scientific pipeline to fix metadata problems.
- Each assessable selected hypothesis obtains a new matched twelve-candidate
  screen per round: four genuinely distinct rows per conditioning arm. Seeds are
  namespaced by hypothesis, round, condition and replicate. Earlier work supplies
  context, not a substitute for current-round evidence or a reason to rerun a
  completed current-round batch.
- Use `data/make_campaign_manifest.py` through `run_code` in the assigned run
  directory. It computes real paths, hashes and seeds. After preparation succeeds,
  submit its actual returned `inputs_path` in a later model turn. Do not transcribe
  long manifests through the model or invent checksums.
- The first scientific delegation covers preparation, the registered complete
  screen, terminal batch attestation and the registered CPU summary. A metadata-
  only child can correctly return `prepared`; a later child executes that prepared
  plan. Reuse any existing owned screen receipt instead of submitting it twice.
- Poll the real batch directory using the supplied `wait_for_batch.py` helper,
  with an at-most-600-second bound and 15-second checks, then call `batch_status`.
  Return immediately when done/status files are terminal. Do not unconditionally
  sleep 600 seconds or spend many model calls in a tight polling loop.
- Use `make_intact_followup_manifest.py` to select eligible real cysteine-free
  designed sequences and prepare the intact-context funnel. Preserve the declared
  cohort/threshold/control policy and record the absence of eligible survivors.
  Never invent an energy or unverified binder-disulfide topology.
- Use exact artifact-index paths and supplied helper entry points. File paths
  returned by tools are authoritative; do not strip leading slashes or recursively
  scan shared campaign/software trees. After two failed attempts at the same
  metadata preparation, return the error and existing evidence to the validator.
- Distinguish confidence, mapped interface geometry, pose dispersion, minimization
  and static MM/GBSA. Check contact persistence on the same energy pose instead of
  sorting energies alone. Five predicted poses are correlated models, and a single
  minimized frame is not MD. Preserve failed rows, negative evidence and uncertainty.
- Hypothesis claims must cite actual owned executed artifacts. Standard matched
  screens cannot demonstrate mechanisms they did not manipulate. Unsupported custom
  mechanisms need a later discriminating assay, not a relabelled control experiment.
- All scientific execution goes through the campaign MCP. Preserve the assigned
  physical GPU mask; metadata summaries request zero GPUs. Agents do not submit or
  cancel scheduler jobs or change installed software.
