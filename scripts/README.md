# Campaign tooling

Start with the [scaling recipe book](../docs/scaling-recipe-book.md). Run Python
helpers with the installed controller environment and `PYTHONPATH=.:scripts`.

## Current combined-allocation workflow

- `recommend_scaling.py`: print a node/GPU-density starting recipe; no staging or submission.
- `stage_single_allocation.py`: stage an explicitly supplied external science bundle, fresh state, an immutable
  control-source copy, native serving profile and ownership ledger. Only `--submit`
  invokes `sbatch`, once for the parent allocation.
- `prepare_scaling_inputs.py`: validate/copy an approved external scientific bundle;
  it never downloads/publishes data or submits jobs.
- `check_single_allocation.py`: inspect paths, role/node counts, thinking profiles,
  source/input seals and any existing submission receipt; no submission.
- `run_single_allocation.py`: parent batch coordinator. Starts private Redis,
  telemetry, disjoint inference/science steps, MCP and controller; terminates only
  its own children. Invoke through the generated job script inside Slurm.
- `run_single_replica.py`, `jupiter_nemotron.py`, `jupiter_nemotron_api.py`: native
  vLLM replica, private API/readiness and parent-role checks. The coordinator uses
  the native `serve` path; other native launcher subcommands retain historical
  integrations and are not the combined-job entry point.
- `serve_nemotron_pool.py`: authenticated pool gateway and bounded batch API.
- `run_single_allocation_campaign.py`: frozen EGFR tool registration, optional
  process-local GC policy and the tournament/evidence/synthesis loop.
- `stage_saturation_scaleup.py`: original audited 20→256 promotion gate;
  `--check-only` is read-only. Requires the complete external qualification lineage.
- `jupiter_env.sh`: site software/cache/library environment. `LOAM_SOFTWARE_ROOT`
  selects installations; `LOAM_CC`, `LOAM_CXX`, `LOAM_LD` and
  `LOAM_REDIS_LIBRARY_PATH` can select site compiler/runtime paths.

`examples/scaling/run-20i.sh` and `run-256c.sh` select the measured admission
settings while creating fresh jobs with the current code. They stage by default.
Their historical original launch scripts, operational setting extracts and audit status are retained in
`examples/scaling/history`.

## Verification and reports

`audit_scale20_campaign.py`, `verify_saturation_tournament.py` and
`validate_synthesis_repair.py` check scientific coordinates/provenance,
tournament work and synthesis receipts. `audit_saturation_campaign.py` measures
whole-parent GPU/throughput gates; `saturation_accounting.py` retains allocated
requeues and failed attempts. The historical hardware integrators require
four-GPU Jupiter telemetry and must be adapted/qualified for a different layout.

`scientific_stage_inventory.py` witnesses RFD backbones, sampled MPNN sequences,
folding input sets versus pose multiplicity, minimizations, static MM/GBSA and
actual integration steps. Its scientific schema is the frozen EGFR assay.
`collect_report_batch_budgets.py` collects observed row deadlines from owned
receipts. Both are read-only with respect to campaign evidence and require new
output paths.

`build_saturation_report.py` and `build_full_goal_report.py` export offline HTML,
figures, CSVs, editable slides and PDFs from independently audited data. The full
20→256 study renderer also needs scientific stage inventories, budget inventory
and metric caches under the experiment's diagnostics directory. The supporting
`full_goal_*.py`, `prepare_report_metrics.py` and `verify_full_goal_report.py`
retain failed/partial qualification status; they do not upgrade numerical
outputs to accepted scientific claims. Install `.[reports]` for these tools.
Old raw campaign directories/report exports are external artifacts, not Git data.

## Historical experiment helpers

The `*_inference_balance*`, replica grow/retire/renew, attached-campaign,
conference-slide and old study-report helpers preserve earlier workflows. They
require the corresponding retained ownership ledger, reference assets, service
records and measured campaign data; some defaults name those dated experiments.
They are not a fresh-checkout deployment path or a prerequisite of the current
single-allocation launcher. Never redirect them at another campaign's jobs.
The original split-allocation ratio study and the later combined-allocation
campaigns have different accounting scopes, as the recipe book explains.

`inference_tunnel.sh`, `globus_compute_loam_env.sh` and
`render_globus_endpoint_config.py` support the separate external-provider/managed
Globus route. The renderer only writes a new configuration directory. The
combined internal inference/science job needs none of those external services.
