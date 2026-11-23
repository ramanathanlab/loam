# What stays on `moeen`

Final cleanup for [#62](https://github.com/ramanathanlab/loam/issues/62), under
[#43](https://github.com/ramanathanlab/loam/issues/43). Compared with `develop`
at `cc20912147426b3e53898b6e52f8541e23f784d2`; the source snapshot is
[`moeen` at `9b8357436dc46f8c59b3eb8f1a790ced533a548e`](https://github.com/ramanathanlab/loam/tree/9b8357436dc46f8c59b3eb8f1a790ced533a548e).
The pinned snapshot keeps this inventory meaningful after the branch is retired.

## Deliberately left behind

The following are historical experiment tools and records, not requirements for
a fresh campaign on `develop`. Paths below refer to the pinned source snapshot.

| Material | Retained on `moeen` |
|---|---|
| Reports, figures and slides | `scripts/build_*`, `full_goal_*`, `prepare_report_metrics.py`, `collect_report_batch_budgets.py`, `report_production_events.py`, `plot_*`, `saturation_*_plots.py`, `saturation_control_plane_figures.py`, `scaleup_slides.py`; conference preparation, rendering and report-verification helpers |
| Run-specific telemetry and inference-balance studies | `scripts/start_inference_telemetry.py`, `watch_science_telemetry.py`, `watch_saturation_trial.py`, `snapshot_saturation.py`, the `*_inference_balance*` scripts, and the corresponding job/service records held outside Git |
| Probes, profiles and scientific audits of particular runs | `scripts/probe_*`, `profile_*`, `benchmark_loam_nemotron.py`, `audit_scale20_campaign.py`, `audit_claim_resolutions.py`, `scientific_stage_inventory.py`, `qc_saturation_sample.py`, claim-source review/resolution helpers, `verify_conference_science.py`, `verify_saturation_tournament.py` and `validate_synthesis_repair.py` |
| Historical launch and service wrappers | `scripts/run_attached_campaign.py`, `hold_science_server.py`, `release_completed_science.py`, `renew_inference_at_checkpoint.py`, `preflight.py`, `inference_tunnel.sh`, `globus_compute_loam_env.sh`, `render_globus_endpoint_config.py`; the original launcher subcommands and their dated defaults |
| Archived platform material | `archive/` as a directory, including `archive/original-platform-material/`. Reusable Aurora instructions were mined into the per-System skill references; the archive container is not needed at runtime. |
| Scaling history | `examples/scaling/history/`: original 20i/256c job scripts, settings and provenance, the old staging-script snapshot, native runtime versions and result records. The measured numbers and their limitations survive in the [scaling recipe book](scaling-recipe-book.md). |
| Campaign-specific inputs and qualification | `examples/scaling/egfr/` as a frozen campaign bundle, its input manifests and scientific protocols; `examples/scaling/run-20i.sh`, `run-256c.sh` and the personal Nemotron configuration. Reusable scientific runners were adapted into the tool skills. External research inputs, model weights and raw run/report directories are not Git data. |

The EGFR-specific registration/smoke workflow and its `tournament-audit.json`
and `scientific-protocol.json` qualification gates were not generalized by the
staging slice. Historical Nemotron configure/run/probe integrations and the
Apptainer route also remain there. These omissions do not remove the existing
MCP `register_tool`/`list_tools` API or its output contracts on `develop`.

## Functionality ported into modules

An old filename staying on `moeen` does not mean its reusable behavior was lost:

| Source scripts | Supported implementation on `develop` |
|---|---|
| `stage_single_allocation.py`, `check_single_allocation.py`; external-input validation | `loam.staging` (`loam-stage`), with campaign paths and seals checked by `loam.allocation` |
| `run_single_allocation.py`, `run_single_allocation_campaign.py` | `loam.coordinator`, configured from the campaign and Site profile |
| `sample_jupiter_telemetry.py` | `loam.telemetry`, using the System's GPU telemetry adapter; this reusable sampler is distinct from run-specific watchers |
| `audit_saturation_campaign.py`, `saturation_accounting.py`, `stage_saturation_scaleup.py` | `loam.saturation` (`loam-saturation`, `loam-scaleup`) |
| `recommend_scaling.py` | `loam.scaling` (`loam-scaling`), with the measured-System guard |
| `serve_nemotron_pool.py` | `loam.inference_pool` (`loam-inference-pool`), accepting OpenAI-compatible backends |
| Native replica/relay code in `jupiter_nemotron.py`, `jupiter_nemotron_api.py`, `run_single_replica.py`, `loam_nemotron_relay.py`, `add_nemotron_replicas.py`, `retire_nemotron_replicas.py` | `loam.nemotron` (`loam-nemotron`) through the optional `nemotron` extra; ALCF refresh is conditional through `loam.token_refresh` |
| `jupiter_env.sh` | Worker initialization and software/cache/toolchain settings in an external Site profile |

## Test and skill cleanup

The test configuration puts only the checkout root on the Python path; CI runs
the suite without the old root `scripts/` directory. The autouse synthetic
staging fixture from `moeen/tests/conftest.py` is absent. Tests use explicit
fixtures and import the campaign modules from `loam`.

The shared deployment payload under `skills/hpc-deployment/scripts/` remains:
it is the actual MCP server/agent implementation, not the discarded root
`scripts/` collection. Its tests still load that bundled payload.

Every `skills/**/SKILL.md` is now discovered by the layout check, including
`hpc-deployment`. Its obsolete, personal-path Chai probe
`skills/hpc-deployment/test/run_via_mcp.py` is removed; use the generic deployment
smoke client and the Chai skill's own workload instead.

## Landing

After the final cleanup slice lands and CI passes, post the deliberately-left-
behind inventory on #43, referencing this document and the pinned source
snapshot. Retire `moeen` only after that handoff; this cleanup does not delete or
rewrite the source branch.
