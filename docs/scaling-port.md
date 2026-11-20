# Scaling changes carried into Loam

This branch starts from `ramanathanlab/loam` main commit
`c0597d10f73f8f31094d9beeffc9b9c14e4b0364`, which imported the original streamlined
code and renamed its Python package. The Jupiter working tree and the final
frozen controller were reconciled against that base. The public package and
commands are `loam`; historical launch records retain `loamlite` for traceability.

| Change | Main implementation |
|---|---|
| Select `top_k` independently of active validator `concurrency`; retain queued work | `loam/orchestrator.py`, `loam/config.py` |
| Generation batches, Swiss workers and safe generation-timeout boundaries | `loam/tournament.py` |
| Shared cooldown/jitter and staggered recovery across async roles and threaded Jnana | `loam/rate_limits.py` |
| Bounded, lossless-context hierarchical synthesis and receipts | `loam/synthesis.py`, `loam/history.py`, `loam/agents.py` |
| Atomic, coalesced campaign-local Jnana checkpoints | `loam/jnana_checkpoint.py` |
| Version-aware bounded skill metadata cache | `loam/skills.py` |
| Configurable 60-second MCP initialization and separate agent/tool/request budgets | `loam/config.py`, `loam/execution.py` |
| Native local vLLM serving with the measured parser/context/token profile | `scripts/jupiter_nemotron.py`, `scripts/jupiter_nemotron_api.py` |
| Replica pooling, continuous-batch-compatible routing and bounded bulk API | `loam/inference_pool.py`, `scripts/serve_nemotron_pool.py` |
| One parent allocation, disjoint roles, private Redis and all-node telemetry | `loam/allocation.py`, `loam/allocation_startup.py`, `scripts/run_single_allocation*.py` |
| Shared physical GPUs with separate logical slots; real submitted-row receipts | `skills/hpc-deployment/scripts/hpc_agent.py`, `mcp_server.py` |
| Process-local single-writer registry index; canonical disk/failure checks retained | `skills/hpc-deployment/scripts/hpc_agent.py` |
| Optional process-local controller GC policy | `scripts/run_single_allocation_campaign.py` |
| Fresh source snapshots, input/native hashes, ownership ledger, site paths and node/GPU parameters | `scripts/stage_single_allocation.py`, `loam/scaling.py` |
| Audited 20→256 promotion, preserving the qualified scientific/model settings | `scripts/stage_saturation_scaleup.py` |
| Full scientific stage inventories, allocation costs and report/deck exports | `scripts/scientific_stage_inventory.py`, `build_full_goal_report.py`, `full_goal_*.py` |

The [recipe book](scaling-recipe-book.md) explains settings, timing changes,
measured results, failed approaches and what still needs qualification. The
[Jupiter guide](../JUPITER.md) describes installations and fresh staging.

The current checkout makes scientific data an explicit external input and removes
fixed private-run directory prerequisites from the primary launcher. It
ships the EGFR scientific programs/checksum manifest and native launcher, makes source snapshots part of
staging, and exposes GPU/CPU counts and initialization timeouts. Those packaging
and portability changes are covered by offline tests and fresh-environment
staging checks; they have not been rebenchmarked on a new GPU allocation.
Original historical measurements retain their original source/configuration
hashes and audit status in `examples/scaling/history`.

Original Aurora-specific support files are retained in `archive/original-platform-material` outside
the active skill catalog. Current scientific skills use environment-selected
Jupiter/CUDA installations. Other supercomputers require their own launcher,
software, chemistry and accounting qualification; the density calculator does
not imply hardware-independent optimality.

No campaign evidence, API tokens, TLS keys, model weights, old environments or
large generated reports are committed. The original failed provenance findings
and incomplete saturation goal are documented, not converted into success by
a code change. The empty-output citation guard and priority-scheduling replay
remain undeployed experiments rather than claimed production improvements.
