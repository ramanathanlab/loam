# Chai-1 folding test on Aurora XPU

## Validated configuration

- Environment: `/flare/FRAME-IDP/agentic_campaign_09_26/binder_design/envs/chai-2026-venv`
- Import: `chai_lab`
- Entry point: `chai_lab.chai1.run_inference`
- Accelerator selection: `device="xpu:0"`
- Availability probe: `torch.xpu.is_available()`; do not use CUDA availability as the Aurora/XPU signal.
- Preexisting model assets were found under the Chai source tree's `downloads/` directory (about 7 GB at the time of validation).

## Chai FASTA grammar

Chai requires an entity type and a label in each FASTA header. For a single protein:

```fasta
>protein|name=example_01
MKT...
```

A bare header such as `>example_01` fails in `chai_lab.data.dataset.inference_dataset.read_inputs` with `ValueError: <name> is not a valid entity type`.

## Minimal per-row runner pattern

Each JSONL row should carry at least `id`, `sequence`, and a reproducible `seed`. The runner writes `input.fasta`, calls `run_inference`, then writes `result.json` containing sequence metadata, device, test configuration, candidate count, and a recursive list of output files.

For a low-cost functionality test:

```python
run_inference(
    Path("input.fasta"), output_dir=Path("chai_output"),
    use_esm_embeddings=True, use_msa_server=False,
    num_trunk_recycles=1,
    num_diffn_timesteps=20,
    num_diffn_samples=1,
    num_trunk_samples=1,
    seed=row["seed"], device="xpu:0", low_memory=True,
)
```

This validates the actual XPU inference route but is not a production-quality prediction protocol. Increase inference settings and consider MSAs for scientific use.

## Batch outcome pattern

A 12-sequence, 80-residue deterministic-random test completed successfully after correcting FASTA headers:

- failed formatting attempt: `b-fd238f65025e`
- successful retry: `b-41cdf9439923` (12 succeeded; 0 failed)
- results root: `/lus/flare/projects/FRAME-IDP/avasan/hpc-run/batches/b-41cdf9439923`

### Tile semantics (important)

`run_batch(gpus=...)` specifies the number of Aurora GPU/XPU tiles **per JSONL row**, not the total tiles for the batch. Thus, to fold 12 independent sequences simultaneously with one XPU each, create 12 rows and submit with `gpus=1` (provided 12 tiles are available). Passing `gpus=12` requests 12 tiles for *each* sequence; it is inappropriate for this single-XPU Chai runner and can delay concurrency through scheduling pressure.

Do not describe `gpus=1` as “one tile for the entire 12-row batch.” It is a one-tile-per-task request; the scheduler may run all 12 rows concurrently across 12 tiles if capacity permits, or queue/time-slice them if not. To determine achieved concurrency, use task/scheduler evidence rather than inferring it from the request or from a few early completions.

When the user asks for a 12-tile-first test, first submit 12 rows with `gpus=1`, label the run as `12-concurrent-single-tile-runs`, and report it as submitted until scheduler/task status demonstrates the actual state. Verify successful outputs only after `complete: true` and, where needed, inspect a `result.json`.

Use separate batch IDs for retries; report both when the correction is material to interpreting the run.

## Verified per-task timing and placement

For batch `b-684e9ae96a58` (12 random 80-residue inputs, Chai test settings above), the per-task `status.json` records showed:

- each task completed in **47.7–48.8 seconds** (mean/median **48.25 seconds**; 579.0 seconds summed over 12 tasks);
- all tasks ran on `x4219c4s2b0n0`, with GPU IDs spanning `0` through `11`.

This is the correct evidence to answer both “how long does a single task take?” and “did the batch occupy multiple tiles?” The batch-level creation/submission timestamp and later `batch_status` polls only provide coarse wall-clock upper bounds, not exact completion times. Read `run_<id>/status.json` across all rows to compute duration statistics and enumerate hosts/GPU IDs.

## Capacity-wave validation: 24 tasks on 12 Aurora tiles

A follow-up batch verified the expected capacity behavior using 24 independent 80-residue random inputs and the same lightweight Chai settings:

- batch: `b-e938f496cce8`
- request: 24 JSONL rows with `gpus=1` (one tile per task)
- outcome: 24 succeeded; 0 failed
- placement: GPU IDs `0`–`11`, with **exactly 2 tasks assigned to each tile**
- per-task durations: 44.6–48.8 seconds (mean 47.68 s; median 48.0 s)
- aggregate compute: 1,144.4 seconds
- expected compute makespan: two waves, approximately `2 × 48.8 = 97.6` seconds, before scheduler, process-launch, and output-finalization overhead.

For capacity predictions, distinguish the **per-task run time** in `status.json` from end-to-end wall time. Do not use the interval between submission and an arbitrary later status poll as a precise benchmark. When a workload has more rows than available tiles, estimate compute makespan as `ceil(rows / confirmed_concurrent_tiles) × max(per-task seconds)`; then state that scheduler overhead is additional and may be material.
