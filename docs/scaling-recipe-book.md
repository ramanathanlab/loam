# Loam scaling recipe book

This branch carries the Loamlite Jupiter scaling work into the renamed `loam`
package. Start here for a new campaign. The original study ran in November 2026
on Jupiter GH200 nodes with **four physical GPUs and 288 CPU cores per node**.
The numbers below are measured operating points for the supplied EGFR assay,
not a hardware-independent optimum.

## What actually worked

| Case | Inference : science nodes | Active validators | Selected per round | Whole-allocation GPU busy | Distinct screens | Full-context trials | Full-context trials / allocated GPU-hour | Qualification |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 20b, job 2952205 | 8 : 12 | 12 | 48 | 26.66% | 1,304 | 272 | 1.049 | Passed |
| 20e, job 2961727 | 8 : 12 | 48 | 96 | 38.71% | 2,616 | 528 | 1.711 | Passed |
| 20g, job 2972192 | 8 : 12 | 48 | 144 | 47.96% | 4,722 | 764 | 1.987 | Passed |
| **20i, job 2979745** | **8 : 12** | **48** | **144** | **51.06%** | **4,858** | **819** | **2.120** | **Best qualified 20-node recipe** |
| 20j, job 2991226 | 8 : 12 | 48 | 144 | 48.35% | 5,234 | 871 | 1.954 | Passed; later controller fixes |
| 256a, job 2976829 | 102 : 154 | 616 | 1,848 | 24.87% | 16,994 | 1,870 | 0.733 | Retired partial campaign; requeues charged |
| 256b, job 2987415 | 102 : 154 | 616 | 1,848 | 32.30% | 22,700 | 2,890 | 0.934 | Retired partial campaign; input/provenance findings |
| **256c, job 2999784** | **102 : 154** | **616** | **1,848** | **40.51%** | **61,134** | **10,337** | **1.371** | **Three rounds completed; claim-provenance qualification failed** |

GPU busy means the time integral of sampled NVML `utilization.gpu` over **all
allocated physical GPU-seconds**, with missing intervals charged zero. Startup,
round barriers, idle/drain time, failed work and requeues remain in the cost.
This is neither SM occupancy nor FLOP efficiency. The final 256-node run had
99.70% telemetry coverage, 65.95% inference GPU busy and 23.66% science GPU busy.
Its 40.51% allocation-wide result **did not meet the 50% target**.

20f briefly achieved 51.18% allocation-wide busy but failed reasoner output
completion. It is not the best qualified recipe. Controller caching/GC/MCP
repairs carried into 20j and 256c improved specific CPU and initialization
problems; 20j did **not** beat 20i's GPU utilization or normalized science rate.
These adaptive campaigns are not a controlled fixed-work strong-scaling test.

A screen is a distinct, receipt-owned input/seed trial. In 256c, the independently
checked pipeline yielded 61,134 RFD backbones, 61,134 sampled MPNN sequences,
305,670 crop cofold models, 51,685 full-context cofold models, 10,337 minimizations
and 10,337 single-frame MM/GBSA evaluations. Five correlated predicted poses per
input do not represent five independent binders. There were 61,134 unique screen
sequences and 9,305 unique full-context sequences. **Production MD trajectories:
zero.** The energy route minimizes and evaluates one frame; it does not integrate
MD. Confidence/geometry/static energies are computational screening measurements,
not experimentally measured affinity.

The native final audit found zero numerical errors but 1,434 unresolved citation
sources, including at least one claim citing an empty own run directory. Some
others are path/alias resolution issues; they are not all false scientific
claims. Numerical output, orchestration completion and claim qualification are
separate checks. The original findings are retained, not repaired retrospectively.

## Size the useful-work backlog separately from admission

Let:

- `N` = total nodes; `I` = inference nodes; `S = N - I` = science nodes.
- `G` = physical GPUs per science node.
- `P` = logical science workers per physical GPU; the measured dense recipe uses 2.
- `W = S * G * P` = logical scientific row slots, **not physical GPUs**.

The measured dense starting point is:

| Setting | Formula | 20 nodes, G=4 | 256 nodes, G=4 |
|---|---|---:|---:|
| Inference : science | `I ≈ round(0.4N)`; `S=N-I` | 8 : 12 | 102 : 154 |
| Physical science GPUs | `S*G` | 48 | 616 |
| Logical science workers | `W=S*G*2` | 96 | 1,232 |
| Active validators, `concurrency` | `C=ceil(W/2)` | 48 | 616 |
| Selected hypotheses, `tournament.top_k` | `K=3C` | 144 | 1,848 |
| Initial pool, `initial_hypotheses` | `2K` | 288 | 3,696 |
| Later fresh hypotheses, `new_per_round` | `ceil(K/2)` | 72 | 924 |
| Hypotheses per generation request, `gen_batch_size` | 4 | 4 | 4 |
| Swiss rounds | 1 | 1 | 1 |
| Parallel ranking workers | `2I` | 16 | 204 |
| Bounded synthesis concurrency | `I` | 8 | 102 |
| Campaign rounds | 3 | 3 | 3 |

`top_k` determines useful work selected for the whole round. `concurrency` bounds
simultaneously active validator/reviewer threads with an async semaphore. A
completed thread admits the next selected hypothesis immediately. The old
`ranked[:concurrency]` truncation was removed: a three-wave backlog now survives
selection. Increasing `top_k` alone does not raise simultaneous admission.
Each hypothesis can submit multiple scientific rows; validators are not GPUs.

Generate an estimate for another layout without staging/submitting:

```bash
export PYTHONPATH="$PWD:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
.venv/bin/python scripts/recommend_scaling.py --nodes 64 --gpus-per-node 8
```

This calculator generalizes **workload density**, not demonstrated saturation.
For a new accelerator, memory size, model, target, GPU count or scheduler, qualify
one small layout, measure inference and science service times, then adjust the
ratio. A smaller tensor-parallel group must fit the BF16 weights plus KV cache;
the 120B checkpoint is not a single-GH200 deployment. Validate tensor-parallel
compatibility and memory before attempting a different `G`.

One-node functional debugging needs local/external inference or a different
placement. The combined dedicated-role launcher intentionally requires at least
two nodes. `system: jupiter` currently selects the Slurm/CUDA adapter; PBS/XPU
systems retain their original adapters and require their own software/skill
qualification. A changed `G`/CPU count is configurable for the Slurm adapter, not
proof that its numerical tool chain has been ported to another supercomputer.

## Choose the inference : science split from measured demand

The earlier 20-node ratio study examined 1:19, 2:18, 4:16 and 8:12. Its observed
rates were approximately 674, 1,139, 1,411 and 1,623 coordinate files per science
allocation-hour, with an 8:12 repeat at 1,452. Those historical rates have a
different denominator and scope from the table above. They support **8:12 as
an empirical starting point**, not a universal optimal fraction.

With only one inference node, agents waited while inference was busy and science
GPUs lacked submitted work. The pool now routes across healthy Nemotron replicas,
using in-flight/estimated token work and soft prefix-cache affinity. It also
supports round-robin and least-loaded routing. Native vLLM performs continuous
batching (`max_num_seqs=16`). A bounded `/pool/batch` API accepts independent
nonstreaming requests; regular campaign agents use the ordinary chat endpoint
and obtain continuous batching from their concurrency. No prompt merging,
scientific downsampling or thinking change is needed.

When tuning a new machine, inspect per-role GPU busy, TTFT/request latency,
queue depth, row submission/dispatch times, logical slot occupancy, and stage
production per science-node-hour **and per total allocated GPU-hour**. Increasing
science nodes without a ready science backlog can reduce efficiency. The final
256-node run still had substantial science starvation, round barriers and long
hypothesis tails. Larger scientific batches and rolling/overlapped tournament
groups remain future experiments; neither is credited as a measured improvement.

## Fresh checkout: stage one Slurm allocation

Install the controller separately from scientific/vLLM environments. Jnana is a
local checkout selected by `tournament.jnana_root`; its dependencies and model
access must be available. `requirements-jupiter.txt` records the measured
controller environment. `examples/scaling/history/native-runtime-versions.json` records the
observed native inference packages. These are site observations, not universal
wheel installation recipes.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,hpc]'
export LOAM_SOFTWARE_ROOT=/shared/your/software
export LOAM_SCIENCE_INPUTS=/shared/science/egfr/inputs
export PYTHONPATH="$PWD:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"

# No submission: creates a new job, operator-supplied scientific inputs and source snapshot.
bash examples/scaling/run-20i.sh /shared/scratch/my-new-experiment \
  --account YOUR_ACCOUNT --partition YOUR_PARTITION

# A fresh 256-node recipe, with the final controller's GC policy.
bash examples/scaling/run-256c.sh /shared/scratch/my-new-experiment \
  --account YOUR_ACCOUNT --partition YOUR_PARTITION
```

Set `LOAM_RUN_LABEL` to a fresh label for each attempt. Add `--submit` only when
you intend to submit that newly staged job. Existing staged directories and
submission intents are refused; a lost `sbatch` acknowledgement requires
reconciliation rather than another submission. A stage operation itself never
allocates resources. The generic entry point allows overrides:

```bash
.venv/bin/python scripts/stage_single_allocation.py /shared/scratch/experiment \
  --nodes 20 --inference 8 --gpus-per-node 4 --cpus-per-node 288 \
  --label dense-20 --account YOUR_ACCOUNT --partition YOUR_PARTITION \
  --software-root "$LOAM_SOFTWARE_ROOT" --walltime 08:00:00
```

`--concurrency`, `--top-k`, `--initial-hypotheses`, `--new-per-round`,
`--generation-batch-size`, `--tournament-workers`, `--tournament-max-seconds`,
`--science-workers-per-gpu`, `--controller-gc-threshold0`, `--reference`,
`--model-dir`, `--vllm` and `--inference-profile` are explicit overrides. The CLI
defaults to the density formula; historical 20i leaves the GC policy unchanged.
Read `scripts/stage_single_allocation.py --help` for every option.

The example contains the registered scientific programs and checksums for external
verified three-condition EGFR inputs. The assay runs the RFD4→target-fixed MPNN→Chai
screen and intact-EGFR cofold/minimization/MMGBSA follow-up. A custom `--reference` directory supplies its own `campaign.yaml`,
`goal.md`, `inputs-manifest.json`, external `inputs/` and optional private `seed-hypotheses.json`. The registration entry
point currently implements this EGFR assay; a different scientific assay also
needs its registration definitions adapted and requalified.

The staged job starts private Redis, telemetry, disjoint inference/science
`srun` steps, the inference gateway, MCP science service and controller under
one parent allocation. Source is copied to `infra/control-runtime`; inputs,
source files and native serving sources are hashed. The same step hostnames
must partition the parent allocation. Agents cannot submit secondary scheduler
jobs. Cleanup terminates child processes created by this controller; it does
not cancel someone else's allocation. Queue waiting happens before service
credentials/model-loading clocks start. Private bearer tokens and TLS material
are generated after allocation starts, and never committed.

On Jupiter the measured account was `deepwing`, partition `booster`. The earlier
`bne_deepwing` allocation was a time-limited **reservation**; do not hard-code it
as a default account/partition. Pass `--reservation` only if your reservation is
currently valid. `scripts/stage_saturation_scaleup.py QUALIFIED_20_NODE_JOB
--nodes 256 --label NEW_LABEL --check-only` retains the original, non-submitting
qualification gate for an audited 20→256 promotion. Its reference audit lineage
must be available; the Git repository deliberately does not contain old run data.

## Native serving profile and thinking

The measured local profile is recorded in `examples/scaling/nemotron.json`:

- Checkpoint `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`, revision
  `fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe`.
- One replica per inference node, tensor parallelism across that node's four GPUs;
  GPU-memory fraction 0.85, context 131,072 tokens, `max_num_seqs=16`.
- Language-model-only serving; automatic tool choice, `qwen3_coder` tool parser,
  `nemotron_v3` reasoning parser, local model weights and offline Hub access.
- Validator/code-generator/reviewer/reasoner: `enable_thinking=false`.
  Tournament generation/ranking retain the prior **default thinking behavior**;
  the agent-role flag is not silently applied to Jnana.
- Output caps 8,192 tokens for validator/code-generator/reviewer and 32,768 for
  reasoner. The reasoner was raised from 8,192 after 20f truncation failures.
- `max_history_bytes=192000` is a **byte** bound, not a token count. Bounded
  synthesis additionally limits each request's encoded input and stores receipts.

The pool forwards tools, token caps and thinking flags unchanged. Changing a
model/thinking/output budget requires a fresh qualification; do not conflate it
with a routing or scheduler change. The modern serving path is entirely internal
to the allocation and requires neither Argo nor Tara tunnels. The optional
external-provider launcher and tunnel helper remain available for that separate
execution mode.

## Timeout recipe: size the boundary that actually waits

| Boundary | Dense 20-node setting | Final 256-node setting | Reason |
|---|---:|---:|---|
| Model request | 600 s | 600 s | One request and SDK retries |
| Tool/MCP read | 6,000 s | 6,000 s | One tool call, not the whole validator |
| Validator | 10,800 s | 10,800 s | Complete plan including delegated science |
| Code-generator delegation | 6,600 s | 6,600 s | One delegated plan |
| Reviewer | 1,200 s | 1,200 s | Evidence review |
| Reasoner | 1,800 s | 1,800 s | One synthesis call; output cap is separate |
| Tournament phase | 2,400 s | 10,800 s | Larger generation/ranking pool |
| MCP initialization | old SDK default 5 s; later 60 s | 60 s | Observed fan-out initialization failures |
| Native model startup | 5,400 s | 5,400 s | Starts after allocation, not submission |
| HPC startup allowance | 7,200 s | 7,200 s | Cluster/service readiness |
| Typical screen/full-context row | 5,000 s | 5,000 s | A few final rows requested 6,000 s |
| Typical CPU summary row | 300 s | 300 s | Recorded exceptions included 900 s |
| Requested Slurm walltime | 8 h for 20i | 9 h for 256c | Entire parent allocation |

Current MCP initialization is configurable as `timeouts.mcp_init`, default 60 s.
This makes the campaign-runtime fix part of normal configuration. The modern
launcher uses 60 s for both recipes; historical 20i's operational settings, launch script and original configuration
hash are retained separately; its research-bearing full reference is external. Do not multiply every timer by `N`: per-row numerical work
is unchanged with more nodes. Measure cold and warm tails, inference contention,
metadata/checkpoint overhead and enclosing plan duration. Add a documented
margin and retain failed/unfinished rows; a generous deadline does not eliminate
idle barriers or long tails. A scheduler wait of hours must not consume a
model-loading deadline.

The dense execution capsule used 160 requests per agent, at most six code-
generator delegations per validator, ten SDK model retries and three structured-
output retries. Each child has its own budget. On HTTP 429/529 or a 503 carrying
Retry-After, shared endpoint cooldown honors the header and adds jitter. Requests
are staggered at two-second spacing through recovery instead of reducing the
scientific campaign. Pool backends also cool down/fail over; a stream is not
replayed after output starts. Authentication failures are not treated as rate
limits.

## Other changes to preserve when porting

1. **Bounded hierarchical synthesis.** Large rounds no longer send every thread
   in one oversized prompt. Raw inputs/responses and coverage receipts persist;
   every thread is represented. Evidence stays distinct from model ranking.
2. **Checkpoint policy.** Campaign-local Jnana memory autosaves are coalesced to
   at least five seconds apart, with atomic writes and explicit phase-boundary
   saves. No global installed Jnana package is monkey-patched. A crash can lose
   recent unsaved updates; record that tradeoff.
3. **Registry index.** A process-local index avoids campaign-wide JSON rescans
   for each metadata query on the single-writer local MCP server. Disk remains
   canonical; forked/distributed/multiwriter paths retain disk queries. Row
   output/failure checks are not cached away.
4. **Skill metadata cache.** Bounded caches avoid repeated YAML parsing, retain
   fresh skill-body reads, invalidate on file version changes and return
   independent mutable metadata. File deletion/retargeting remains visible.
5. **Controller GC policy.** Optional generation-zero threshold 10,000 leaves
   automatic collection enabled and later-generation thresholds unchanged.
   Only the controller interpreter is affected. It is a CPU tuning knob, not a
   demonstrated GPU improvement over 20i.
6. **Scientific task granularity.** Frozen complete pipelines avoid repeated
   model-mediated setup and prevent scientific parameter drift. Matched screens
   use twelve genuinely distinct candidates (four per condition), fresh
   hypothesis/round/condition/replicate seeds and owned receipts. Metadata-only
   preparation is not evidence; never rerun finished inputs to fill GPUs.
7. **Manifest/navigation/polling fast paths.** Helpers compute checksums/seeds;
   exact artifact indices replace recursive shared-filesystem searches. Bounded
   polling returns immediately when actual done/status files are terminal.
8. **Native runtime fixes.** CPU subprocess timeouts kill descendants; zero-GPU
   metadata tasks clear inherited GPU slots; bounded logical slots map to one
   physical GPU; Amber paths under its existing cwd become relative to avoid
   native filename-buffer failures without changing chemistry.
9. **Source and role ownership.** Fresh campaign/state paths, source/input hashes,
   one ownership ledger and explicit host partitioning prevent stale evidence
   and foreign allocations from entering a run. Telemetry starts on all GPUs
   before inference loading, so startup costs remain visible.

The empty-own-output citation guard and prior-duration-first scheduler replay
were investigated after/alongside the campaigns but **not deployed** in 256c.
They are not presented as measured saturation fixes. The original reconciliation
policy and failed claim gate have not been silently replaced.

## Verification and reproducibility

```bash
# Offline controller, ownership, pool, backlog, synthesis and staging regressions.
PYTHONPATH=.:scripts .venv/bin/python -m pytest -q

# Additional compact-coordinate provenance tests use the scientific environment.
PYTHONPATH=.:scripts "$LOAM_SOFTWARE_ROOT/RFD4-Proteina/.venv/bin/python" \
  -m unittest discover -s tests -p test_compact_model_metadata.py -v

# Inspect a new staged job before submission.
PYTHONPATH=.:scripts .venv/bin/python scripts/check_single_allocation.py \
  /shared/scratch/my-new-experiment/recipe-20i
```

The historical code records, inert launch scripts and operational setting extracts,
input hashes, serving versions and measured settings are in
[`examples/scaling/history`](../examples/scaling/history). Original records
retain the old package name/site paths for traceability. The current wrappers
stage fresh evidence directories and use `loam`; they do not rerun an old job
in place. The original evidence directories, model weights, virtual environments,
service tokens and large report exports are deliberately not in Git.

For a new qualification, measure scientific production at every stage, per-node
GPU traces, whole-parent cost/coverage, inference latency and scientific queue
occupancy. Run independent coordinate, scientific-input, ownership, claim and
three-round tournament audits. Promote only the gates actually passed. The
historical Jupiter report/accounting scripts assume four-GPU node telemetry;
requalify their hardware integrators before using those plots to certify a
new GPU layout. A successful scheduler exit is insufficient by itself.

## Scientific data and proposal texts are separate from Git

The public branch contains code, operational settings, checksum identities and
historical run scripts. It excludes target structures/annotations, preparation
logs, prior scientific outputs, original research goal text and proposal texts.
On the current computer, the preserved full local reference is
`../moeen-private-inputs` relative to the checkout; it is outside this Git tree.
Treat that as an operator-owned data bundle, not a Git dependency or public asset.

For a new site, obtain an approved input bundle through the site's normal data
transfer process. The code does not upload or publish scientific data. Validate
and copy a local bundle to another location outside the checkout:

```bash
PYTHONPATH=.:scripts .venv/bin/python scripts/prepare_scaling_inputs.py \
  /approved/source/inputs /shared/science/egfr/inputs
export LOAM_SCIENCE_INPUTS=/shared/science/egfr/inputs
```

The helper verifies all files named by
`examples/scaling/egfr/inputs-manifest.json` and ignores unrelated source files.
The stager accepts `--inputs` instead of the environment variable. It checks the
reference hashes before staging and again before execution; generated job trees
ignore all their contents in Git.

Historical campaigns started with 24 proposal texts. To reproduce that warm
start, supply an approved private proposal JSON with `--seed-hypotheses` or
`LOAM_SEED_HYPOTHESES`; IDs, Elo/reviews/matches reset for the new campaign.
Without a proposal file the public example generates its initial pool from
scratch, adding six generation requests at batch size four compared with that
24-proposal warm start. This is a prompt/startup difference, not a measured new
performance result. For the original full scientific prompt, use `--reference`
with an operator-owned reference containing the goal/config/input manifest;
the public example goal describes the assay contract rather than unpublished
campaign context.
