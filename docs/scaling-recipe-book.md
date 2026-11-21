# Scaling recipe book

The scaling recipe (`loam.scaling`, `loam-scaling`) sizes a campaign's settings to an allocation: how many nodes serve inference, how many validators run at once, and how many hypotheses each round selects and generates. Its numbers come from one measured dense recipe, and this page records what was measured, where, and how far the numbers can be trusted.

Moeen Meigooni measured it in November 2026 on **Jupiter**, on GH200 nodes with **four GPUs and 288 CPU cores per node**, at **20 and 256 nodes**, running the EGFR binder assay against a self-hosted Nemotron inference pool. These are measured operating points for that assay on that System. They are not a hardware-independent optimum.

## Asking for a recipe

```bash
loam-scaling --system jupiter --nodes 20      # a measured configuration
loam-scaling --config loam.yaml               # the campaign config's System, node shape and node count
```

The command prints the recipe, whether this exact configuration was measured, the measurement it comes from (`measured_on`), and the recipe as campaign config settings (`campaign_settings`: `rounds`, `concurrency`, `synthesis_concurrency`, the `tournament` keys and `execution.hpc.science_workers_per_gpu`). It never stages or submits anything. From Python, `loam.scaling.recipe(system, nodes)` returns the same recipe for a `System` (with its node shape, e.g. `hpc.system_definition()`).

The recipe does not quietly extrapolate:

| Asked for | Result |
|---|---|
| Jupiter, 4 GPUs and 288 CPUs per node, 20 or 256 nodes, measured split and sharing | The measured recipe, `measured: true` |
| Same System and node shape, another node count, or another `--inference` split or `--science-workers-per-gpu` | The recipe, an `UnmeasuredScalingWarning` naming what differs, and `measured: false` |
| Another System (Aurora, Polaris) or node shape (e.g. `--gpus-per-node 8`) | Refused (`UnmeasuredScaling`; the command exits non-zero) |
| The same with `--allow-unmeasured` (`allow_unmeasured=True`) | The formula's recipe, an `UnmeasuredScalingWarning`, and `measured: false` |

An unmeasured recipe is a starting point for qualification, not a throughput prediction. A System that allows fewer science workers per GPU than the measured two (Aurora and Polaris allow one) gets fewer, so its recipe is about half as dense.

## The formula

Let `N` be the total nodes, `I` the inference nodes, `S = N - I` the science nodes, `G` the GPUs per node, and `P` the science workers sharing each GPU (two in the measurement). `W = S * G * P` is the number of logical science slots, not physical GPUs.

| Setting | Formula | 20 nodes, G=4 | 256 nodes, G=4 |
|---|---|---:|---:|
| Inference : science nodes | `I = round(0.4N)`, `S = N - I` | 8 : 12 | 102 : 154 |
| Physical science GPUs | `S*G` | 48 | 616 |
| Logical science slots | `W = S*G*2` | 96 | 1,232 |
| Validators at once, `concurrency` | `C = ceil(W/2)` | 48 | 616 |
| Hypotheses selected per round, `tournament.top_k` | `K = 3C` | 144 | 1,848 |
| Initial pool, `tournament.initial_hypotheses` | `2K` | 288 | 3,696 |
| Fresh hypotheses each later round, `tournament.new_per_round` | `ceil(K/2)` | 72 | 924 |
| Hypotheses per generation request, `tournament.gen_batch_size` | 4 | 4 | 4 |
| Swiss rounds, `tournament.swiss_rounds` | 1 | 1 | 1 |
| Parallel ranking workers, `tournament.workers` | `2I` | 16 | 204 |
| Reasoner requests at once while reducing, `synthesis_concurrency` | `I` | 8 | 102 |
| Campaign rounds, `rounds` | 3 | 3 | 3 |

`top_k` sets the useful work selected for the whole round, and `concurrency` bounds how many validators run at once: a finished validator admits the next selected hypothesis straight away, so a three-wave backlog keeps the science slots fed. Raising `top_k` alone doesn't raise how many run at once. Each hypothesis can submit several science rows, so validators are not GPUs.

The inference : science split has no campaign config key; it is placed by single-allocation staging.

## What was measured

| Case | Inference : science nodes | Validators at once | Selected per round | Whole-allocation GPU busy | Distinct screens | Full-context trials | Full-context trials per allocated GPU-hour | Qualification |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 20b, job 2952205 | 8 : 12 | 12 | 48 | 26.66% | 1,304 | 272 | 1.049 | Passed |
| 20e, job 2961727 | 8 : 12 | 48 | 96 | 38.71% | 2,616 | 528 | 1.711 | Passed |
| 20g, job 2972192 | 8 : 12 | 48 | 144 | 47.96% | 4,722 | 764 | 1.987 | Passed |
| **20i, job 2979745** | **8 : 12** | **48** | **144** | **51.06%** | **4,858** | **819** | **2.120** | **Best qualified 20-node recipe** |
| 20j, job 2991226 | 8 : 12 | 48 | 144 | 48.35% | 5,234 | 871 | 1.954 | Passed; later controller fixes |
| 256a, job 2976829 | 102 : 154 | 616 | 1,848 | 24.87% | 16,994 | 1,870 | 0.733 | Retired partial campaign; requeues charged |
| 256b, job 2987415 | 102 : 154 | 616 | 1,848 | 32.30% | 22,700 | 2,890 | 0.934 | Retired partial campaign; input/provenance findings |
| **256c, job 2999784** | **102 : 154** | **616** | **1,848** | **40.51%** | **61,134** | **10,337** | **1.371** | **Three rounds completed; claim-provenance qualification failed** |

The recipe is 20i at 20 nodes and 256c at 256 nodes (both Jupiter Slurm jobs).

**GPU busy** is the time integral of sampled NVML `utilization.gpu` over all allocated physical GPU-seconds, with missing intervals counted as zero. Startup, round barriers, idle and drain time, failed work and requeues all stay in the cost. It is neither SM occupancy nor FLOP efficiency. `loam-saturation` computes it the same way. The 256c run had 99.70% telemetry coverage, 65.95% inference GPU busy and 23.66% science GPU busy. Its 40.51% allocation-wide result **did not meet the 50% target**.

20f briefly reached 51.18% allocation-wide busy but failed reasoner output completion, so it is not the best qualified recipe. Controller fixes (caching, GC, MCP initialization) carried into 20j and 256c fixed specific CPU and initialization problems, but 20j did not beat 20i's GPU busy or its trials per GPU-hour. These adaptive campaigns are not a controlled fixed-work strong-scaling test.

A **screen** is a distinct, receipt-owned input/seed trial. In 256c the independently checked pipeline yielded 61,134 RFD backbones, 61,134 sampled MPNN sequences, 305,670 crop cofold models, 51,685 full-context cofold models, 10,337 minimizations and 10,337 single-frame MM/GBSA evaluations. Five correlated predicted poses per input are not five independent binders. There were 61,134 unique screen sequences and 9,305 unique full-context sequences. No production MD trajectories were run: the energy route minimizes and evaluates one frame. Confidence, geometry and static energies are computational screening measurements, not measured affinity.

The final audit of 256c found no numerical errors but 1,434 unresolved citation sources, including at least one claim citing an empty run directory of its own. Some are path or alias resolution issues rather than false claims. Numerical output, orchestration completion and claim qualification are separate checks.

### The inference : science split

An earlier 20-node study tried 1:19, 2:18, 4:16 and 8:12 inference : science nodes. It observed about 674, 1,139, 1,411 and 1,623 coordinate files per science allocation-hour, with an 8:12 repeat at 1,452. Those rates have a different denominator and scope from the table above. They support **8:12 (`I ≈ 0.4N`) as an empirical starting point**, not a universal optimum. With one inference node, agents waited on inference while science GPUs sat without submitted work.

### What the measurement ran with

Changing any of these needs a fresh qualification; don't conflate it with a routing or scheduler change.

- **Inference.** Checkpoint `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16`, revision `fb0ee0bc6b3f70342afd8d84cb80dbadbaa8a1fe`, served by vLLM. One replica per inference node, tensor-parallel across its four GPUs, GPU memory fraction 0.85, 131,072-token context, `max_num_seqs=16`. Language-model-only serving, automatic tool choice, the `qwen3_coder` tool parser and the `nemotron_v3` reasoning parser. The 120B BF16 checkpoint does not fit one GH200, so a smaller tensor-parallel group must be checked for weights plus KV cache before trying a different `G`.
- **Thinking and output.** `enable_thinking=false` for validator, code-generator, reviewer and reasoner; tournament generation and ranking kept their default thinking. Output caps of 8,192 tokens for validator, code-generator and reviewer, and 32,768 for the reasoner (raised from 8,192 after 20f's truncation failures). `max_history_bytes=192000`, a byte bound, not a token count.
- **Agent budgets.** 160 requests per agent, at most six code-generator delegations per validator, ten SDK model retries and three structured-output retries. Requests staggered two seconds apart while recovering from rate limits.
- **Timeouts.**

| Boundary | 20i | 256c |
|---|---:|---:|
| Model request | 600 s | 600 s |
| Tool/MCP read | 6,000 s | 6,000 s |
| Validator | 10,800 s | 10,800 s |
| Code-generator delegation | 6,600 s | 6,600 s |
| Reviewer | 1,200 s | 1,200 s |
| Reasoner | 1,800 s | 1,800 s |
| Tournament phase | 2,400 s | 10,800 s |
| MCP initialization | 5 s, later 60 s | 60 s |
| Inference model startup | 5,400 s | 5,400 s |
| HPC startup allowance | 7,200 s | 7,200 s |
| Typical screen/full-context row | 5,000 s | 5,000 s |
| Typical CPU summary row | 300 s | 300 s |
| Requested Slurm walltime | 8 h | 9 h |

Don't multiply every timer by `N`: per-row numerical work doesn't change with more nodes. Only the tournament phase grew, because the pool it generates and ranks grew.

## Qualifying a new System or node shape

The formula generalizes workload density, not demonstrated saturation. For a new System, accelerator, memory size, model, target, GPU count or scheduler: run one small layout, measure inference and science service times, and adjust the split. Inspect per-role GPU busy, request latency, queue depth, row submission and dispatch times, logical slot occupancy, and stage production per science-node-hour and per total allocated GPU-hour. Adding science nodes without a ready science backlog can lower efficiency; 256c still had substantial science starvation, round barriers and long hypothesis tails. Record a new measurement in `loam/scaling.py` and here only once it has passed the same audits: a successful scheduler exit isn't enough.

## Where the numbers come from

The measurements, job IDs and settings above are from Moeen's `docs/scaling-recipe-book.md` and `examples/scaling/history/` on the `moeen` branch, which also hold the original launch scripts, provenance records and native runtime versions. Those records, the run directories, the scientific inputs and the model weights are deliberately not in this repository.
