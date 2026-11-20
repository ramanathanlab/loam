# Nemotron replica pool

The current entry point is the [scaling recipe book](scaling-recipe-book.md).
Use `examples/scaling/run-20i.sh` or `run-256c.sh` to stage all services in one
parent Slurm allocation. The original split-allocation ratio-study helpers
remain in `scripts/` for historical analyses and require their external
experiment ledger/artifacts; they are not prerequisites for a fresh combined job.

`loam.inference_pool.InferencePool` exposes authenticated `/v1/chat/completions`,
`/pool/status` and `/pool/batch` routes. In the combined job, the gateway is local
to the coordinator compute node. It discovers only readiness files belonging
to inference hosts in that exact parent allocation, validates bearer tokens and
pinned TLS, and forwards requests without changing model, thinking, tools or
output budgets. No compute-node Internet or external inference tunnel is needed.

`least_loaded_affinity` selects a healthy replica by in-flight requests and
estimated token work, with soft affinity for a warm prefix. `round_robin` and
`least_loaded` are available. Per-backend capacity, queue backpressure and
cooldowns are bounded. Failed initial requests can try another healthy backend;
a stream is never replayed after it starts emitting output. SDK/Jnana shared
endpoint cooldowns honor rate-limit headers and stagger recovery with jitter.

Native vLLM continuously batches concurrent requests (`max_num_seqs=16` in the
measured profile). Ordinary campaign agents use chat requests and already benefit
from this. `POST /pool/batch` supports up to 64 independent nonstreaming requests
with explicit custom IDs and bounded fan-out; it does not merge prompts. It is
available to callers but was not a separate campaign-agent batch scheduler.

The gateway writes timing/token/latency records without prompt, completion or
credential contents. Its private service record points to the token file rather
than embedding a token in campaign YAML. Credentials are minted only after the
allocation begins, so scheduler queue time does not consume their lifetime.

The earlier 20-node ratio study selected 8 inference : 12 science as its best
observed split. Workload admission/backlog and controller changes subsequently
raised qualified allocation-wide sampled GPU busy to 51.06% at 20 nodes. The
latest 102:154 combined 256-node campaign reached 40.51%, completed three rounds
and retained claim-provenance findings. Node ratio alone does not eliminate
science starvation, round barriers or long hypothesis tails. Exact settings,
timeout changes, scientific throughput and limits are in the recipe book and
`examples/scaling/history/results.json`.
