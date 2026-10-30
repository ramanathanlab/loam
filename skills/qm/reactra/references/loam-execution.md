# Execute Reactra through Loam

Read this before running Reactra remotely or recovering interrupted work. Use
[Loam's HPC deployment skill](https://github.com/ramanathanlab/loam/blob/develop/skills/hpc-deployment/SKILL.md) for endpoint/pool
provisioning. This reference describes the payload, not a second deployment system.

## Make the skill available to the campaign

Install this skill directory, including `references/`, into the active Hermes
skill library. With `run-campaign`, that library is shared from
`<loam-repo>/.hermes/.loam-home/skills`; a source checkout alone does not register
its root `skills/` directory. Confirm `skill_view(name="reactra")` resolves it.
Add the name to the selected profile in your external Loam configuration:

```yaml
campaign:
  skills: [reactra]
```

This is a fragment: retain the campaign's existing directory, goal, rules, and
other skills. Naming a skill does not install it. Put the actual control/native
environment setup and absolute worker paths in `execution.setup`, and include
the YAML, MECH, assets, store/request IDs and budget in the parent's structured
input contract. Code-generators must use that contract, not discover a convenient
old simulation store. Preserve Loam's required response format when reporting
the CLI invocation and result.

## Match the deployment to one task

The MCP tool is
`run_code(payload, workdir=None, timeout=300.0, venv_path=None)`.
Its Python payload returns `{stdout, stderr, returncode}`; `returncode` is a string.
`venv_path` selects `<venv_path>/bin/python` for the payload; it does not provision
Reactra or its native environment. The pool rotates handles and does not promise
that a later call reaches the same host or tile.

In direct mode, the agent is pinned and the payload inherits that reservation.
In cluster mode, the client is deliberately unpinned: Ensemble Launcher places
each `run_code` task on one GPU/tile with `ppn=1`. Inspect placement inside the
executing payload, not the submitting client. Use the existing HPC deployment
skill to establish the pool before sending scientific work.

Before the first experiment, establish the control interpreter, native profile,
shared asset paths, visible accelerator/CPU reservation, remaining allocation time,
and a writable per-experiment directory. Use a lightweight payload to inspect the
hostname and existing `CUDA_VISIBLE_DEVICES` / `ZE_AFFINITY_MASK`; do not change
these to expose additional devices. Run `capabilities` with the selected Reactra
interpreter. It establishes software availability, not native qualification.

Use a profile scoped to the worker's reservation. An omitted/empty
`native_device_slots` means Reactra runs ensemble members sequentially while
inheriting its caller's device environment. Explicit slots may name only resources
reserved for that task. Copying a full-node profile onto a one-tile Loam agent can
override the endpoint's pinning and contend with other agents. Keep native threads
and any CPU affinity within the reservation as well. Do not assume a generic
XPU/CUDA/CPU selection recipe establishes Reactra provider support.

Parallelize independent experiments with independent stores, one active run per
reserved accelerator. Direct-mode round-robin alone is not a concurrency limit:
an agent accepts overlapping calls. Cluster mode delegates placement to Ensemble
Launcher; client count is not the number of reserved devices. Bound in-flight
work to the authorized resources and verify placement before scientific work.

## Submit the ordinary command as a payload

For an already reviewed run and authorized allocation, send Python like this to
`run_code`, replacing the paths/ID with the task's actual deployment:

```python
import os

control_python = "/shared/project/reactra/control-python"
os.execv(control_python, [
    control_python, "-m", "reactra.interfaces.cli", "run",
    "/shared/project/experiments/trial-001/run.yaml",
    "--store", "/shared/project/experiments/trial-001/store",
    "--request-id", "trial-001",
])
```

`execv` uses the supplied interpreter and preserves environment pinning and exit
status without leaving an extra waiting Python wrapper. Use the same form for
`status`, `result`, or `resume` with their CLI arguments. Do not introduce a custom
OpenMM loop, detach a background controller, or submit a nested PBS/Slurm job from
this payload. The Globus endpoint already owns the allocation. Reactra's standalone
PBS launcher is an alternative deployment route, not a step inside this one.

Set the tool's explicit `timeout` for the authorized operation, including startup,
assessment, and publication, within the remaining allocation and transport limits.
**The default 300 seconds is too short for an ordinary full enzyme run.** If the
endpoint cannot sustain the required duration, resolve that deployment limit before
launching; do not hide it with detached jobs. Leave time for committed checkpoints
and failure reconciliation. Historical example timings are estimates, not deadlines
or guarantees for another system.

## Store ownership and failure reconciliation

- One mutable store has one active controller on one node. Independent allocations
  must not share its SQLite writer. Use new stores for parallel experiments;
  same-store prefix reuse is serialized after the source is accepted and idle.
- For live `status`, use the owning node through an available placement mechanism.
  The stock round-robin MCP call has no worker selector. When ownership cannot be
  guaranteed, observe the in-flight call or existing logs and inspect the store
  after the writer has stopped; do not open live cross-node SQLite/WAL access.
- The current HPC helper times out its immediate payload process; it does not
  establish that Reactra's native descendants were terminated. Tool timeout,
  transport failure, or an absent reply therefore requires checking the owning
  process/allocation and persisted attempt/journal before any new writer or resume.
  Ask the operator if the existing tools cannot resolve ownership. A timeout is
  not proof of cancellation.
- After ownership is clear, inspect `status`, `result`, the ensemble journal, and
  committed checkpoints. Preserve stdout/stderr and task/store IDs. Retry only a
  diagnosed infrastructure interruption within the authorized budget. Scientific
  refusals, incompatible identities, and deterministic defects require correction.
- Resume the original store/request only with a compatible runtime and accessible
  committed artifacts. Do not delete leases, edit SQLite/blob files, or substitute
  checkpoint bytes to make recovery appear successful.

## A revised experiment using accepted work

For example, when revising sampling after an accepted source selection:

```bash
reactra plan REVISED.yaml --store STORE \
  --reuse-prefix-from SOURCE_ID --reuse-through reactive_seed_selection --format text
reactra run REVISED.yaml --store STORE --request-id REVISED_ID \
  --reuse-prefix-from SOURCE_ID --reuse-through reactive_seed_selection
```

Both options must accompany both commands. Choose a boundary before the changed
inputs take effect: a model/region change cannot reuse that model's minimization
or thermalization. Let the installed resolver establish compatibility; refusal is
not permission to weaken the identity checks. Use CLI help for supported boundaries.
The inspected plan labels reused versus fresh operations and bounds fresh work.
`--expect-plan-sha256 DIGEST` can bind `run` to a specifically reviewed plan.

For an exact unfinished member checkpoint, take the identifiers from status:

```bash
reactra result --store STORE --request-id RUN_ID --operation OPERATION_ID \
  --attempt-id ATTEMPT_ID --checkpoint-generation 0 --lineage m000
```

Inspection/export does not make the checkpoint portable to a different experiment.

## Larger experiment batches

For many independent experiments, use the cluster's existing `run_batch` and
`batch_status` interface with one thin CLI-invoking script and one JSONL row per
experiment. Each row needs its own YAML, request ID, and store; `gpus=1` reserves
one tile per row, and the profile must remain within that reservation. Set
`timeout_each` explicitly. Keep simulation and recovery logic inside Reactra.

Batch status tracks process completion, not scientific qualification. Preserve
Reactra's exit code and reports. In particular, exit `4` can be a terminal unmet
scientific outcome: do not resubmit it automatically as a transient failure.
Likewise, a batch lease becoming stale does not prove native descendants stopped.
Reconcile ownership before retrying a row or resuming a batch; an unchanged
interrupted Reactra request needs `resume`, not a new `run` against the same store.
