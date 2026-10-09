# Managed Polaris endpoint repair notes

Use when a Globus Compute manager endpoint is online but a user endpoint cannot execute a campaign probe.

## Required readiness ladder

1. Manager endpoint is running and subscribed to its command queue.
2. The manager can fork a user endpoint under the submitting user's identity.
3. The user endpoint can resolve `globus-compute-endpoint` from its *sanitized child PATH*.
4. The user endpoint's Parsl provider can resolve PBS commands (`qsub`, `qstat`) from that same managed PATH.
5. The provider submits a one-node debug block.
6. HTEx workers register with the interchange.
7. A source-registered exact-Fraction probe returns placement and arithmetic evidence.
8. Only then start the Academy/FastMCP agent pool and campaign code generators.

## Diagnosis

- Read the manager endpoint log and the newest matching `uep.<manager-id>.*` `endpoint.log`.
- Inspect the generated `submit_scripts/` only for non-secret facts: queue, node count, walltime, worker launch command, and output/error locations.
- A PBS job in `R` state is not sufficient: verify a worker registration in `GlobusComputeEngine-HighThroughputExecutor/interchange.log`.
- `ENDPOINT_NOT_ONLINE` can be a user-endpoint readiness failure even if the manager reports online.

## Sanitized PATH pitfall

Managed endpoint children may not inherit an interactive login-shell PATH. If logs report either of these, repair the manager-supported persistent user environment and restart the manager:

- `FileNotFoundError` for `globus-compute-endpoint` — put the actual endpoint CLI directory first in PATH.
- `/bin/sh: qsub: command not found` — include the PBS client directory in PATH.

After changing the persistent environment, confirm the newly started manager/user child uses the intended path. Do not trust a shell-local export or merely checking `command -v` over SSH.

## Probe contract

Use source-registered or importable top-level probe code to avoid dill/module-resolution ambiguity. The probe must return:

- compute hostname;
- scheduler job ID;
- accelerator visibility;
- exact `Fraction` arithmetic result and comparison;
- remote task/function identity when available.

Keep the complete user endpoint configuration unchanged for all related probes. Configuration changes can spawn new user endpoints and consume limited debug-queue capacity.
