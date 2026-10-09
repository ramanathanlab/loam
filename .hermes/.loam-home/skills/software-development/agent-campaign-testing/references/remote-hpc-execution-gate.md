# Remote HPC execution gate

Use this when campaign code-generator work is intended to run through a remote execution service rather than the campaign host.

## Preconditions

- Confirm endpoint status and retrieve endpoint metadata/configuration schema under the intended identity.
- Request the smallest permitted debug allocation shape first.
- Use a normal importable module-level function for the first Globus Compute probe. Lambda/inline closures can fail function registration independently of endpoint readiness.
- Keep the probe harmless: return hostname, scheduler job ID, runtime version, accelerator visibility, and a short accelerator diagnostic.

## Acceptance before campaign use

Endpoint status alone is insufficient. A remote backend is ready only after the probe returns successfully with:

1. a remote placement indicator such as hostname and scheduler job ID;
2. the expected runtime and visible accelerator assignment;
3. an execution result that can be persisted with task ID, stdout, stderr, return code, and endpoint target.

Only then may the campaign switch its code-generator execution contract to remote execution.

## Polaris / Globus Compute example pitfalls

- An endpoint can advertise `online` while dynamic allocation startup still fails; trust the completed probe, not control-plane status.
- Validate endpoint-generated configuration carefully. A template that quotes `walltime` may reject a caller-provided quoted/string value through double quoting. Prefer the endpoint's documented default first, then repair the endpoint template/configuration rather than guessing additional client-side forms.
- Endpoint submission failures such as remote YAML parsing errors or internal startup exceptions are backend failures. Preserve the returned error verbatim, stop before crediting remote execution, and escalate to the endpoint owner/admin with the minimal probe and request configuration.
- Do not treat an unsuccessful submission attempt as a valid HPC task ID or allocation. Record a task ID only after the service has assigned one.

## Campaign integration

Keep the remote-execution adapter narrow: code generator submits an exact payload, captures the remote task handle/result, and reports observed evidence. Validators must inspect that recorded result; they must not substitute their own local calculation. If the remote backend is not proven, keep the existing verified local contract rather than silently falling back or claiming HPC evidence.
