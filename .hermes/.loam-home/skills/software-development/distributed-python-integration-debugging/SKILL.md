---
name: distributed-python-integration-debugging
description: Investigate Python distributed-execution integration failures by separating interface compatibility, payload transport, remote runtime, and control-plane health without changing application files.
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [debugging, python, distributed-systems, executors, futures, integration]
    related_skills: [systematic-debugging, spike]
---

# Distributed Python Integration Debugging

## Trigger

Use when a Python library submits work through a third-party executor, task queue, RPC system, scheduler, or remote function service and work fails to become healthy. This applies especially where the local library accepts a standard executor/future abstraction but executes payloads in another environment.

Use this skill for **investigation without application-file changes** unless the user explicitly requests a fix.

## Core model: verify the seam in layers

Do not treat “accepted by the constructor” as proof that remote execution works. Classify evidence in this order:

1. **Interface compatibility** — local executor type, `submit` signature, returned future, and shutdown semantics meet the caller’s expectations.
2. **Submission semantics** — identify the exact callable, positional arguments, forwarded keyword arguments, and how the caller awaits the future.
3. **Payload transport** — callable, arguments, captured state, configuration, and dependencies can serialize and cross the boundary.
4. **Remote runtime** — workers can import the callable’s module and application-defined agent/task modules; their dependencies and Python compatibility match.
5. **Control-plane health** — remote work can authenticate and communicate with its service, exchange, queue, or callback endpoint.
6. **Lifecycle semantics** — distinguish task submission from readiness. Long-lived workers/agents require an explicit health action/probe after launch.

Report the layer established by evidence and the first layer not yet established. Do not claim root cause from a compatible interface alone.

## Investigation procedure

### 1. Establish versions and source locations

Use package metadata plus runtime module locations. Packages can be importable without normal distribution metadata, so report both facts separately.

```python
import importlib.metadata as md
import package_name
print(md.version("distribution-name"))
print(package_name.__file__)
```

Read the installed caller and provider source, not only annotations or web documentation.

### 2. Extract the caller’s real executor contract

Find:

- runtime acceptance check (`isinstance`, duck typing, or no validation);
- exact `submit(...)` invocation;
- whether submit keyword arguments are forwarded as task/resource options;
- return-value treatment (`asyncio.wrap_future`, `.result()`, callbacks, polling);
- cleanup invocation and its keyword arguments.

State the expectations precisely. Example form:

```text
submit(callable, spec, **submit_kwargs) -> concurrent.futures.Future-compatible object
shutdown(wait=True, cancel_futures=True)
```

### 3. Compare provider behavior offline first

Prefer `issubclass`, `inspect.signature`, method-resolution order, and source inspection. Verify the future class is compatible with the consumer’s await bridge.

Avoid creating a real remote executor merely to inspect static compatibility: construction may start threads, load credentials, or allocate network resources; `submit()` may register code or submit work.

### 4. Trace remote launch lifecycle

Read the target callable end-to-end:

- Does it create an event loop?
- Does it construct an agent/task instance remotely?
- Does it wait for shutdown, making the future intentionally long-lived?
- What factory/client/control-plane configuration is included in the serialized argument object?

A launch API that returns a handle may only mean “registered and scheduled.” Determine whether it starts a background launch task and where that task’s exception is retained.

### 5. Design the smallest deliberate live diagnostic

Only after the offline seam is understood, propose or run one controlled remote operation if authorized. It should:

1. use a single worker/task;
2. invoke one trivial health operation through the real control plane;
3. retain the submitted future/task ID;
4. inspect the framework-owned launch/background task exception if it ends early;
5. correlate task ID with remote worker/scheduler logs.

This separates:

| Observation | Likely layer |
|---|---|
| Submission/registration raises | client configuration, serialization, or API authorization |
| Launch/background task fails immediately | serialization, worker import, or remote startup |
| Worker task remains live but health action times out | control-plane connectivity/authentication or worker initialization |
| Health action returns an exception | application action/runtime |

## Evidence and reporting

Report four sections:

1. **Verdict:** whether the raw provider satisfies the caller’s interface.
2. **Exact contract:** source-derived method calls, signatures, future handling, and shutdown semantics.
3. **Execution semantics:** lifecycle and readiness behavior, including why a returned handle/future does or does not prove health.
4. **Minimal diagnostic:** the lowest-impact next probe and the evidence it yields.

Call out private/internal objects used for diagnostics as version-sensitive. Do not recommend permanent production coupling to them.

## Pitfalls

- **ABC pass ≠ integration pass.** Standard `Executor`/`Future` compatibility says nothing about remote serialization, imports, network reachability, or credentials.
- **Do not confuse submit success with worker health.** Remote frameworks frequently return before a worker is ready.
- **Inspect actual installed source.** Documentation can differ by package version.
- **Do not change files for a no-change investigation.** Use read-only source inspection and runtime introspection first.
- **Avoid duplicate shutdown assumptions.** Establish whether provider shutdown is idempotent before treating a second cleanup call as causal.

## Reference

See `references/academy-globus-executor-seam.md` for a source-derived Academy 1.0.0 / Globus Compute example.
