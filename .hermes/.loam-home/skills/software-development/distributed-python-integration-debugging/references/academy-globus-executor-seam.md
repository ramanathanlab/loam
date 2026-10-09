# Academy 1.0.0 + Globus Compute executor seam

A source-derived reference for debugging a third-party `concurrent.futures.Executor` integration without submitting remote work.

## Verified package facts

- `academy-py==1.0.0`
- `globus-compute-sdk==4.15.0`
- `globus_compute_sdk.Executor` subclasses `concurrent.futures.Executor`.
- `Executor.submit(fn, *args, **kwargs)` returns `ComputeFuture`, which subclasses `concurrent.futures.Future`.
- `Executor.shutdown(wait=True, *, cancel_futures=False)` accepts Academy's ownership/cleanup invocation.

## Academy Manager runtime contract

Academy does not require a custom protocol beyond standard executor/future behavior:

```python
# academy.manager.Manager._run_agent
await asyncio.wrap_future(
    executor.submit(_run_agent_on_worker, spec, **spec.submit_kwargs),
    loop=loop,
)
```

On `Manager.close()`, it invokes:

```python
executor.shutdown(wait=True, cancel_futures=True)
```

`_run_agent_on_worker(spec, academy_debug_mode=False, **kwargs)` starts a fresh event loop with `asyncio.run(_run_agent_async(spec))`. That function constructs or uses the agent, creates `Runtime`, and waits for the agent to receive shutdown. An Academy agent is therefore a long-lived remote executor task.

## Diagnostic interpretation

Interface compatibility does **not** prove a viable distributed launch. The task payload must serialize; the endpoint worker must import the registered worker target and agent module; and the worker must reach/authenticate to the Academy exchange.

`Manager.launch()` returns after registration/background-task creation, before worker health is established. Follow it with an intentional health action/probe. On failure, inspect the Manager-owned launch task (`manager._acbs[handle.agent_id].task`; private and version-sensitive) and correlate its exception/task identity with remote worker logs. This separates submit/serialization/import failure from worker-to-exchange reachability/authentication failure.

## Safe first pass

Prefer source and runtime-interface inspection first. Do not construct a real Executor or call `submit()` merely to prove subclass/signature/Future compatibility: construction can start background machinery, and submission can register functions or allocate remote work.
