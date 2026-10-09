# Approval-denial discipline for remote execution

## Rule

When an approval or policy gate denies a command and explicitly says not to retry, rephrase, or seek the same outcome through another route, stop the affected workflow immediately. Do not substitute a different remote command, script, or transport to obtain the blocked information.

## Why it matters in execution investigations

Remote-execution preflight commonly combines harmless read-only inspection with stateful endpoint operations. A denied inspection is not permission to proceed with launch/retry steps based on assumptions. Report the already-established state separately from the blocked check, then wait for renewed user direction or approval.

## Safe reporting shape

- State exactly what was verified before the denial.
- Identify the blocked check without reproducing sensitive command details.
- State that no workaround or retry was attempted.
- Do not claim endpoint readiness, scheduler cleanliness, or completion based on stale evidence.
