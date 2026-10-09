# Polaris Globus Compute flex-scaling inspection notes

Session-specific, non-secret operational reference for a Polaris deployment.

## Discovery method

On the Polaris login host, inspect `~/.globus_compute/` for endpoint directories. For each candidate, read only:

- `endpoint.json` for the manager endpoint UUID
- `config.yaml` for the display name and whether an engine is deliberately absent
- `user_config_schema.json` for accepted client configuration
- `user_config_template.yaml.j2` to verify which resource fields are actually rendered
- recent `endpoint.log` and process state for operational health

Do **not** print credential material, certificate directories, token stores, or complete historical working directories.

## Flex-scaling verification

A permissive endpoint schema alone does not prove routing. Confirm that the Jinja template feeds the intended PBSPro provider fields from `user_endpoint_config`, especially:

- `account`
- `queue`
- `walltime`
- `nodes_per_block`
- `init_blocks`, `min_blocks`, `max_blocks`
- filesystem/placement scheduler options

An intentionally minimal manager `config.yaml` with the engine in the per-submission template is a valid design: it permits one manager UUID to provision distinct allocation shapes. Record the manager UUID separately in campaign configuration (`HPC_GLOBUS_ENDPOINT_ID`); never bake it into this skill.

## Polaris queue checks

Before selecting an allocation shape, query current queues with `qstat -Q`; inspect detailed limits with `qstat -Qf <queue>` as necessary. Verify both scheduler availability and compatibility with node count/walltime. In the inspected configuration, the template explicitly warned that `prod` is a route queue with a minimum 10-node request, so do not use it for a one-node job.

## Known endpoint status at inspection time

One endpoint labeled `Polaris GPU (academy-compute-mcp)` was the intended per-submission flex-scaling endpoint. A separate historical `qmagent` endpoint had stale command-queue errors and was inactive; validate live health rather than relying on endpoint directory presence.
