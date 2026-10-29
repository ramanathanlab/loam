---
name: aurora-genie3
description: >-
  Generate protein backbones with Genie3 on Aurora, including its built-in
  multi-device sharding flags and the requirement to run from its repo. Use when
  you need volume generation that fans out across tiles without writing a
  sharding scheme yourself.
---

# Genie3 on Aurora

```bash
R=/flare/FRAME-IDP/agentic_campaign_09_26/binder_design
source $R/envs/_scripts/activate_genie3-2026-venv.sh
echo $GENIE3_REPO     # <root>/software/Genie3-Aurora
cd $GENIE3_REPO       # MUST run from here
genie3 generate -c <config.yaml>
```

Real usage (verified rc=0):

```
genie3 generate [-h] -c CONFIG [--verbose] [--log-dir LOG_DIR]
                [--num-devices N] [--shard-id ID] [--num-shards N]
```

`-c/--config` is required and is a YAML config, not flags. **`--num-shards` /
`--shard-id` / `--num-devices` give you sharding for free**: launch one
`run_batch` row per shard with the same `--num-shards` and a distinct
`--shard-id`, and the work splits without a custom dedup scheme.

`cd $GENIE3_REPO` is mandatory — the tool resolves config and model paths
relative to its repo root and fails elsewhere.

Evaluation inside this venv uses ESMFold v1, ProteinMPNN, TMscore, DSSP and
IPSAE from `software/genie3-tools`, so a generate → evaluate loop stays in one
environment.
