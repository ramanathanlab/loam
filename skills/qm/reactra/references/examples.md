# Packaged flooding examples and adaptation

Use the installed package's examples as a runnable starting point, rather than
copying a historical store or an agent's private engine checkout. The maintained
choices are `pmd-flooding` and `cm-flooding`. The same CLI and engine run both.

## Materialize one reference experiment

On a provisioned host, with actual deployment paths and a new output directory:

```bash
reactra example pmd-flooding --output pmd-trial-001 \
  --native-profile /shared/project/reactra/deployment/pool-native.yaml \
  --model-weights /shared/project/reactra/assets/uma-s-1p2p1.pt \
  --forcefield /shared/project/reactra/assets/protein.ff14SB.xml \
  --forcefield /shared/project/reactra/assets/tip3p.xml
reactra preview pmd-trial-001/run.yaml --authoring-only --format text
reactra validate pmd-trial-001/run.yaml
```

The `/shared/project/...` paths are examples, not existing deployment locators.
For CM, substitute `cm-flooding` and a new CM directory. The materializer binds
exact profile/weight/forcefield assets, copies the bundled MECH and prepared data,
and writes `run.yaml`. It does not simulate, import accepted run history, or
rebuild the preparation. Wrong asset bytes refuse. The profile must satisfy the
[worker reservation rules](loam-execution.md), not just point at working libraries.

The reference flow starts in an empty store: classical construction/relaxation,
NVT/NPT, NAC sampling, candidate minimization, reactive construction,
thermalization/source selection, then flooding. Native libraries, the qualified
UMA provider and weights remain separately provisioned dependencies.

Execute through Loam with this equivalent ordinary command, then obtain the actual
operation/member IDs from status/result:

```bash
reactra run pmd-trial-001/run.yaml --store pmd-trial-001/store --request-id pmd-trial-001
reactra status --store pmd-trial-001/store --request-id pmd-trial-001
```

## What the PMD example establishes

The packaged PMD reaction is the **phosphorylated-substrate plus ADPH hypothesis**,
not complete PMD turnover. Its explicit draft MECH opt-in, geometry criteria,
sampling controls, and proton treatment are part of the experiment.

- Physical spectator-hydrogen restraints are declared in the initial YAML.
  The participating `mvap.H1` proton remains free to migrate. Do not apply a blanket
  hydrogen restraint or infer invalidity from any proton transfer alone.
- The discovery recipe disables sustained NAC residence with
  `minimum_sampled_occupancy_ps: null`. Individual target/integrity checks,
  minimization, and thermal source checks still apply. This is a visible scientific
  choice, not a universal PMD or new-enzyme setting.
- Aggressive flooding settings prioritize obtaining useful crossings quickly.
  Keep that discovery purpose explicit; do not claim qualified kinetics from it.
  Read actual target/endpoint checks and every member's outcome, including
  censored noncrossings and failures.

The documented ordinary reference completed in about 54 minutes on its Aurora
deployment, producing one verified crossing and one valid capped noncrossing.
This is historical guidance for budgeting, not a promise for a pool-scoped profile
with serial ensemble execution. The maintained
[successful-run guide](https://github.com/MechanismAtlas/reactra/blob/main/docs/guides/successful-flooding-runs.md)
owns the precise inputs, observations and limitations; use the version matching
the deployment. Those runs did not qualify an enzyme barrier.

## Another enzyme or another PMD step

Author its chemistry and scientific choices using the installed schema and MECH;
do not relabel the packaged PMD inputs. Preserve exact atom identities and explicit
cofactors/waters/charge. Select a supported workflow and requested endpoint. Keep
new methods or unmerged multi-step code out of an otherwise reproducible deployment
unless the task explicitly chooses and qualifies that revision.

Compare runs only with their declared differences visible. A successful software
run, a verified crossing, and an accurate free-energy estimate are separate claims.
