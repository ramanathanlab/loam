# jnana

This is the spiritual successor of StructBioReasoner, wherein we have condensed the execution details into a set of skills that are empowered with the hypothesis tournament framework of Jnana and the hierarchical memory system of Pando.

## Run with Loam

This repository is run through Hermes with the checked-in Loam lifecycle plugin. Install the `hermes` CLI, configure its model provider and credentials, then launch from the repository root:

```bash
.loam/run-hermes
```

The wrapper changes to the repository root, enables project-plugin discovery, activates Loam, and uses the isolated Hermes home at `.hermes/.loam-home`. It enables the `loam-hooks` plugin and invokes the pre-LLM hook every five turns by default.

Set `LOAM_PRE_LLM_EVERY` to change that interval; for example, run it every three turns:

```bash
LOAM_PRE_LLM_EVERY=3 .loam/run-hermes
```

Set `LOAM_STATE_DIR` for deployment-specific outputs; it overrides the configured
state directory and keeps lifecycle, orchestration, Jnana state, and tournament
reports isolated:

```bash
LOAM_STATE_DIR=/var/tmp/loam/${DEPLOYMENT_ID} .loam/run-hermes
```


The hook integration and its runtime state are under `.loam/`. See `.loam/README.md` for lifecycle-hook behavior and optional global Hermes installation.

Campaigns use two best-effort services, both on by default and never able to fail or stall a step: Cairn tool reputation (`docs/cairn.md`) and FALDA campaign memory (`docs/falda.md`).
## Verify Loam orchestration

The focused verification suite exercises the ranked-hypothesis lifecycle without a live MCP execution service: it loads the three prompt roles, verifies validator-to-code-generator tool boundaries, simulates tagged subagent start/stop records, collates completed validator reports into a reasoner request, and confirms the reasoner brief reaches the next tournament goal. It also covers tournament profile parsing and the existing lifecycle/provenance ledger contracts.

Run it from the repository root, using a Python environment that has Loam's
dependencies installed (PyYAML in particular — without it `load_profile` returns
`None` and the tournament profile-parsing tests fail; the tournament suite also
imports `jnana`, which imports `yaml` at module load). Use the same interpreter
the campaign runs under, e.g. the Loam virtualenv:

```bash
/Users/architvasan/.loam-venv-312/bin/python -m unittest \
  tests.test_hypothesis_orchestration \
  tests.test_hypothesis_tournament \
  tests.test_subagent_ledger -v
```

Substitute your own interpreter if it differs; a bare `python3` works only if
`pip install -e .` (or at least `pyyaml`) has been run in it.

This is deterministic fixture coverage. It does not invoke a real harness subagent, MCP execution endpoint, or inference endpoint.


## Run the GC/codon survey

The standalone survey downloads 12 public NCBI nucleotide records, computes genomic GC, GC3, and synonymous codon usage, then writes derived artifacts to the selected directory. It requires Python 3 with Biopython, NumPy, SciPy, and Matplotlib:

```bash
python3 -m pip install biopython numpy scipy matplotlib
python3 experiments/codon_usage_gc_survey.py \
  --out-dir .loam/state/codon-usage-gc-survey
```

The output directory receives `genome_summary.csv`, `codon_usage.csv`, `gc_vs_gc3.png`, `statistics.json`, `provenance.json`, and `REPORT.md`. The default path is ignored local runtime state; use another writable directory to retain results elsewhere.
