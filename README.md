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

The hook integration and its runtime state are under `.loam/`. See `.loam/README.md` for lifecycle-hook behavior and optional global Hermes installation.

## Run the GC/codon survey

The standalone survey downloads 12 public NCBI nucleotide records, computes genomic GC, GC3, and synonymous codon usage, then writes derived artifacts to the selected directory. It requires Python 3 with Biopython, NumPy, SciPy, and Matplotlib:

```bash
python3 -m pip install biopython numpy scipy matplotlib
python3 experiments/codon_usage_gc_survey.py \
  --out-dir .loam/state/codon-usage-gc-survey
```

The output directory receives `genome_summary.csv`, `codon_usage.csv`, `gc_vs_gc3.png`, `statistics.json`, `provenance.json`, and `REPORT.md`. The default path is ignored local runtime state; use another writable directory to retain results elsewhere.
