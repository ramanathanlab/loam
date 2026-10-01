#!/usr/bin/env python3
"""Survey bacterial genomic GC composition, GC3, and synonymous codon usage.

This is a deliberately modest, reproducible public-data experiment. It obtains
annotated reference sequences from NCBI Nucleotide E-utilities, extracts CDSs
using Biopython feature locations, and respects each CDS's annotated translation
table. It writes only derived results and a provenance manifest; raw sequences
are fetched again on each run.

Usage:
    python3 experiments/codon_usage_gc_survey.py \
        --out-dir .loam/state/codon-usage-gc-survey

Requires: Biopython, NumPy, SciPy, Matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from Bio import SeqIO, __version__ as BIOPYTHON_VERSION
from Bio.Data import CodonTable
from scipy.stats import linregress, spearmanr


# Complete, annotated RefSeq nucleotide records selected to cover a broad GC
# gradient. This is a purposive comparison panel, not a random biological sample.
PANEL = (
    ("Mycoplasmoides genitalium G37", "NC_000908.2", "low-GC; code 4"),
    ("Staphylococcus aureus N315", "NC_002745.2", "low-GC Firmicute"),
    ("Streptococcus pneumoniae TIGR4", "NC_003028.3", "low-GC Firmicute"),
    ("Helicobacter pylori 26695", "NC_000915.1", "low/intermediate GC"),
    ("Bacillus subtilis 168", "NC_000964.3", "intermediate-GC Firmicute"),
    ("Escherichia coli K-12 MG1655", "NC_000913.3", "intermediate-GC Proteobacterium"),
    ("Mycobacterium tuberculosis H37Rv", "NC_000962.3", "high-GC Actinobacterium"),
    ("Pseudomonas aeruginosa PAO1", "NC_002516.2", "high-GC Proteobacterium"),
    ("Deinococcus radiodurans R1 chromosome 1", "NC_001263.1", "high-GC Deinococcota chromosome"),
    ("Caulobacter vibrioides CB15", "NC_002696.2", "high-GC Proteobacterium"),
    ("Streptomyces coelicolor A3(2)", "NC_003888.3", "very-high-GC Actinobacterium"),
    ("Thermus thermophilus HB8 chromosome 1", "NC_006461.1", "high-GC thermophile chromosome"),
)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
USER_AGENT = "LoamCodonUsageGCSurvey/1.0 (public reproducible analysis)"


def fetch_genbank(accession: str, retries: int = 5) -> Any:
    """Fetch one versioned NCBI nucleotide record with bounded backoff."""

    request_url = f"{EUTILS}?db=nuccore&id={accession}&rettype=gbwithparts&retmode=text"
    for attempt in range(retries):
        try:
            request = Request(request_url, headers={"User-Agent": USER_AGENT})
            with urlopen(request, timeout=90) as response:
                return SeqIO.read(StringIO(response.read().decode("utf-8")), "genbank")
        except (HTTPError, URLError, TimeoutError) as error:
            if attempt == retries - 1:
                raise RuntimeError(f"could not fetch {accession} after {retries} attempts") from error
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def gc_percent(sequence: str) -> float:
    return 100 * (sequence.count("G") + sequence.count("C")) / len(sequence)


def analyze_record(organism: str, requested_accession: str, rationale: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    record = fetch_genbank(requested_accession)
    genome = str(record.seq).upper()
    counts: Counter[str] = Counter()
    table_counts: Counter[int] = Counter()
    cds_total = 0
    cds_retained = 0
    cds_skipped_ambiguous = 0
    cds_skipped_length = 0

    for feature in record.features:
        if feature.type != "CDS":
            continue
        cds_total += 1
        coding_sequence = str(feature.extract(record.seq)).upper()
        if len(coding_sequence) % 3:
            cds_skipped_length += 1
            continue
        if set(coding_sequence) - set("ACGT"):
            cds_skipped_ambiguous += 1
            continue
        translation_table = int(feature.qualifiers.get("transl_table", ["11"])[0])
        table_counts[translation_table] += 1
        table = CodonTable.unambiguous_dna_by_id[translation_table]
        # Count only sense codons under each CDS's actual genetic code. This
        # handles code-4 TGA (Trp) correctly and excludes actual stop codons.
        for offset in range(0, len(coding_sequence), 3):
            codon = coding_sequence[offset : offset + 3]
            if codon in table.forward_table:
                counts[codon] += 1
        cds_retained += 1

    sense_codon_count = sum(counts.values())
    gc3_percent = 100 * sum(n for codon, n in counts.items() if codon[2] in "GC") / sense_codon_count

    # For each synonymous amino-acid family represented under the dominant code,
    # calculate within-family codon use. GC-ending fraction is meaningful only
    # for families that contain both GC- and AT-ending codons.
    dominant_table = table_counts.most_common(1)[0][0]
    code = CodonTable.unambiguous_dna_by_id[dominant_table]
    families: dict[str, list[str]] = defaultdict(list)
    for codon, amino_acid in code.forward_table.items():
        families[amino_acid].append(codon)

    codon_rows = []
    informative_family_gc_fractions = []
    for amino_acid, family in sorted(families.items()):
        family_total = sum(counts[codon] for codon in family)
        if not family_total:
            continue
        gc_ending = [codon for codon in family if codon[2] in "GC"]
        at_ending = [codon for codon in family if codon[2] in "AT"]
        if gc_ending and at_ending:
            informative_family_gc_fractions.append(sum(counts[codon] for codon in gc_ending) / family_total)
        for codon in sorted(family):
            codon_rows.append(
                {
                    "organism": organism,
                    "accession": record.id,
                    "translation_table": dominant_table,
                    "amino_acid": amino_acid,
                    "codon": codon,
                    "count": counts[codon],
                    "family_total": family_total,
                    "within_family_fraction": counts[codon] / family_total,
                    "RSCU": counts[codon] / (family_total / len(family)),
                }
            )

    row = {
        "organism": organism,
        "requested_accession": requested_accession,
        "retrieved_accession": record.id,
        "rationale": rationale,
        "genome_bp": len(genome),
        "genome_GC_percent": gc_percent(genome),
        "CDSs_total": cds_total,
        "CDSs_retained": cds_retained,
        "CDSs_skipped_ambiguous": cds_skipped_ambiguous,
        "CDSs_skipped_nontriplet": cds_skipped_length,
        "translation_tables": ";".join(f"{table}:{count}" for table, count in sorted(table_counts.items())),
        "sense_codon_count": sense_codon_count,
        "GC3_percent": gc3_percent,
        "mean_family_GC_ending_fraction": float(np.mean(informative_family_gc_fractions)),
        "informative_synonymous_families": len(informative_family_gc_fractions),
    }
    return row, codon_rows


def correlation_summary(rows: list[dict[str, Any]], x_key: str, y_key: str) -> dict[str, float | int]:
    x = np.array([float(row[x_key]) for row in rows])
    y = np.array([float(row[y_key]) for row in rows])
    pearson_slope, pearson_intercept, pearson_r, pearson_p, _ = linregress(x, y)
    spearman_rho, spearman_p = spearmanr(x, y)
    return {
        "n_genomes": len(rows),
        "pearson_r": float(pearson_r),
        "pearson_p_value": float(pearson_p),
        "slope": float(pearson_slope),
        "intercept": float(pearson_intercept),
        "r_squared": float(pearson_r**2),
        "spearman_rho": float(spearman_rho),
        "spearman_p_value": float(spearman_p),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], stats: dict[str, float | int], output: Path) -> None:
    x = np.array([float(row["genome_GC_percent"]) for row in rows])
    y = np.array([float(row["GC3_percent"]) for row in rows])
    slope, intercept = float(stats["slope"]), float(stats["intercept"])
    fig, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    axis.scatter(x, y, color="#1f77b4", s=50)
    line_x = np.linspace(x.min() - 1, x.max() + 1, 100)
    axis.plot(line_x, intercept + slope * line_x, color="#d62728", linewidth=1.5)
    label_offsets = {
        "Pseudomonas aeruginosa PAO1": (8, -14),
        "Deinococcus radiodurans R1 chromosome 1": (8, 6),
        "Caulobacter vibrioides CB15": (-108, -3),
        "Thermus thermophilus HB8 chromosome 1": (-128, 12),
        "Streptomyces coelicolor A3(2)": (-130, -15),
    }
    for row in rows:
        offset = label_offsets.get(row["organism"], (4, 4))
        axis.annotate(
            row["organism"],
            (row["genome_GC_percent"], row["GC3_percent"]),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
            arrowprops={"arrowstyle": "-", "color": "#777777", "linewidth": 0.5}
            if row["organism"] in label_offsets
            else None,
        )
    axis.set(xlabel="Whole-genome GC (%)", ylabel="Sense-codon GC3 (%)", title="Bacterial GC composition and synonymous third positions")
    axis.text(0.02, 0.98, f"n = {stats['n_genomes']}; Pearson r = {stats['pearson_r']:.3f}; Spearman ρ = {stats['spearman_rho']:.3f}", transform=axis.transAxes, va="top")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def markdown_report(rows: list[dict[str, Any]], stats: dict[str, float | int]) -> str:
    table_rows = "\n".join(
        f"| {row['organism']} | {row['retrieved_accession']} | {row['genome_GC_percent']:.2f} | {row['GC3_percent']:.2f} | {row['mean_family_GC_ending_fraction']:.3f} | {row['CDSs_retained']} | {row['translation_tables']} |"
        for row in rows
    )
    return f"""# GC composition and synonymous codon-usage survey

## Research question
Across a purposive panel of 12 annotated bacterial reference chromosomes/genomes, how strongly is whole-genome GC composition associated with GC at synonymous coding third positions (GC3) and with within-family preference for GC-ending codons?

## Hypotheses and outcomes
- **Primary hypothesis:** genomes with higher whole-genome GC% have higher sense-codon GC3%.
- **Primary outcome:** `GC3_percent`, the fraction of all counted sense codons whose third base is G or C.
- **Secondary outcome:** `mean_family_GC_ending_fraction`, the unweighted mean, across synonymous amino-acid families containing both GC- and AT-ending codons, of the fraction of that family represented by GC-ending codons. This reduces—but does not eliminate—amino-acid-composition confounding.

## Provenance and scope
- Source: NCBI Nucleotide E-utilities (`gbwithparts`), fetched at `{datetime.now(timezone.utc).date().isoformat()}`.
- The panel is intentionally GC-diverse rather than randomly sampled. Each row in `genome_summary.csv` records a versioned requested and retrieved accession.
- CDSs were extracted with their annotated locations/orientations. CDSs with ambiguous bases or lengths not divisible by three were excluded; each triplet was classified by that CDS's annotated `transl_table`. Stops were excluded from GC3 and codon-usage denominators.

## Results

| Organism | Accession | Genome GC% | GC3% | Mean family GC-ending fraction | CDSs retained | Translation table(s) |
|---|---|---:|---:|---:|---:|---|
{table_rows}

Primary association (n = {stats['n_genomes']}): Pearson r = {stats['pearson_r']:.4f}, R² = {stats['r_squared']:.4f}, two-sided parametric p = {stats['pearson_p_value']:.3g}; Spearman ρ = {stats['spearman_rho']:.4f}, p = {stats['spearman_p_value']:.3g}. The fitted descriptive slope is {stats['slope']:.3f} GC3 percentage points per one percentage point of genomic GC.

## Interpretation
The analysis estimates a strong compositional association in this deliberately broad panel. It is consistent with genome-wide nucleotide-composition bias shaping synonymous third-position and codon-family usage. It does **not** by itself show that mutation bias is the only mechanism, nor that individual codons are selected solely by GC composition.

## Validity checks and limitations
1. The QC columns report CDS exclusion counts, total sense codons, and translation tables; `M. genitalium` uses table 4, so TGA is treated as tryptophan rather than an excluded stop.
2. Organisms are not independent biological replicates: phylogeny, ecology, genome reduction, horizontal transfer, tRNA pools, and expression selection are possible confounders.
3. The p-values assume independent observations and should be treated as descriptive/exploratory; this panel was deliberately selected for GC range.
4. Some accessions represent a chromosome rather than a complete multi-replicon genome. The analysis reports the exact analyzed sequence rather than silently combining records.
5. Extension: sample one randomly selected complete reference genome per genus, analyze all replicons, and use phylogenetically informed regression.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--request-delay", type=float, default=1.0, help="seconds between NCBI records")
    arguments = parser.parse_args()
    arguments.out_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    codon_rows: list[dict[str, Any]] = []
    for number, (organism, accession, rationale) in enumerate(PANEL, start=1):
        print(f"[{number}/{len(PANEL)}] {organism} ({accession})", file=sys.stderr, flush=True)
        summary, per_codon = analyze_record(organism, accession, rationale)
        summaries.append(summary)
        codon_rows.extend(per_codon)
        if number < len(PANEL):
            time.sleep(arguments.request_delay)

    summaries.sort(key=lambda row: float(row["genome_GC_percent"]))
    stats = correlation_summary(summaries, "genome_GC_percent", "GC3_percent")
    write_csv(arguments.out_dir / "genome_summary.csv", summaries)
    write_csv(arguments.out_dir / "codon_usage.csv", codon_rows)
    plot(summaries, stats, arguments.out_dir / "gc_vs_gc3.png")
    (arguments.out_dir / "statistics.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "analysis": "bacterial GC composition and codon-usage survey",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "NCBI Nucleotide E-utilities efetch; db=nuccore; rettype=gbwithparts; retmode=text",
        "panel": [{"organism": o, "accession": a, "rationale": r} for o, a, r in PANEL],
        "software": {"python": sys.version.split()[0], "biopython": BIOPYTHON_VERSION, "numpy": np.__version__},
        "parameters": {"request_delay_seconds": arguments.request_delay, "retry_count": 5},
    }
    (arguments.out_dir / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (arguments.out_dir / "REPORT.md").write_text(markdown_report(summaries, stats), encoding="utf-8")
    print(json.dumps({"out_dir": str(arguments.out_dir), "n_genomes": len(summaries), **stats}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
