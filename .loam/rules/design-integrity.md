---
name: design-integrity
scope: all
enforced_by: partially -- _SCRUBBERS strips filesystem-recovery steering from validator instructions; the rest is advisory
---

- External reference data — anything published rather than designed in this
  campaign — has an authoritative source. Fetch it from that source by
  identifier at run time (for a protein: a UniProt accession plus residue range,
  or a PDB ID) and verify what came back: the identifier matches, the size and
  format are sane, the expected region or feature is present.
- Never reproduce a reference datum from memory, and never accept one that was.
  A hardcoded or unverifiable reference datum is a blocked result, not a
  validation.
- A designed datum — a candidate sequence, a proposed length or topology —
  originates in the hypothesis and exists nowhere else, so it must travel in full
  rather than as an identifier. Paste what was designed; fetch what was
  published.
- When the hypothesis gives only a design concept (length, topology, motifs) and
  no exact sequence, use a design tool to construct one or a few concrete
  candidates realizing that architecture, and write them to disk under the
  working directory. Do not invent a sequence to stand in for a design step.
- A failed, unfinished, or unexecuted run is not evidence. Say so rather than
  reporting a plausible number, and never substitute unexecuted code or
  reasoning for a measurement.
