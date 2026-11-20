# EGFR scientific programs and input manifest

Only reusable programs, a generic assay contract and checksum identities are
versioned here. Scientific structures/annotations, preparation logs, original
research prompts, starting proposal texts and prior outputs remain in an
operator-supplied external bundle. Set `LOAM_SCIENCE_INPUTS` or pass `--inputs`;
use `scripts/prepare_scaling_inputs.py` to validate/copy an approved local bundle
outside the checkout. No data are downloaded or published by this helper.

The manifest identifies the 33 files used by the historical 256c assay. Its 11
Python programs are included. The public structure IDs are PDB 3NJP and UniProt
P00533; rebuilding a reference from those providers requires independent input
preparation/chemistry validation, rather than assuming arbitrary downloads have
the recorded hashes.

The registered assay screens three matched conditioning arms, then evaluates
eligible actual designed sequences in intact EGFR context with verified target
disulfides, minimization and single-frame MM/GBSA. It does not integrate production
MD. Exact scientific settings live in the programs and operational recipe book.
A new target/assay needs its own registration, manifest and scientific qualification.

The historical warm start used 24 proposal texts. They are not public source.
Supply an approved private `--seed-hypotheses` file when reproducing that warm
start; otherwise generate the initial pool from scratch. Neither proposal text,
rank nor prior context is fresh scientific evidence.
