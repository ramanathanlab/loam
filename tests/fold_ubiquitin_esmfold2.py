"""Fold human ubiquitin once with ESMFold2 (https://github.com/Biohub/esm).

Setup:
    pip install esm            # esm >= 3.4 ships ESMFold2
    huggingface-cli login      # weights are pulled from biohub/ESMFold2

Run:
    python fold_ubiquitin_esmfold2.py [--out ubiquitin_esmfold2.cif]
"""

from __future__ import annotations

import argparse
import time

from esm.models.esmfold2 import (
    ESMFold2InputBuilder,
    EsmFold2Model,
    ProteinInput,
    StructurePredictionInput,
)

# Human ubiquitin, UniProt P0CG48 (76 aa).
UBIQUITIN = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="ubiquitin_esmfold2.cif", help="output mmCIF path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model = EsmFold2Model.from_pretrained("biohub/ESMFold2", device=args.device).eval()

    spi = StructurePredictionInput(sequences=[ProteinInput(id="A", sequence=UBIQUITIN)])

    t0 = time.perf_counter()
    result = ESMFold2InputBuilder().fold(
        model,
        spi,
        num_loops=20,
        num_sampling_steps=100,
        num_diffusion_samples=1,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - t0

    with open(args.out, "w") as fh:
        fh.write(result.complex.to_mmcif())

    print(f"wrote {args.out}")
    print(f"pLDDT: {result.plddt:.3f}  pTM: {result.ptm:.3f}  time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
