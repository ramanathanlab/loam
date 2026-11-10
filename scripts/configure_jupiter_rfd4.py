#!/usr/bin/env python3
"""Set Jupiter's single-process RFD4 data-loading default, preserving other config."""
import argparse
import os
from pathlib import Path
import re


def configure(software_root: Path) -> Path:
    config = software_root / "RFD4-Proteina/configs/inference.yaml"
    original = config.read_text()
    updated, count = re.subn(
        r"(?m)^(num_workers:[ \t]*)[0-9]+([^\n]*)$",
        lambda match: match[1] + "0" + match[2],
        original,
    )
    if count != 1:
        raise ValueError(f"Expected one integer num_workers field in {config}; found {count}")
    if updated != original:
        config.write_text(updated)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--software-root", type=Path,
        default=Path(os.environ.get("LOAM_SOFTWARE_ROOT") or Path(__file__).resolve().parents[2] / "software"),
    )
    args = parser.parse_args()
    print(f"RFD4 Jupiter default num_workers=0: {configure(args.software_root)}")


if __name__ == "__main__":
    main()
