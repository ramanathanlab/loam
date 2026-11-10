#!/usr/bin/env python
"""Aurora/XPU launcher for Chai-1 that works around two env incompatibilities
WITHOUT modifying the chai-lab install:

1. chai_lab does `import intel_extension_for_pytorch` in many modules, but IPEX is
   not shipped in aurora_frameworks-2026.1.0 (torch has native XPU). The import is
   unused, so we inject an empty stub module.
2. The installed antipickle (0.2.2) registers a default "torch" adapter; chai_lab
   registers its own TorchAntipickleAdapter with the same typestring -> AssertionError
   while loading conformers_v1.apkl. We make register_adapter skip duplicates
   (user adapters are registered first, so chai's adapter wins).
3. pandas 3.x: `groupby("source_database").apply(...)` no longer passes the grouping
   column into the result, so MSA parsing (--use-msa-server / --msa-directory) fails
   with KeyError: 'source_database'. We apply the per-database quota ourselves
   (same semantics) and call the original parser with quota_sizes=None.
4. RDKit 2026.09 removed EmbedParameters.useChirality, so every SMILES ligand fails
   to tokenize ("Cannot set unknown attribute 'useChirality'") and is SILENTLY
   DROPPED from the complex (only an ERROR log line). We give chai's rdkit module a
   proxy rdDistGeom that maps useChirality -> enforceChirality and
   maxAttempts -> maxIterations (both removed in RDKit 2026.09).

Usage: identical to the `chai` CLI, e.g.
    python chai_xpu.py fold input.fasta out_dir --device xpu:0
"""
import sys
import types

if "intel_extension_for_pytorch" not in sys.modules:
    sys.modules["intel_extension_for_pytorch"] = types.ModuleType("intel_extension_for_pytorch")

import antipickle.base as _apb

_orig_register = _apb.TypeMatcher.register_adapter


def _register_skip_dupes(self, adapter):
    if adapter.typestring in self.adapters:
        return
    return _orig_register(self, adapter)


_apb.TypeMatcher.register_adapter = _register_skip_dupes

import os, tempfile  # noqa: E402
import pandas as pd  # noqa: E402
import chai_lab.data.parsing.msas.aligned_pqt as _apq  # noqa: E402
import chai_lab.data.dataset.msas.load as _msaload  # noqa: E402
from chai_lab.data.parsing.msas.data_source import (  # noqa: E402
    MSADataSource, msa_dataset_source_to_quota)

_orig_parse = _apq.parse_aligned_pqt_to_msa_context


def _parse_pandas3_safe(aligned_pqt_path, quota_sizes=msa_dataset_source_to_quota):
    if quota_sizes is None:
        return _orig_parse(aligned_pqt_path, quota_sizes=None)
    t = pd.read_parquet(aligned_pqt_path)
    rank = t.groupby("source_database", sort=False).cumcount()
    limit = t["source_database"].map(
        lambda x: quota_sizes.get(MSADataSource(x), 1_000_000) - 1)
    t = t[rank < limit].reset_index(drop=True)
    fd, tmp = tempfile.mkstemp(suffix=".aligned.pqt")
    os.close(fd)
    try:
        t.to_parquet(tmp)
        return _orig_parse(tmp, quota_sizes=None)
    finally:
        os.remove(tmp)


_apq.parse_aligned_pqt_to_msa_context = _parse_pandas3_safe
_msaload.parse_aligned_pqt_to_msa_context = _parse_pandas3_safe

import chai_lab.data.sources.rdkit as _chai_rdkit  # noqa: E402
from rdkit.Chem import rdDistGeom as _rdDG  # noqa: E402


_RENAMED = {"useChirality": "enforceChirality", "maxAttempts": "maxIterations"}


class _ParamsProxy:
    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def __setattr__(self, k, v):
        k = _RENAMED.get(k, k)
        setattr(self._real, k, v)

    def __getattr__(self, k):
        return getattr(self._real, k)


class _rdDGProxy:
    def __getattr__(self, k):
        return getattr(_rdDG, k)

    @staticmethod
    def ETKDGv3():
        return _ParamsProxy(_rdDG.ETKDGv3())

    @staticmethod
    def EmbedMultipleConfs(mol, *args, params=None, **kw):
        if isinstance(params, _ParamsProxy):
            params = params._real
        if params is not None:
            kw["params"] = params
        return _rdDG.EmbedMultipleConfs(mol, *args, **kw)


if not hasattr(_rdDG.ETKDGv3(), "useChirality"):
    _chai_rdkit.rdDistGeom = _rdDGProxy()

from chai_lab.main import cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(cli())
