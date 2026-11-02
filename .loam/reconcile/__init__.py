"""Reconcilers: the step that turns reviewed candidates into the ledgers.

This file exists to make `.loam/reconcile/` a package, so a hook can import a
reconciler as `reconcile.hypothesis_provenance` instead of loading it by path.
`hooks/hypothesis_provenance.py` is a different module with the same basename --
the contract parser, not the reconciler -- and without the package qualifier the
two collide in `sys.modules`.

Each module here also runs as a standalone script (`python
.loam/reconcile/hypothesis_provenance.py --help`), which is still the supported
way to reconcile by hand; they put `hooks/` on `sys.path` themselves for that.
"""
