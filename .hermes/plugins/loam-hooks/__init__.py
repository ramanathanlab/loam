"""Hermes discovery shim; Loam's real adapter lives under `.loam/`."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ADAPTER = (
    Path(__file__).resolve().parents[3]
    / ".loam"
    / "adapters"
    / "hermes"
    / "plugin"
    / "__init__.py"
)
_SPEC = importlib.util.spec_from_file_location("loam_hermes_adapter", _ADAPTER)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load Loam Hermes adapter at {_ADAPTER}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
register = _MODULE.register
