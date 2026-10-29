"""Aurora compat shim for RFD4-Proteina.

The RFD4-Proteina .venv has lightning 2.3.0dev, whose seed_everything() has no
`verbose` kwarg, but src/rfproteina/generate.py calls
L.seed_everything(cfg.seed, verbose=...). Drop the unsupported kwarg instead of
patching the (read-only) software tree. Enable with:
    PYTHONPATH=<this dir>:$PYTHONPATH
"""
try:
    import inspect

    import lightning as _L

    _orig = _L.seed_everything
    if "verbose" not in inspect.signature(_orig).parameters:

        def seed_everything(seed=None, workers=False, verbose=True, **kw):
            return _orig(seed, workers=workers)

        _L.seed_everything = seed_everything
        try:
            import lightning.fabric.utilities.seed as _s

            _s.seed_everything = seed_everything
        except Exception:
            pass
except Exception:
    pass

# --- Enable lightning's XPUAccelerator without IPEX -------------------------
# frameworks/2026.1.0 ships torch 2.13 with *native* XPU support and no
# intel_extension_for_pytorch. The venv's Intel-patched lightning 2.3.0dev
# gates XPUAccelerator on a RequirementCache for IPEX, so
# XPUAccelerator.is_available() is False and Trainer(accelerator="gpu") (which
# generate.py hardcodes) raises
#   MisconfigurationException: No supported gpu backend found!
# Unblock it: the class body only ever calls torch.xpu.* anyway.
try:
    import torch as _torch

    from lightning.fabric.accelerators import xpu as _fxpu
    from lightning.pytorch.accelerators import xpu as _xpu

    if _torch.xpu.is_available():
        for _m in (_xpu, _fxpu):
            _m._IPEX_AVAILABLE = True
            _m.XPUAccelerator.__init__ = lambda self, *a, **k: None
            _m.XPUAccelerator.is_available = staticmethod(lambda: _torch.xpu.is_available())
            _m.num_xpu_devices = lambda: _torch.xpu.device_count()
        # device_parser imports this by name at call time
        _fxpu._get_all_visible_xpu_devices = lambda: list(range(_torch.xpu.device_count()))
except Exception:
    pass

# --- rich >= 14 vs lightning 2.3 RichProgressBar ---------------------------
# frameworks/2026.1.0 provides a newer `rich` than the venv's lightning expects.
# RichProgressBar._init_progress() calls Console.clear_live() on a console with
# no live display, and the newer rich pops an empty stack:
#   IndexError: pop from empty list
# (hit via the repo's GenerationProgressBar during trainer.validate()).
try:
    from rich.console import Console as _Console

    _orig_clear_live = _Console.clear_live

    def _safe_clear_live(self):
        try:
            return _orig_clear_live(self)
        except IndexError:
            return None

    _Console.clear_live = _safe_clear_live
except Exception:
    pass
