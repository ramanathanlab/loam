# Empty stand-in for intel_extension_for_pytorch (IPEX).
# ProteinMPNN's Aurora port does `import intel_extension_for_pytorch as ipex` but never
# calls it; native torch.xpu (torch>=2.5) provides the XPU backend. IPEX is not shipped
# in the aurora_frameworks-2026.1.0 stack, so this shim lets the import succeed.
__version__ = "0.0.0-shim"
