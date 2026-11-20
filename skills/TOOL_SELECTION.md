# Jupiter scientific skills

These skills target ARM64 GH200 CUDA workers on Jupiter. Source `scripts/jupiter_env.sh` by absolute path and activate each tool's dedicated environment under `LOAM_SOFTWARE_ROOT`. Keep the GPU visibility assigned by Ensemble Launcher. Use `read_skill(name, file=...)` to read packaged runnable recipes. Installed software is read-only; put outputs in the hypothesis workspace.

Core binder stages: `rfd4-proteina` generates backbones/binders; `proteinmpnn` designs sequences; `chai`, `boltz`, `protenix` and `esmfold2` predict structures; `openmm` and `molecular-simulations` provide CUDA simulation and AmberTools. `fetch-literature`, `rag-build-index` and `rag-query` supply literature workflows.

Known limitations from the existing Jupiter installation audit: RFdiffusion3 was disabled; Reactra is absent; PyRosetta lacks its licensed ARM64 runtime; UMA and GenSLM checkpoints require authorized access. Read each skill before selecting it. Historical measurements are not fresh validation of this checkout.

See [JUPITER.md](../JUPITER.md) for launch and verification.
