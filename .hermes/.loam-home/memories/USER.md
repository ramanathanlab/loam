User prefers biology experiment suggestions to be in silico and executable by the assistant, rather than wet-lab experiments.
§
User prefers end-to-end Loam/Hermes campaign tests to run in isolated temporary state/config directories and to fail closed with explicit first-unmet acceptance criteria rather than simulated completion.
§
User expects requested campaign tests to be actively executed end-to-end, with clear failure reporting only after the actual acceptance path has been attempted; avoid stopping at plans, preflight, or partial verification.
§
User expects campaign tasks to continue autonomously until the requested multi-round acceptance path has actually run or a concrete external blocker prevents it; do not stop at partial progress or summaries.
§
User expects explicit permission for requested Polaris/remote-execution work to be treated as authorization to proceed without repeated permission questions or unnecessary pauses; verify state and continue autonomously within the requested scope.