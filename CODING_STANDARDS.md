# Coding standards

Judgement-call rules for loam: the ones a linter cannot check. Lint and style rules belong to ruff (`[tool.ruff]` in `pyproject.toml`, run by CI), not here. Each rule points to where loam already states it; that source is authoritative, and this file only collects them.

1. **A component failure costs at most one hypothesis, never the round.** A failed validator, reviewer or code-generator marks that one hypothesis untested; a failed tournament or reasoner falls back to the previous state. Only an empty hypothesis pool, or misconfiguration caught at startup, stops a campaign.
   Sources: `loam/orchestrator.py` module docstring ("Failure policy"); for startup, the `memory.start()` comment in `Campaign.run` (`loam/orchestrator.py`), `check_mcp_tools` in `loam/execution.py` ("Fail once at startup"), and README, "Memory" ("Misconfiguration fails fast, before round 1").

2. **An agent's capabilities come only from the toolsets `agents.build` constructs, never from prompt wording.** If a role must not do something, it must not have a tool that does it; telling it not to in a prompt is not a restriction.
   Source: `loam/execution.py` module docstring ("Restriction is by construction"); README, "Why `loam`?".

3. **Every agent output is a strict pydantic model (`extra='forbid'`).** No free-form dicts or loosely parsed text cross an agent boundary.
   Source: README, "Why `loam`?"; models in `loam/schemas.py`.

4. **External state is written only by orchestrator code, never by an agent through a tool.** This covers the ledger, lesson memory and workspace receipts.
   Sources: lesson memory, `docs/adr/0001-lesson-memory-is-advice-never-evidence.md`; the ledger, `reconcile.record` called from `loam/orchestrator.py`; workspace receipts, `OwnedMCP` in `loam/execution.py` ("save a receipt for every batch submitted") and `Workspace` in `loam/workspace.py`.

5. **Round behaviour is tested through `Campaign`, with scripted `FunctionModel`s and injected fakes.** Tests use no real models, network or Jnana checkout.
   Source: `tests/test_round.py`, `tests/test_memory.py`.

6. **Names and docs use the terms in `GLOSSARY.md`.** A word listed there under _Avoid_ is a finding unless it names a different, existing concept (EL's "cluster mode" is not a System).
   Source: `GLOSSARY.md`.
