# Carried Hermes patches

Patches to the **Hermes** install (`hermes_agent`, an editable checkout at
`~/.hermes/hermes-agent`) that Loam campaigns depend on. Hermes is a separate
codebase from this repo; a reinstall or `herdr update` of Hermes can silently
drop these, so Loam carries them here and `scripts/campaign_preflight.py` probes
for them at every launch.

## `hermes-role-models-endpoints.patch`

**What it does.** Extends `delegation.role_models` in Hermes' `delegate_task`
(`tools/delegate_tool.py`) so a per-role entry may be a **mapping** carrying its
own `base_url` / `provider` / `api_key` / `api_mode`, not just a model string.
Each delegated child role (`code-generator`, `hypothesis-validator`, `reasoner`,
`evidence-reviewer`) can then run on a **different endpoint**, resolved through
the same `_resolve_delegation_credentials` path as a top-level delegation.

**Why Loam needs it.** It is the lever that **isolates the code generator**:
hold the validator/reasoner/reviewer on the default model+endpoint and point
*only* the code-generator at another model on another ALCF cluster — the
clean independent variable for a code-gen-model benchmark. A plain model-string
entry (the original behavior) still works unpatched; a mapping entry does not —
unpatched Hermes **silently ignores it** and leaves that role on the default
endpoint, which would corrupt the comparison without any error. That silent
failure is why preflight hard-fails when the config needs the patch and the
installed Hermes lacks it.

**Backward compatible.** Inert unless `delegation.role_models` is set; string
entries behave exactly as before.

### Config example (`config.yaml`)

```yaml
delegation:
  model: argo:claude-opus-4.8            # default for any unmapped role
  base_url: http://localhost:51900/v1    # default endpoint (Argo tunnel)
  api_key: local-no-auth
  role_models:
    # Isolated code-generator: different model AND endpoint (ALCF cluster proxy).
    code-generator:
      model: mistral-large-4-nvfp4
      base_url: http://127.0.0.1:51911/v1
      api_key: local-no-auth
    # Held on the default via the legacy string form.
    hypothesis-validator: argo:claude-opus-4.8
    # reasoner / evidence-reviewer: unset -> inherit delegation.model+endpoint.
```

### Apply

```bash
LOAM=/path/to/loam                       # this repo's checkout
cd ~/.hermes/hermes-agent
git apply "$LOAM/patches/hermes-role-models-endpoints.patch"
# verify:
./venv/bin/python -c "import tools.delegate_tool as d; print(hasattr(d,'_apply_role_override'))"  # -> True
```

If `git apply` refuses because the surrounding code has moved, apply with
`git apply --3way` or re-create the edit by hand (two spots in
`tools/delegate_tool.py`: the `_single_role_from_markers` / `_role_override_entry`
/ `_apply_role_override` helpers near `_normalize_role`, and the call site in
`delegate_task` that replaced `creds["model"] = _role_model_override(...)`).

### Detection

`scripts/campaign_preflight.py::check_role_endpoint_capability` resolves the
Hermes venv from the `hermes` launcher and checks for `_apply_role_override`.
When `config.yaml` uses a mapping (endpoint) entry and Hermes lacks the symbol,
the launch fails with instructions pointing back here. An inconclusive probe
(e.g. `hermes` not on `PATH`) never blocks a launch.
