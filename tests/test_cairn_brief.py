"""Tests for the Cairn reputation brief appended to campaign delegations."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".loam" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

import cairn_brief  # noqa: E402
import cairn_client  # noqa: E402
from cairn_brief import SECTION_END, SECTION_START, appendix, decorate_delegation, reputation  # noqa: E402

SCORES = [{"composite_score": 0.62, "confidence": 0.41}, {"composite_score": 0.5, "confidence": 0.0}]


def _cairn(**overrides):
    block = {
        "enabled": True,
        "base_url": "https://cairn.invalid",
        "project": "loam-test-project",
        "read_side": True,
        "rate_agent_reports": True,
        "rate_roles": ["code-generator", "hypothesis-validator"],
        "timeout_seconds": 5.0,
        "executor": "mcp://hpc#run_code",
        "entities": {
            "tool://lammps": {"patterns": ["lmp"], "task_tags": ["md_simulation"]},
            "mcp://hpc#run_code": {"patterns": [], "task_tags": ["code_exec"]},
        },
    }
    block.update(overrides)
    return block


@pytest.fixture
def payload(tmp_path):
    return {"loam_state_dir": str(tmp_path), "repo_root": str(ROOT), "loam_cairn": _cairn()}


def payload_with(payload, **overrides):
    return {**payload, "loam_cairn": _cairn(**overrides)}


class Fake:
    def __init__(self, result):
        self.calls = 0
        self.result = result

    def __call__(self, base_url, refs, timeout, project=None):
        self.calls += 1
        self.refs = refs
        self.project = project
        return self.result


@pytest.fixture
def fake_score_batch(monkeypatch):
    fake = Fake(cairn_client.CairnResult(True, 200, SCORES, "", echo="ok"))
    monkeypatch.setattr(cairn_client, "score_batch", fake)
    return fake


@pytest.fixture
def fake_score_batch_down(monkeypatch):
    fake = Fake(cairn_client.CairnResult(False, 0, None, "unreachable"))
    monkeypatch.setattr(cairn_client, "score_batch", fake)
    return fake


def test_reputation_fetches_once_per_plan(payload, fake_score_batch):
    assert reputation(payload, "plan-a")["tool://lammps"] == {"composite_score": 0.62, "confidence": 0.41}
    reputation(payload, "plan-a")
    reputation(payload, "plan-a")
    assert fake_score_batch.calls == 1
    assert fake_score_batch.refs[0] == {"type": "capability", "external_id": "tool://lammps"}
    reputation(payload, "plan-b")
    assert fake_score_batch.calls == 2


def test_failed_fetch_is_cached_and_returns_none(payload, fake_score_batch_down):
    assert reputation(payload, "plan-a") is None
    assert reputation(payload, "plan-a") is None and fake_score_batch_down.calls == 1


def test_malformed_response_returns_none(payload, monkeypatch):
    monkeypatch.setattr(cairn_client, "score_batch", Fake(cairn_client.CairnResult(True, 200, [SCORES[0]], "", echo="ok")))
    assert reputation(payload, "plan-a") is None


def test_unwritable_state_dir_does_not_raise(payload, fake_score_batch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    bad = {**payload, "loam_state_dir": str(blocker / "sub")}
    assert reputation(bad, "plan-a") is None or isinstance(reputation(bad, "plan-a"), dict)


def test_appendix_for_code_generator(payload, fake_score_batch):
    text = appendix(payload, "plan-a", "code-generator")
    assert "tool://lammps: composite 0.62, confidence 0.41" in text
    assert "mcp://hpc#run_code: no signal yet" in text
    assert "Tool ids you may rate: tool://lammps, mcp://hpc#run_code" in text
    assert "### Tool Ratings" in text and "Put ### Tool Ratings first" in text
    assert "Reputation is a prior from past runs" in text
    assert "[loam-" not in text


def test_reasoner_gets_nothing_by_default(payload, fake_score_batch):
    # Spec 7: the brief goes to code-generator and validator contexts only; the reasoner's synthesis
    # becomes Jnana steering, so reputation must not reach it.
    assert appendix(payload, "plan-a", "reasoner") == ""
    assert fake_score_batch.calls == 0


def test_reasoner_placement_when_enabled(payload, fake_score_batch):
    text = appendix(payload_with(payload, rate_roles=["reasoner"]), "plan-a", "reasoner")
    assert "Put ### Tool Ratings last in your synthesis." in text and "Tool ids you may rate" in text
    assert "Tool reputation from Cairn" not in text and "Reputation is a prior" not in text
    assert fake_score_batch.calls == 0


def test_brief_goes_to_code_generator_and_validator(payload, fake_score_batch):
    for role in ("code-generator", "hypothesis-validator"):
        assert "Tool reputation from Cairn" in appendix(payload, "plan-a", role), role
    assert appendix(payload, "plan-a", "coordinator") == ""


def test_validator_placement_does_not_conflict_with_exact_headings(payload, fake_score_batch):
    text = appendix(payload, "plan-a", "hypothesis-validator")
    assert ("In addition to the required headings, add ### Tool Ratings after ### Outcome "
            "and before ### Evidence Score.") in text


def test_read_side_off_omits_brief_keeps_rating_request(payload, fake_score_batch):
    text = appendix(payload_with(payload, read_side=False), "plan-a", "hypothesis-validator")
    assert "Tool reputation" not in text and "before ### Evidence Score" in text and fake_score_batch.calls == 0


def test_failed_fetch_keeps_rating_request(payload, fake_score_batch_down):
    text = appendix(payload, "plan-a", "code-generator")
    assert "Tool reputation" not in text and "Tool ids you may rate" in text


def test_missing_skill_file_still_lists_scores_and_ids(payload, fake_score_batch, tmp_path):
    bare = {**payload, "repo_root": str(tmp_path / "nowhere")}
    text = appendix(bare, "plan-a", "code-generator")
    assert "tool://lammps: composite 0.62" in text and "Tool ids you may rate:" in text
    assert "Reputation is a prior" not in text and "Score anchors" not in text


def test_decorate_appends_to_tagged_tasks_only(payload, fake_score_batch):
    args = {"tasks": [
        {"goal": "x [loam-agent-role=hypothesis-validator] [loam-plan-id=plan-a] [loam-hypothesis-id=h1]",
         "context": '{"loam_validator_contract": "v1"}', "role": "orchestrator"},
        {"goal": "untagged", "context": "c"}]}
    out = decorate_delegation(args, payload)
    assert out["tasks"][0]["context"].startswith('{"loam_validator_contract": "v1"}\n\n')
    assert '"loam_validator_contract": "v1"' in out["tasks"][0]["context"]
    assert out["tasks"][0]["goal"] == args["tasks"][0]["goal"]
    assert out["tasks"][1] == args["tasks"][1]
    assert args["tasks"][0]["context"] == '{"loam_validator_contract": "v1"}'


def test_decorate_single_task_shape(payload, fake_score_batch):
    out = decorate_delegation({"goal": "Run [loam-agent-role=code-generator] [loam-plan-id=plan-a]", "context": "d"}, payload)
    assert out["context"].startswith("d\n\n")


def test_decorate_leaves_reasoner_untouched_by_default(payload, fake_score_batch):
    task = {"goal": "Synthesize [loam-agent-role=reasoner] [loam-plan-id=plan-a]", "context": "d"}
    assert decorate_delegation(task, payload) == task


def test_decorate_without_cairn_is_identity():
    args = {"tasks": [{"goal": "[loam-agent-role=code-generator] [loam-plan-id=p]", "context": "c"}]}
    assert decorate_delegation(args, {"loam_cairn": None}) == args


def test_decorate_never_raises(payload, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr(cairn_brief, "appendix", boom)
    args = {"tasks": [{"goal": "[loam-agent-role=code-generator] [loam-plan-id=p]", "context": "c"}]}
    assert decorate_delegation(args, payload) == args


def test_adapter_applies_decoration(monkeypatch, payload, fake_score_batch):
    path = ROOT / ".loam" / "adapters" / "hermes" / "plugin" / "__init__.py"
    spec = importlib.util.spec_from_file_location("loam_hermes_adapter_cairn_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    task = {"goal": "run [loam-agent-role=code-generator] [loam-plan-id=plan-a]", "context": "ctx"}
    stub = types.ModuleType("hypothesis_orchestration")
    stub.canonical_delegation_args = lambda p: {"tasks": [dict(task)]}
    monkeypatch.setitem(sys.modules, "hypothesis_orchestration", stub)
    monkeypatch.setattr(module, "_profile", lambda: types.SimpleNamespace(hypothesis_orchestration=True))
    monkeypatch.setattr(module, "payload_values", lambda profile: {
        "loam_state_dir": payload["loam_state_dir"], "loam_cairn": payload["loam_cairn"]})

    result = module.canonicalize_campaign_delegation("delegate_task", {})
    assert "Tool reputation from Cairn" in result["args"]["tasks"][0]["context"]


def test_round_dispatch_applies_decoration(monkeypatch, payload, fake_score_batch):
    # loam_dispatch_round hands the plan straight to delegate_task through ctx.dispatch_tool, so the
    # tool_request middleware (canonicalize_campaign_delegation) never sees it: it must decorate too.
    path = ROOT / ".loam" / "adapters" / "hermes" / "plugin" / "__init__.py"
    spec = importlib.util.spec_from_file_location("loam_hermes_adapter_cairn_dispatch_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    task = {"goal": "x [loam-agent-role=hypothesis-validator] [loam-plan-id=plan-a]", "context": "ctx"}
    stub = types.ModuleType("hypothesis_orchestration")
    stub.canonical_delegation_args = lambda p: {"tasks": [dict(task)]}
    monkeypatch.setitem(sys.modules, "hypothesis_orchestration", stub)
    monkeypatch.setattr(module, "_profile", lambda: types.SimpleNamespace(hypothesis_orchestration=True))
    monkeypatch.setattr(module, "payload_values", lambda profile: {
        "loam_state_dir": payload["loam_state_dir"], "loam_cairn": payload["loam_cairn"]})
    dispatched = []
    context = types.SimpleNamespace(dispatch_tool=lambda name, args: dispatched.append((name, args)) or "ok")
    monkeypatch.setattr(module, "_PLUGIN_CONTEXT", context)

    assert module.dispatch_round({}) == "ok"
    [(name, args)] = dispatched
    assert name == "delegate_task"
    assert "### Tool Ratings" in args["tasks"][0]["context"]


def test_nonempty_appendix_is_delimited_exactly_once(payload, fake_score_batch):
    text = appendix(payload, "plan-a", "code-generator")
    assert text.startswith(SECTION_START + "\n\n") and text.endswith("\n\n" + SECTION_END)
    assert text.count(SECTION_START) == 1 and text.count(SECTION_END) == 1
    assert "[loam-" not in SECTION_START + SECTION_END


def test_empty_appendix_has_no_section(payload, fake_score_batch):
    assert appendix(payload, "plan-a", "reasoner") == ""
    assert appendix(payload_with(payload, enabled=False), "plan-a", "code-generator") == ""


def test_decorated_validator_context_keeps_original_before_section(payload, fake_score_batch):
    original = '{"loam_validator_contract": "v1"}'
    args = {"tasks": [{"goal": "x [loam-agent-role=hypothesis-validator] [loam-plan-id=plan-a]", "context": original}]}
    ctx = decorate_delegation(args, payload)["tasks"][0]["context"]
    assert '"loam_validator_contract": "v1"' in ctx
    assert ctx.startswith(original + "\n\n" + SECTION_START)
    assert ctx.index(SECTION_START) >= len(original) and ctx.count(SECTION_START) == 1
    assert ctx.endswith(SECTION_END)


def test_brief_reads_the_configured_project_and_notes_the_silo(payload, fake_score_batch):
    text = appendix(payload, "plan-a", "code-generator")
    assert fake_score_batch.project == "loam-test-project"
    assert cairn_brief.SILO_NOTE in text


@pytest.mark.parametrize("echo", ["missing", "mismatch"])
def test_scores_without_a_matching_echo_are_not_shown(payload, monkeypatch, echo):
    monkeypatch.setattr(cairn_client, "score_batch", Fake(cairn_client.CairnResult(True, 200, SCORES, "", echo=echo)))
    assert reputation(payload, "plan-a") is None
    assert "Tool reputation" not in appendix(payload, "plan-b", "code-generator")


def test_no_project_means_no_read_but_ratings_still_requested(payload, monkeypatch):
    monkeypatch.setattr(cairn_client, "_urlopen", lambda *a, **k: pytest.fail("network reached"))
    text = appendix(payload_with(payload, project=""), "plan-a", "code-generator")
    assert "Tool reputation" not in text and "Tool ids you may rate" in text


def test_brief_timeout_capped_at_two_seconds():
    assert cairn_brief._timeout({"timeout_seconds": 5.0}) == 2.0
    assert cairn_brief._timeout({"timeout_seconds": 1.0}) == 1.0
    assert cairn_brief._timeout({}) == 2.0


def test_brief_is_wall_clock_bounded(payload, monkeypatch):
    import time

    def _slow(base_url, refs, timeout, project=None):
        time.sleep(5)
        return cairn_client.CairnResult(True, 200, SCORES, "", echo="ok")

    monkeypatch.setattr(cairn_client, "score_batch", _slow)
    start = time.monotonic()
    assert reputation(payload, "plan-slow") is None
    assert time.monotonic() - start < 3.0


def test_busy_brief_lock_does_not_block(payload, fake_score_batch, tmp_path):
    import fcntl
    import threading
    import time

    lock = open(tmp_path / cairn_brief.common.REPUTATION_LOCK, "a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    try:
        box = {}
        worker = threading.Thread(target=lambda: box.setdefault("v", reputation(payload, "plan-a")), daemon=True)
        start = time.monotonic()
        worker.start()
        worker.join(3.0)
        assert not worker.is_alive(), "reputation blocked on a held lock"
        assert time.monotonic() - start < 1.0
        assert box["v"]["tool://lammps"]["composite_score"] == 0.62
        assert not (tmp_path / cairn_brief.common.REPUTATION).exists()
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
