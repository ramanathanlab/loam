"""Tests for the Tool Ratings block parser."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOKS = ROOT / ".loam" / "hooks"
if str(HOOKS) not in sys.path:
    sys.path.insert(0, str(HOOKS))

from cairn_flush import validate_event  # noqa: E402
from cairn_ratings import RatingEntry, parse_rating_block, record_ratings  # noqa: E402

REPORT = """### Hypothesis
h
### Tool Ratings
- tool://lammps | score: 0.80 | origin: none | failure_modes: none | note: 5000 steps in 212 s | energies stable
- mcp://hpc#run_code | score: 0.3 | origin: infrastructure | failure_modes: Walltime, node churn! | metrics: wall_s=1800, retries=2
### Evidence Score
+0.50
"""


def test_parses_entries():
    entries, errors = parse_rating_block(REPORT)
    assert errors == []
    assert entries[0] == RatingEntry("tool://lammps", 0.8, "none", (), {}, "5000 steps in 212 s | energies stable")
    assert entries[1].failure_modes == ("walltime", "node_churn")
    assert entries[1].metrics == {"wall_s": 1800.0, "retries": 2.0}


def test_last_heading_wins_and_prose_mention_ignored():
    text = "I will add Tool Ratings below.\n## Tool Ratings\n- tool://a | score: 0.1 | origin: none\n" + REPORT
    entries, _ = parse_rating_block(text)
    assert [e.entity for e in entries] == ["tool://lammps", "mcp://hpc#run_code"]


def test_bad_lines_rejected_individually():
    text = (
        "### Tool Ratings\n- tool://a | score: 1.7 | origin: none\n"
        "- tool://b | score: 0.5 | origin: maybe\n"
        "- tool://c | score: 0.5 | origin: agent | failure_modes: bad input\n"
    )
    entries, errors = parse_rating_block(text)
    assert [e.entity for e in entries] == ["tool://c"] and len(errors) == 2


def test_missing_score_and_empty_entity_rejected():
    text = "### Tool Ratings\n- tool://a | origin: none\n-  | score: 0.5 | origin: none\n- tool://b | score: 0.5 | origin: none\n"
    entries, errors = parse_rating_block(text)
    assert [e.entity for e in entries] == ["tool://b"] and len(errors) == 2


def test_duplicate_entity_last_wins():
    entries, _ = parse_rating_block(
        "### Tool Ratings\n- tool://a | score: 0.1 | origin: none\n- tool://a | score: 0.9 | origin: none\n"
    )
    assert [(e.entity, e.score) for e in entries] == [("tool://a", 0.9)]


def test_no_block():
    assert parse_rating_block("### Outcome\nsupported\n") == ([], [])


def test_at_most_20_entry_lines_read():
    lines = "".join(f"- tool://t{i} | score: 0.5 | origin: none\n" for i in range(25))
    entries, _ = parse_rating_block("### Tool Ratings\n" + lines)
    assert len(entries) == 20


def test_caps_on_modes_and_metrics():
    modes = ",".join(f"m{i}" for i in range(15))
    metrics = ",".join(f"k{i}={i}" for i in range(15))
    text = f"### Tool Ratings\n- t | score: 0.5 | origin: none | failure_modes: {modes}, {'x' * 80} | metrics: {metrics}, Bad Name=3, ok=nope\n"
    (entry,), errors = parse_rating_block(text)
    assert errors == [] and len(entry.failure_modes) == 10 and len(entry.metrics) == 10
    assert all(len(m) <= 48 for m in entry.failure_modes)


def test_does_not_disturb_evidence_score_parse():
    spec = importlib.util.spec_from_file_location("hypothesis_orchestration_t5", HOOKS / "hypothesis_orchestration.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["hypothesis_orchestration_t5"] = module
    spec.loader.exec_module(module)
    assert module._evidence_score(REPORT) == 0.5
    assert len(parse_rating_block(REPORT)[0]) == 2


def test_crlf_report_parses():
    entries, errors = parse_rating_block(REPORT.replace("\n", "\r\n"))
    assert errors == [] and [e.entity for e in entries] == ["tool://lammps", "mcp://hpc#run_code"]
    assert parse_rating_block(REPORT.replace("\n", "\r"))[0] == entries


def test_styled_heading_variants():
    line = "- tool://a | score: 0.5 | origin: none\n"
    for heading in ("### Tool Ratings:", "### **Tool Ratings**", "## __Tool Ratings__:", "### `Tool Ratings`", "### **Tool Ratings:**"):
        entries, errors = parse_rating_block(f"{heading}\n{line}")
        assert [e.entity for e in entries] == ["tool://a"] and errors == [], heading


def test_styled_heading_last_wins():
    text = "### **Tool Ratings**\n- tool://a | score: 0.1 | origin: none\n## Tool Ratings:\n- tool://b | score: 0.2 | origin: none\n"
    assert [e.entity for e in parse_rating_block(text)[0]] == ["tool://b"]


def test_bold_line_without_hash_is_not_a_heading():
    assert parse_rating_block("**Tool Ratings**\n- tool://a | score: 0.5 | origin: none\n") == ([], [])


def test_wrapped_entity_ids():
    text = "### Tool Ratings\n- `tool://a` | score: 0.5 | origin: none\n- **tool://b** | score: 0.5 | origin: none\n- __tool://c__ | score: 0.5 | origin: none\n"
    assert [e.entity for e in parse_rating_block(text)[0]] == ["tool://a", "tool://b", "tool://c"]


def test_capitalised_keys():
    text = "### Tool Ratings\n- tool://a | Score: 0.5 | Origin: Agent | Failure_Modes: Slow | Metrics: n=2 | Note: fine | really\n"
    (entry,), errors = parse_rating_block(text)
    assert errors == [] and entry == RatingEntry("tool://a", 0.5, "agent", ("slow",), {"n": 2.0}, "fine | really")


# ---- record_ratings -------------------------------------------------------

EXEC = {"tool_name": "mcp_hpc_run_code", "function_name": "run_code", "tool_prefix": "mcp_hpc_"}
CFG = {
    "enabled": True,
    "rate_roles": ["code-generator", "hypothesis-validator"],
    "rate_agent_reports": True,
    "rate_hook_fallback": True,
    "auto_flush": False,
    "executor": "mcp://hpc#run_code",
    "entities": {
        "mcp://hpc#run_code": {"patterns": [], "task_tags": ["hpc_execution"]},
        "tool://lammps": {"patterns": ["lammps", "lmp"], "task_tags": ["md_simulation"]},
        "tool://chai": {"patterns": ["chai"], "task_tags": ["structure"]},
    },
}
REPORT_AGENT = (
    "### Tool Ratings\n- tool://lammps | score: 0.2 | origin: agent | failure_modes: bad input\n"
)


def _append(path, row):
    with open(path, "a") as handle:
        handle.write(json.dumps(row) + "\n")


def started(tmp_path, session, role, plan, hyp):
    _append(tmp_path / "hypothesis-orchestration.jsonl", {
        "event": "agent_started", "role": role, "plan_id": plan, "hypothesis_id": hyp,
        "child_subagent_id": "id-" + session, "child_session_id": session,
    })


def completed(tmp_path, plan, hyp, status):
    _append(tmp_path / "hypothesis-orchestration.jsonl", {
        "event": "code_generator_completed", "plan_id": plan, "hypothesis_id": hyp,
        "status": status, "child_session_id": "cg",
    })


def hpc_call(tmp_path, session, code, tool="mcp_hpc_run_code", arguments=None):
    _append(tmp_path / "tool-calls.jsonl", {
        "session_id": session, "tool_name": tool,
        "arguments": arguments if arguments is not None else {"code": code},
    })


def stop(tmp_path, session, status, summary, cfg=CFG):
    return {
        "event": "subagent_stop", "child_session_id": session, "child_subagent_id": "id-" + session,
        "child_status": status, "child_summary": summary, "loam_run_id": "run", "workflow_id": "wf",
        "loam_execution": EXEC, "loam_state_dir": str(tmp_path), "loam_cairn": cfg,
    }


def rows(tmp_path, status=None):
    path = tmp_path / "cairn-ratings.jsonl"
    if not path.exists():
        return []
    out = [json.loads(line) for line in path.read_text().splitlines()]
    return [r for r in out if status is None or r["status"] == status]


def test_one_event_per_child_and_entity(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "module load lammps; lmp -in in.lj")
    record_ratings(stop(tmp_path, "s1", "completed", REPORT))
    q = rows(tmp_path, "queued")
    assert sorted(r["event"]["reviewee"]["external_id"] for r in q) == ["mcp://hpc#run_code", "tool://lammps"]
    assert all(r["event"]["weight"] == 1.0 and r["event"]["context"] == "general" for r in q)
    assert all(validate_event(r["event"]) == "" for r in q)
    assert all("task" not in r["event"] for r in q)
    lmp = next(r for r in q if r["event"]["reviewee"]["external_id"] == "tool://lammps")
    assert lmp["event"]["rationale"].startswith("[code-generator, agent-authored] ")
    assert lmp["event"]["task_tags"] == ["md_simulation"]
    assert lmp["source"] == "agent" and lmp["role"] == "code-generator"
    assert (lmp["plan_id"], lmp["hypothesis_id"], lmp["child_session_id"]) == ("p", "h", "s1")
    assert (lmp["run_id"], lmp["workflow_id"]) == ("run", "wf")
    exe = next(r for r in q if r["event"]["reviewee"]["external_id"] == "mcp://hpc#run_code")
    assert exe["event"]["failure_modes"] == ["infrastructure_walltime", "infrastructure_node_churn"]
    assert exe["event"]["metrics"] == {"wall_s": 1800.0, "retries": 2.0}
    assert "metrics" not in lmp["event"]


def test_two_children_two_events(tmp_path):
    for s in ("s1", "s2"):
        started(tmp_path, s, "code-generator", "p", "h")
        hpc_call(tmp_path, s, "lmp -in x")
        record_ratings(stop(tmp_path, s, "completed", REPORT))
    lmp = [r for r in rows(tmp_path, "queued") if r["event"]["reviewee"]["external_id"] == "tool://lammps"]
    assert sorted(r["child_session_id"] for r in lmp) == ["s1", "s2"]
    assert len({r["event_id"] for r in lmp}) == 2


def test_weights_from_role_not_agent(tmp_path):
    started(tmp_path, "cg", "code-generator", "p", "h")
    completed(tmp_path, "p", "h", "completed")
    started(tmp_path, "s2", "hypothesis-validator", "p", "h")
    record_ratings(stop(tmp_path, "s2", "completed", "### Tool Ratings\n- tool://lammps | score: 0.9 | origin: none | weight: 5\n"))
    [r] = rows(tmp_path, "queued")
    assert r["event"]["weight"] == 0.5
    assert r["event"]["rationale"].startswith("[hypothesis-validator, agent-authored] ")


def test_unbacked_when_no_matching_call(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "chai fold")
    record_ratings(stop(tmp_path, "s1", "completed", "### Tool Ratings\n- tool://lammps | score: 0.8 | origin: none\n"))
    [r] = rows(tmp_path)
    assert r["status"] == "unbacked" and r["event"] is not None


def test_other_sessions_calls_do_not_back(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "other", "lmp")
    record_ratings(stop(tmp_path, "s1", "completed", "### Tool Ratings\n- tool://lammps | score: 0.8 | origin: none\n"))
    assert [r["status"] for r in rows(tmp_path)] == ["unbacked"]


def test_truncated_arguments_are_unbacked_not_crash(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "", arguments={"truncated": True, "chars": 70000})
    record_ratings(stop(tmp_path, "s1", "completed", REPORT))
    by = {r["event"]["reviewee"]["external_id"]: r["status"] for r in rows(tmp_path)}
    assert by == {"tool://lammps": "unbacked", "mcp://hpc#run_code": "queued"}


def test_validator_unbacked_without_completed_code_generator(tmp_path):
    started(tmp_path, "s2", "hypothesis-validator", "p", "h")
    completed(tmp_path, "p", "h", "error")
    text = "### Tool Ratings\n- tool://lammps | score: 0.8 | origin: none\n"
    record_ratings(stop(tmp_path, "s2", "completed", text))
    assert [r["status"] for r in rows(tmp_path)] == ["unbacked"]
    completed(tmp_path, "p", "h", "completed")
    record_ratings(stop(tmp_path, "s2", "completed", text))
    assert [r["status"] for r in rows(tmp_path)] == ["unbacked", "queued"]


def test_agent_origin_suppressed(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "lmp")
    record_ratings(stop(tmp_path, "s1", "completed", REPORT_AGENT))
    assert rows(tmp_path, "queued") == []
    [r] = rows(tmp_path, "suppressed_agent_error")
    assert r["event"]["failure_modes"] == ["agent_bad_input"]


def test_unknown_entity_invalid(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "gmx")
    record_ratings(stop(tmp_path, "s1", "completed", "### Tool Ratings\n- tool://gromacs | score: 0.5 | origin: none\n"))
    r, _ = rows(tmp_path)
    assert r["status"] == "invalid" and r["reason"] == "unknown_entity" and r["event"] is None


def test_parser_errors_become_invalid_rows(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "lmp")
    record_ratings(stop(tmp_path, "s1", "completed", "### Tool Ratings\n- tool://lammps | score: 9 | origin: none\n"))
    r, extra = rows(tmp_path)
    assert r["status"] == "invalid" and r["event"] is None and "score" in r["reason"]
    assert extra["status"] == "missing_block"  # no valid entry survived


def test_reasoner_not_rated_by_default(tmp_path):
    started(tmp_path, "s3", "reasoner", "p", "h")
    record_ratings(stop(tmp_path, "s3", "completed", REPORT))
    assert rows(tmp_path) == []


def test_failed_code_generator_with_hpc_call_gets_executor_fallback(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "lmp")
    record_ratings(stop(tmp_path, "s1", "error", ""))
    [r] = rows(tmp_path, "queued")
    assert len(rows(tmp_path)) == 1
    assert r["source"] == "hook" and r["event"]["reviewee"]["external_id"] == "mcp://hpc#run_code"
    assert r["event"]["weight"] == 0.2 and r["event"]["score"] == 0.2
    assert r["event"]["failure_modes"] == ["lifecycle_error"]
    assert r["event"]["rationale"] == "[code-generator, loam-hook fallback] child ended with status error after 1 HPC call(s)"
    assert r["event"]["task_tags"] == ["hpc_execution"]
    assert validate_event(r["event"]) == ""


def test_missing_status_treated_as_error(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "lmp")
    payload = stop(tmp_path, "s1", "", "")
    del payload["child_status"]
    record_ratings(payload)
    [r] = rows(tmp_path, "queued")
    assert r["event"]["failure_modes"] == ["lifecycle_error"]


def test_failed_with_all_invalid_entries_still_gets_one_fallback(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "lmp")
    record_ratings(stop(tmp_path, "s1", "error", "### Tool Ratings\n- tool://lammps | score: 9 | origin: none\n"))
    assert [r["status"] for r in rows(tmp_path)] == ["invalid", "queued"]


def test_failed_without_hpc_call_records_no_tool_used(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    record_ratings(stop(tmp_path, "s1", "error", ""))
    [r] = rows(tmp_path)
    assert r["status"] == "no_tool_used" and r["event"] is None


def test_completed_without_block_records_missing_block(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "lmp")
    record_ratings(stop(tmp_path, "s1", "completed", "all done"))
    [r] = rows(tmp_path)
    assert r["status"] == "missing_block" and r["event"] is None


def test_interrupted_child_records_nothing(tmp_path):
    for status in ("interrupted", "cancelled"):
        started(tmp_path, status, "code-generator", "p", "h")
        hpc_call(tmp_path, status, "lmp")
        record_ratings(stop(tmp_path, status, status, ""))
    assert not (tmp_path / "cairn-ratings.jsonl").exists()


def test_fallback_off_records_nothing(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "lmp")
    record_ratings(stop(tmp_path, "s1", "error", "", {**CFG, "rate_hook_fallback": False}))
    assert not (tmp_path / "cairn-ratings.jsonl").exists()


def test_note_with_sequence_is_dropped(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "lmp")
    text = "### Tool Ratings\n- tool://lammps | score: 0.8 | origin: none | note: MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ\n"
    record_ratings(stop(tmp_path, "s1", "completed", text))
    [r] = rows(tmp_path)
    assert r["event"]["rationale"] == "[code-generator, agent-authored] " and r["reason"] == "note_dropped"


def test_note_sanitised(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "lmp")
    text = "### Tool Ratings\n- tool://lammps | score: 0.8 | origin: none | note: <b>fast</b>   and\t good\n"
    record_ratings(stop(tmp_path, "s1", "completed", text))
    [r] = rows(tmp_path)
    assert r["event"]["rationale"] == "[code-generator, agent-authored] fast and good"
    for bad in ("see .cairn dir", "api_key=1", "x-api-key abc", '{"a": 1}'):
        text = f"### Tool Ratings\n- tool://lammps | score: 0.8 | origin: none | note: {bad}\n"
        record_ratings(stop(tmp_path, "s1", "completed", text))
    assert [r["reason"] for r in rows(tmp_path)[1:]] == ["note_dropped"] * 4


def test_long_note_capped(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "lmp")
    text = "### Tool Ratings\n- tool://lammps | score: 0.8 | origin: none | note: " + "word " * 600 + "\n"
    record_ratings(stop(tmp_path, "s1", "completed", text))
    [r] = rows(tmp_path)
    assert len(r["event"]["rationale"]) == len("[code-generator, agent-authored] ") + 1500
    assert validate_event(r["event"]) == ""


def test_disabled_or_untagged_is_silent(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "lmp")
    record_ratings({**stop(tmp_path, "s1", "completed", REPORT), "loam_cairn": None})
    record_ratings(stop(tmp_path, "unknown-session", "completed", REPORT))
    assert not (tmp_path / "cairn-ratings.jsonl").exists()


def test_spawns_flusher_only_when_something_queued(tmp_path, monkeypatch):
    import cairn_ratings

    calls = []
    monkeypatch.setattr(cairn_ratings, "spawn_flush", lambda payload: calls.append(1))
    started(tmp_path, "s1", "code-generator", "p", "h")
    record_ratings(stop(tmp_path, "s1", "error", ""))  # no_tool_used only
    assert calls == []
    hpc_call(tmp_path, "s1", "lmp")
    record_ratings(stop(tmp_path, "s1", "error", ""))
    assert calls == [1]


GROMACS = "### Tool Ratings\n- tool://gromacs | score: 0.5 | origin: none\n"


def test_failed_with_only_unknown_entities_still_falls_back(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "gmx")
    record_ratings(stop(tmp_path, "s1", "error", GROMACS))
    assert [r["status"] for r in rows(tmp_path)] == ["invalid", "queued"]
    assert rows(tmp_path, "queued")[0]["source"] == "hook"
    assert rows(tmp_path, "queued")[0]["event"]["reviewee"]["external_id"] == "mcp://hpc#run_code"


def test_completed_with_only_unknown_entities_records_missing_block(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "gmx")
    record_ratings(stop(tmp_path, "s1", "completed", GROMACS))
    assert [r["status"] for r in rows(tmp_path)] == ["invalid", "missing_block"]


def test_non_execution_calls_do_not_back(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "", tool="mcp_hpc_inspect_filesystem", arguments={"path": "/opt/lammps"})
    record_ratings(stop(tmp_path, "s1", "completed", "### Tool Ratings\n- tool://lammps | score: 0.8 | origin: none\n"))
    assert [r["status"] for r in rows(tmp_path)] == ["unbacked"]


def test_executor_not_backed_by_non_execution_call(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "", tool="mcp_hpc_list_tools", arguments={})
    record_ratings(stop(tmp_path, "s1", "completed", "### Tool Ratings\n- mcp://hpc#run_code | score: 0.8 | origin: none\n"))
    assert [r["status"] for r in rows(tmp_path)] == ["unbacked"]


def test_failed_with_only_non_execution_call_is_no_tool_used(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "", tool="mcp_hpc_inspect_filesystem", arguments={"path": "/opt/lammps"})
    record_ratings(stop(tmp_path, "s1", "error", ""))
    [r] = rows(tmp_path)
    assert r["status"] == "no_tool_used" and r["event"] is None


def test_run_batch_counts_as_execution(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "", tool="mcp_hpc_run_batch", arguments={"jobs": ["lmp -in x"]})
    record_ratings(stop(tmp_path, "s1", "completed", "### Tool Ratings\n- tool://lammps | score: 0.8 | origin: none\n"))
    assert [r["status"] for r in rows(tmp_path)] == ["queued"]


# ---- Cairn tag contract (server: failure_modes/task_tags ^[a-z][a-z0-9_]{0,63}$, metric keys ^[a-z][a-z0-9_]{0,31}$)

import cairn_common  # noqa: E402

ADAPTER = ROOT / ".loam" / "adapters" / "hermes" / "plugin"
if str(ADAPTER) not in sys.path:
    sys.path.insert(0, str(ADAPTER))
from cairn_config import cairn_config  # noqa: E402


def test_snake_tag():
    assert cairn_common.snake_tag("Walltime") == "walltime"
    assert cairn_common.snake_tag("node churn!") == "node_churn"
    assert cairn_common.snake_tag("  md-simulation ") == "md_simulation"
    assert cairn_common.snake_tag("a::b--c") == "a_b_c"
    assert cairn_common.snake_tag("__x__") == "x"
    assert cairn_common.snake_tag("9lives") == ""
    assert cairn_common.snake_tag("!!!") == ""
    assert cairn_common.snake_tag("") == ""
    long = cairn_common.snake_tag("a" * 100)
    assert len(long) == 64 and cairn_common.TAG_RE.fullmatch(long)
    assert cairn_common.snake_tag("a" * 63 + "-b") == "a" * 63  # no trailing underscore after the cut


def test_failure_mode_origin_join_and_none_dropped():
    (entry,), _ = parse_rating_block("### Tool Ratings\n- t | score: 0.2 | origin: software | failure_modes: OOM, none, 42, --\n")
    assert entry.failure_modes == ("oom",)


def test_unspecified_failure_mode(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "lmp")
    record_ratings(stop(tmp_path, "s1", "completed", "### Tool Ratings\n- tool://lammps | score: 0.3 | origin: software\n"))
    [r] = rows(tmp_path, "queued")
    assert r["event"]["failure_modes"] == ["software_unspecified"]


def test_every_built_event_matches_cairn_tag_contract(tmp_path, monkeypatch):
    monkeypatch.delenv("LOAM_CAIRN", raising=False)  # config parsing only; auto_flush is off, no network
    cfg = cairn_config({
        "enabled": True,
        "auto_flush": False,
        "entities": {
            "tool://lammps": {"patterns": ["lmp"], "task_tags": ["MD-Simulation", "co folding"]},
            "tool://chai": {"patterns": ["chai"], "task_tags": ["structure:prediction"]},
        },
    })
    assert cfg is not None
    report = (
        "### Tool Ratings\n"
        "- tool://lammps | score: 0.3 | origin: software | failure_modes: Solver diverged, OOM-killer!! | metrics: wall_s=12, Steps=5000\n"
        "- mcp://hpc#run_code | score: 0.3 | origin: infrastructure | failure_modes: Walltime, node churn! | metrics: wall_s=1800, retries=2\n"
        "- tool://chai | score: 0.4 | origin: infrastructure\n"
    )
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "lmp -in x; chai fold")
    record_ratings(stop(tmp_path, "s1", "completed", report, cfg))
    started(tmp_path, "s2", "code-generator", "p", "h")
    hpc_call(tmp_path, "s2", "lmp")
    record_ratings(stop(tmp_path, "s2", "node-failure: exit 137", "", cfg))  # hook fallback
    events = [r["event"] for r in rows(tmp_path, "queued")]
    assert len(events) == 4 and any(r["source"] == "hook" for r in rows(tmp_path, "queued"))
    for event in events:
        for tag in event["task_tags"] + event["failure_modes"]:
            assert cairn_common.TAG_RE.fullmatch(tag), tag
        for name in event.get("metrics", {}):
            assert cairn_common.METRIC_KEY_RE.fullmatch(name), name
        assert validate_event(event) == ""
    by = {e["reviewee"]["external_id"]: e for e in events if "loam-hook" not in e["rationale"]}
    assert by["mcp://hpc#run_code"]["failure_modes"] == ["infrastructure_walltime", "infrastructure_node_churn"]
    assert by["tool://lammps"]["task_tags"] == ["md_simulation", "co_folding"]
    assert by["tool://chai"]["failure_modes"] == ["infrastructure_unspecified"]
    fallback = next(e for e in events if "loam-hook" in e["rationale"])
    assert fallback["failure_modes"] == ["lifecycle_node_failure_exit_137"]
    assert fallback["task_tags"] == ["code_exec"]


def test_metric_keys_follow_cairn_rule():
    names = ["ok", "9x", "_x", "a" * 33, "b" * 32, "with-dash", "x1_y"]
    text = "### Tool Ratings\n- t | score: 0.5 | origin: none | metrics: " + ", ".join(f"{n}=1" for n in names) + "\n"
    (entry,), _ = parse_rating_block(text)
    assert sorted(entry.metrics) == sorted(["ok", "b" * 32, "x1_y"])


def test_lowercase_sequence_note_is_dropped(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "lmp")
    text = "### Tool Ratings\n- tool://lammps | score: 0.8 | origin: none | note: folded mktayiakqrqisfvkshfsrqleerlglievq ok\n"
    record_ratings(stop(tmp_path, "s1", "completed", text))
    [r] = rows(tmp_path)
    assert r["event"]["rationale"] == "[code-generator, agent-authored] " and r["reason"] == "note_dropped"


def test_fenced_quoted_child_report_does_not_steal_ratings():
    for fence in ("```", "~~~"):
        text = (
            "### Outcome\nsupported\n"
            "### Tool Ratings\n- tool://lammps | score: 0.9 | origin: none\n"
            f"Quoting the code-generator:\n{fence}markdown\n### Tool Ratings\n"
            "- tool://lammps | score: 0.1 | origin: software | failure_modes: diverged\n"
            "- tool://chai | score: 0.2 | origin: none\n"
            f"{fence}\n"
        )
        entries, errors = parse_rating_block(text)
        assert [(e.entity, e.score) for e in entries] == [("tool://lammps", 0.9)], fence
        assert errors == []


def test_heading_after_fence_still_counts():
    text = (
        "```\n### Tool Ratings\n- tool://a | score: 0.1 | origin: none\n```\n"
        "### Tool Ratings\n- tool://b | score: 0.7 | origin: none\n"
    )
    assert [e.entity for e in parse_rating_block(text)[0]] == ["tool://b"]


def test_only_fenced_block_is_no_block():
    assert parse_rating_block("```\n### Tool Ratings\n- tool://a | score: 0.1 | origin: none\n```\n") == ([], [])


def test_unknown_entity_recorded_and_reported_across_children(tmp_path, capsys):
    from cairn_flush import main as flush_main

    for session in ("s1", "s2"):
        started(tmp_path, session, "code-generator", "p", "h")
        hpc_call(tmp_path, session, "gmx")
        record_ratings(stop(tmp_path, session, "completed", "### Tool Ratings\n- tool://gromacs | score: 0.5 | origin: none\n"))
    assert [r["entity"] for r in rows(tmp_path, "invalid")] == ["tool://gromacs"] * 2
    assert flush_main(["--state-dir", str(tmp_path), "--base-url", "https://c", "--key-file", "k", "--report"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["unknown_entities"] == {"tool://gromacs": 2}
    assert report["recorded"]["invalid"] == 2  # existing keys unchanged


def test_unknown_entity_id_capped_and_no_key_when_none(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "gmx")
    record_ratings(stop(tmp_path, "s1", "completed", "### Tool Ratings\n- tool://" + "x" * 300 + " | score: 0.5 | origin: none\n"))
    [r] = rows(tmp_path, "invalid")
    assert len(r["entity"]) == 200


def test_other_rows_have_no_entity_field(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "lmp")
    record_ratings(stop(tmp_path, "s1", "completed", "### Tool Ratings\n- tool://lammps | score: 9 | origin: none\n"))
    assert all("entity" not in r for r in rows(tmp_path))


def test_rows_carry_the_project_bound_at_launch(tmp_path):
    started(tmp_path, "s1", "code-generator", "p", "h")
    hpc_call(tmp_path, "s1", "module load lammps; lmp -in in.lj")
    record_ratings(stop(tmp_path, "s1", "completed", REPORT, cfg={**CFG, "project": "exp-silo-1"}))
    assert rows(tmp_path) and all(r["project"] == "exp-silo-1" for r in rows(tmp_path))
