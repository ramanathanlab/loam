"""Pando-style workspace for one hypothesis: layout, record files, history.

Each hypothesis-validator owns one folder,
``<loam_workspace.root>/<plan_id>/<hypothesis_id>/``, laid out the way Pando lays
out an experiment:

    experiment.json   index + state (open -> closed); written only by hooks
    log.jsonl         append-only history, each line chained to the previous one
    src/              code the code-generators ran, captured from their tool calls
    runs/rM/run.json  one record per code-generator child
    outputs/rM/       that run's result files; frozen once the run ends
    scratch/rM/       that run's temporary files; never hashed
    data/             inputs supplied up front; read-only to agents

This module is the only writer of the record files. It never decides whether a
hypothesis is validated: the completion signal stays the ``validator_completed``
row written by ``hypothesis_orchestration.record_stop``. Everything here is a
no-op unless the payload carries an enabled ``loam_workspace`` block.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

EXPERIMENT_FILE = "experiment.json"
LOG_FILE = "log.jsonl"
LOCK_FILE = ".workspace.lock"
RUN_FILE = "run.json"
# "validator" holds the hypothesis-validator's own provenance (its trace), the
# way runs/rM/ holds a code-generator's. Hook-written, so a code-generator still
# cannot write there (_layout_problem has no branch for it); it is listed so the
# stop-time audit does not count it a stray.
LAYOUT_DIRS = ("src", "runs", "outputs", "scratch", "data", "validator")
# Hook-owned top-level files. With LAYOUT_DIRS this is the whole legal top level.
RECORD_FILES = (EXPERIMENT_FILE, LOG_FILE, LOCK_FILE)
# "decision" is a declared high-level choice (loam_record_decision); "trace" is a
# captured per-agent conversation pointer. Both are reasoning provenance, chained
# into log.jsonl like everything else so they are tamper-evident.
LOG_KINDS = ("plan", "result", "note", "report", "violation", "decision", "trace")
# A run's outputs may be read by other runs only with one of these: hashed by the
# hook (local), reported by the child (remote), or hashed later by the verifier.
VERIFIED_MANIFESTS = ("recorded", "recorded_by_loam")
# The hash a first log line chains to.
GENESIS_SHA256 = "0" * 64
# The read-only skill tools a code-generator may call (hypothesis_orchestration
# .SKILL_TOOLS). Named here rather than imported because this module is loaded
# on its own by the verifier, which has no orchestration import.
SKILL_READ_TOOLS = ("skill_view", "skills_list")
# Default ceiling on one hypothesis's seeded data/ (workspace.max_input_bytes
# overrides). The seed is copied per hypothesis, so a careless inputs_dir would
# otherwise multiply a dataset by the number of hypotheses in the campaign.
DEFAULT_MAX_INPUT_BYTES = 1 << 30
# A plan or hypothesis id becomes a path segment. Both come from model-written
# goal tags, so anything that could climb out of the root ("..", "a/b") is
# refused rather than cleaned up into a different, colliding name.
_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_RUN_ID = re.compile(r"r([1-9][0-9]*)")


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def settings(payload: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The enabled ``loam_workspace`` block, or None when the feature is off."""

    value = payload.get("loam_workspace")
    if isinstance(value, Mapping) and value.get("enabled") is True and _text(value.get("root")):
        return value
    return None


def safe_segment(value: str) -> str:
    """Return ``value`` if it is a safe single path segment, else raise ValueError."""

    # _SEGMENT's first character must be alphanumeric, which already rules out "." and "..".
    if not _SEGMENT.fullmatch(value):
        raise ValueError(f"unsafe workspace id: {value!r}")
    return value


def workspace_root(payload: Mapping[str, Any], plan_id: str, hypothesis_id: str) -> Path:
    config = settings(payload)
    if config is None:
        raise ValueError("loam_workspace is not enabled")
    base = Path(_text(config.get("root"))).expanduser()
    return base / safe_segment(plan_id) / safe_segment(hypothesis_id)


# The reasoner is plan-level, not per-hypothesis, so its record sits beside the
# hypothesis folders under <plan_id>/ rather than inside one of them. Hypothesis
# ids are UUIDs, so this name never collides with one.
REASONER_DIR = "reasoner"
REASONER_FILE = "reasoner.json"


def reasoner_root(payload: Mapping[str, Any], plan_id: str) -> Path:
    config = settings(payload)
    if config is None:
        raise ValueError("loam_workspace is not enabled")
    base = Path(_text(config.get("root"))).expanduser()
    return base / safe_segment(plan_id) / REASONER_DIR


@contextlib.contextmanager
def _locked(root: Path) -> Iterator[None]:
    """Serialize writers of one workspace (a validator's children run in parallel).

    Not re-entrant: flock locks are per open file, so a nested _locked on the
    same root from this process would deadlock. Public functions take the lock
    once and call the ``_unlocked`` helpers.
    """

    root.mkdir(parents=True, exist_ok=True)
    with (root / LOCK_FILE).open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Atomically replace ``path`` so a reader never sees a torn record."""

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _log_lines(root: Path) -> list[bytes]:
    path = root / LOG_FILE
    if not path.is_file():
        return []
    return [line for line in path.read_bytes().split(b"\n") if line.strip()]


def _append_log_unlocked(root: Path, kind: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    if kind not in LOG_KINDS:
        raise ValueError(f"unknown log kind: {kind!r}")
    lines = _log_lines(root)
    entry = {
        **dict(fields),
        "seq": len(lines) + 1,
        "ts": _timestamp(),
        "kind": kind,
        "prev_sha256": hashlib.sha256(lines[-1]).hexdigest() if lines else GENESIS_SHA256,
    }
    data = (json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    # O_APPEND: the write lands at the end even if something else grew the file.
    fd = os.open(root / LOG_FILE, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
    return entry


def append_log(root: Path, kind: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    """Append one history entry; ``seq``, ``ts``, ``kind`` and ``prev_sha256`` are set here."""

    with _locked(root):
        return _append_log_unlocked(root, kind, fields)


def verify_log(root: Path) -> list[str]:
    """Problems with the log's hash chain, empty when every line is intact and in order."""

    problems: list[str] = []
    previous = GENESIS_SHA256
    for number, line in enumerate(_log_lines(root), start=1):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"line {number}: not valid JSON")
            previous = hashlib.sha256(line).hexdigest()
            continue
        if not isinstance(entry, dict) or entry.get("prev_sha256") != previous:
            problems.append(f"line {number}: hash chain broken (edited, removed or reordered line before it)")
        elif entry.get("seq") != number:
            problems.append(f"line {number}: seq {entry.get('seq')!r} out of order")
        previous = hashlib.sha256(line).hexdigest()
    return problems


def _file_digest(path: Path) -> tuple[str, int]:
    """(sha256, bytes) of one file, hashed in 1 MB chunks."""

    digest, size = hashlib.sha256(), 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def safe_relative(relative: str) -> bool:
    """Whether a seeded input's path may be written inside data/.

    A real inputs directory holds names a tidy identifier pattern rejects --
    spaces, parentheses, "+", names longer than 64 characters. Refusing those
    refused the WHOLE seed, so one oddly named file left data/ empty, and the
    HPC twin (``seed_program``) applied no such rule, so the same directory
    seeded differently on the two sides. What actually has to be excluded is
    escaping data/: an absolute path, a traversal, or an empty segment. The two
    sides now share this rule; ``seed_program`` enforces the same one remotely.
    """

    if not relative or relative.startswith("/") or "\x00" in relative:
        return False
    return all(part and part not in {os.curdir, os.pardir} for part in relative.split("/"))


def input_cap(payload: Mapping[str, Any]) -> int:
    """The ceiling on a seeded data/ for this campaign."""

    configured = settings(payload) or {}
    try:
        value = int(configured.get("max_input_bytes") or 0)
    except (TypeError, ValueError):
        value = 0
    return value if value > 0 else DEFAULT_MAX_INPUT_BYTES


def seed_inputs(root: Path, source: str, cap: int) -> tuple[list[dict[str, Any]], str]:
    """Copy a campaign inputs directory into this hypothesis's data/. Returns (entries, problem).

    Every hypothesis gets its OWN copy, so the folder stays a self-contained
    record: a later edit to the campaign's inputs_dir cannot retroactively
    change what this hypothesis was given. Each copied file is hashed, and the
    hashes are what make that claim checkable (``record_inputs``).

    Symlinks are resolved and the target copied, so data/ can never point out of
    the workspace. A file already in data/ is left alone (idempotent, and an
    operator's correction by hand is not clobbered). Oversize is refused whole
    rather than copied halfway, because a half-seeded data/ looks complete.
    """

    folder = Path(source).expanduser()
    if not folder.is_dir():
        return [], f"workspace.inputs_dir {source} is not a directory"
    found: list[tuple[Path, str]] = []
    total = 0
    for path in sorted(folder.rglob("*")):
        real = Path(os.path.realpath(path))
        if not real.is_file():
            continue
        relative = path.relative_to(folder).as_posix()
        if not safe_relative(relative):
            return [], f"input {relative} is not a safe relative path"
        found.append((real, relative))
        total += real.stat().st_size
    if total > cap:
        return [], f"workspace.inputs_dir {source} is {total} bytes, over the {cap}-byte limit (workspace.max_input_bytes)"
    data = root / "data"
    entries: list[dict[str, Any]] = []
    for real, relative in found:
        target = data / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(real, target)
        sha256, size = _file_digest(target)
        entries.append({"path": relative, "sha256": sha256, "bytes": size})
    return entries, ""


def record_inputs(root: Path, entries: list[dict[str, Any]], source: str, how: str) -> None:
    """Record a seeded data/ in experiment.json and the log, so the record names its inputs.

    ``inputs`` on experiment.json is the hypothesis's starting material, as
    distinct from a run's ``inputs``, which are other runs' outputs it consumed.
    An entry whose hash differs from one already recorded is a changed input:
    it replaces the old one and the difference is logged as a violation, since
    two hypotheses sharing a campaign inputs_dir must not quietly disagree
    about what was in it.
    """

    with _locked(root):
        path = root / EXPERIMENT_FILE
        experiment = read_json(path)
        if not experiment:
            return
        existing = {str(item.get("path")): item for item in experiment.get("inputs", []) if isinstance(item, Mapping)}
        changed, added = [], []
        for entry in entries:
            previous = existing.get(entry["path"])
            if previous is None:
                added.append(entry["path"])
            elif previous.get("sha256") != entry["sha256"]:
                changed.append(f"{entry['path']} {previous.get('sha256', '')[:12]} -> {entry['sha256'][:12]}")
            existing[entry["path"]] = {**entry, "source": source, "how": how}
        experiment["inputs"] = [existing[key] for key in sorted(existing)]
        _write_json(path, experiment)
        if added:
            _append_log_unlocked(
                root,
                "note",
                {"what": f"seeded {len(added)} input file(s) into data/ from {source} ({how})", "inputs": sorted(added)},
            )
        for difference in changed:
            _append_log_unlocked(
                root, "violation", {"what": f"campaign input changed after it was recorded: {difference}"}
            )


def seed_program(source: str, data_dir: str, cap: int) -> str:
    """The program the verifier runs on HPC to copy a staged inputs dir into data/.

    The remote twin of ``seed_inputs``: a hook has no MCP access, so it cannot
    reach the HPC data/ itself. Prints the same LOAM_MANIFEST block
    ``remote_manifest`` already parses, with paths relative to data/, so the
    local side records exactly what landed.
    """

    return "\n".join((
        "import hashlib, json, os, shutil",
        f"source, data, cap = {source!r}, {data_dir!r}, {cap!r}",
        "",
        "def unsafe(relative):",
        "    parts = relative.replace(os.sep, '/').split('/')",
        "    return relative.startswith('/') or any(p in ('', os.curdir, os.pardir) for p in parts)",
        "",
        "total = 0",
        "found = []",
        "for folder, _, files in sorted(os.walk(source)):",
        "    for name in sorted(files):",
        "        path = os.path.realpath(os.path.join(folder, name))",
        "        if not os.path.isfile(path):",
        "            continue",
        "        relative = os.path.relpath(os.path.join(folder, name), source)",
        # The same rule as the local seed (safe_relative), so one inputs
        # directory cannot seed differently on the two sides.
        "        if unsafe(relative):",
        "            raise SystemExit('input %s is not a safe relative path' % relative)",
        "        found.append((path, relative))",
        "        total += os.path.getsize(path)",
        "if total > cap:",
        "    raise SystemExit(f'inputs are {total} bytes, over the {cap}-byte limit')",
        "print('LOAM_MANIFEST')",
        "for path, relative in found:",
        "    target = os.path.join(data, relative)",
        "    os.makedirs(os.path.dirname(target), exist_ok=True)",
        "    if not os.path.exists(target):",
        "        shutil.copyfile(path, target)",
        "    digest = hashlib.sha256()",
        "    with open(target, 'rb') as handle:",
        "        for chunk in iter(lambda: handle.read(1 << 20), b''):",
        "            digest.update(chunk)",
        "    print(json.dumps({'path': relative, 'sha256': digest.hexdigest(), "
        "'bytes': os.path.getsize(target)}))",
        "print('END_LOAM_MANIFEST')",
    ))


def ensure_workspace(
    payload: Mapping[str, Any],
    plan_id: str,
    hypothesis_id: str,
    statement: str = "",
    validator_session_id: str = "",
) -> Path:
    """Create the hypothesis folder and its open experiment.json; idempotent.

    A configured ``workspace.inputs_dir`` is copied into data/ here, before any
    child runs, so the hypothesis's starting material is in place and recorded.
    A ``remote_inputs_dir`` cannot be: it lives on HPC, which a hook cannot
    reach, so the validator's start launches the verifier's ``seed`` instead.
    """

    root = workspace_root(payload, plan_id, hypothesis_id)
    with _locked(root):
        for name in LAYOUT_DIRS:
            (root / name).mkdir(parents=True, exist_ok=True)
        experiment = root / EXPERIMENT_FILE
        if experiment.is_file():
            existed = True
        else:
            existed = False
            _write_json(
                experiment,
                {
                    "schema": "loam_experiment_v1",
                    "plan_id": plan_id,
                    "hypothesis_id": hypothesis_id,
                    "workflow_id": _text(payload.get("workflow_id")),
                    "loam_run_id": _text(payload.get("loam_run_id")),
                    "statement": statement,
                    "validator_session_id": validator_session_id,
                    "state": "open",
                    "created_at": _timestamp(),
                    "closed_at": None,
                    "runs": [],
                    # What the campaign TOLD its children to read, as distinct
                    # from what a run actually read (run.json "skills"). The
                    # pair is what shows a declared recipe being ignored.
                    "declared_skills": _declared_skills(payload),
                },
            )
            _append_log_unlocked(
                root,
                "note",
                {
                    "what": "hypothesis-validator dispatched",
                    "actor": {"role": "hypothesis-validator", "session_id": validator_session_id},
                },
            )
    # Outside the lock: seed_inputs copies, record_inputs takes the lock itself,
    # and _locked is not re-entrant.
    seed_configured_inputs(payload, root, fresh=not existed)
    return root


def seed_configured_inputs(payload: Mapping[str, Any], root: Path, fresh: bool) -> None:
    """Copy workspace.inputs_dir into data/ and record it, if there is anything to do.

    Skipped once data/ holds files and experiment.json already lists them: the
    seed is idempotent but hashing is not free, and this runs on every validator
    start. ``fresh`` (the workspace was just created) always seeds, so a
    campaign that gains an inputs_dir later still seeds its next hypothesis.
    """

    source = _text((settings(payload) or {}).get("inputs_dir"))
    if not source:
        return
    recorded = bool(read_json(root / EXPERIMENT_FILE).get("inputs"))
    if not fresh and recorded and any((root / "data").rglob("*")):
        return
    entries, problem = seed_inputs(root, source, input_cap(payload))
    if problem:
        append_log(root, "violation", {"what": f"campaign inputs not seeded: {problem}"})
        return
    if entries:
        record_inputs(root, entries, source, "copied_by_hook")


def ensure_reasoner(payload: Mapping[str, Any], plan_id: str, session_id: str) -> Path:
    """Create the plan's reasoner record and its chained log; idempotent.

    ``synthesized_from`` links the record to the hypotheses this plan actually
    produced -- every sibling folder under <plan_id>/ that has an experiment.json
    -- so the reasoner's synthesis is tied to the evidence it reasoned over.
    """

    root = reasoner_root(payload, plan_id)
    plan_dir = root.parent
    synthesized_from = sorted(
        entry.name for entry in (plan_dir.iterdir() if plan_dir.is_dir() else [])
        if entry.name != REASONER_DIR and (entry / EXPERIMENT_FILE).is_file()
    )
    with _locked(root):
        record = root / REASONER_FILE
        if not record.is_file():
            _write_json(
                record,
                {
                    "schema": "loam_reasoner_v1",
                    "plan_id": plan_id,
                    "role": "reasoner",
                    "workflow_id": _text(payload.get("workflow_id")),
                    "loam_run_id": _text(payload.get("loam_run_id")),
                    "session_id": session_id,
                    "state": "open",
                    "created_at": _timestamp(),
                    "closed_at": None,
                    "synthesized_from": synthesized_from,
                    "decisions": [],
                },
            )
            _append_log_unlocked(
                root, "note",
                {"what": f"reasoner dispatched; synthesizing {len(synthesized_from)} hypotheses",
                 "synthesized_from": synthesized_from,
                 "actor": {"role": "reasoner", "session_id": session_id}},
            )
    return root


def close_reasoner(root: Path, summary: str) -> None:
    """Close the reasoner record at its stop and log its synthesis report. Idempotent."""

    with _locked(root):
        path = root / REASONER_FILE
        record = read_json(path)
        if record.get("state") != "open":
            return
        record.update(state="closed", closed_at=_timestamp())
        _write_json(path, record)
        _append_log_unlocked(
            root, "report",
            {"what": "reasoner finished", "results": summary[:MAX_RESULT_CHARS],
             "actor": {"role": "reasoner", "session_id": _text(record.get("session_id"))}},
        )


def _next_run_id(root: Path) -> str:
    numbers = [
        int(match.group(1))
        for path in (root / "runs").iterdir()
        if path.is_dir() and (match := _RUN_ID.fullmatch(path.name))
    ] if (root / "runs").is_dir() else []
    return f"r{max(numbers, default=0) + 1}"


def _reserve_run_unlocked(root: Path, goal: str = "") -> str:
    experiment = read_json(root / EXPERIMENT_FILE)
    if experiment.get("state") != "open":
        raise ValueError(f"workspace {root} is not open")
    run_id = _next_run_id(root)
    for name in ("outputs", "scratch"):
        # exist_ok=False: if two writers ever picked the same id, fail loudly
        # rather than let two children share one output folder.
        (root / name / run_id).mkdir(parents=True, exist_ok=False)
    _write_json(
        root / "runs" / run_id / RUN_FILE,
        {
            "schema": "loam_run_v1",
            "run": run_id,
            "child_session_id": "",
            # The goal the run was reserved FOR, which is how a child that
            # arrived without its [loam-run-id] tag is matched back to the run
            # its instructions describe (``_match_reserved_unlocked``).
            "goal": goal,
            "state": "reserved",
            "reserved_at": _timestamp(),
            "started_at": None,
            "ended_at": None,
            "calls": [],
            "manifest": [],
        },
    )
    experiment["runs"] = [*experiment.get("runs", []), run_id]
    _write_json(root / EXPERIMENT_FILE, experiment)
    return run_id


def _start_run_unlocked(root: Path, run_id: str, child_session_id: str, goal: str, why: str) -> str:
    if not _RUN_ID.fullmatch(run_id):
        raise ValueError(f"not a run id: {run_id!r}")
    path = root / "runs" / run_id / RUN_FILE
    run = read_json(path)
    if run.get("state") != "reserved":
        raise ValueError(f"run {run_id} is {run.get('state') or 'missing'}, not reserved")
    run.update(child_session_id=child_session_id, goal=goal, state="started", started_at=_timestamp())
    _write_json(path, run)
    _append_log_unlocked(
        root,
        "plan",
        {
            "run": run_id,
            "what": goal,
            "why": why,
            "actor": {"role": "code-generator", "session_id": child_session_id},
        },
    )
    return run_id


def _reserved_runs_unlocked(root: Path) -> list[tuple[str, str]]:
    """(run_id, reserved goal) for every run still ``reserved``, lowest id first."""

    runs = root / "runs"
    found: list[tuple[int, str, str]] = []
    for path in runs.iterdir() if runs.is_dir() else []:
        match = _RUN_ID.fullmatch(path.name) if path.is_dir() else None
        if match is None:
            continue
        run = read_json(path / RUN_FILE)
        if run.get("state") == "reserved":
            found.append((int(match.group(1)), path.name, _text(run.get("goal"))))
    return [(run_id, goal) for _, run_id, goal in sorted(found)]


def _match_reserved_unlocked(root: Path, goal: str) -> str:
    """The reserved run a child with this goal belongs to, or "" if none is left.

    Prefers the run reserved for that exact goal, so a batch of children that
    all lost their tags still lands in the folders their instructions name.
    Falls back to the lowest reserved run, which is still better than a brand
    new one: a reserved run is at least a folder some child was told about.
    """

    reserved = _reserved_runs_unlocked(root)
    if goal:
        for run_id, reserved_goal in reserved:
            if reserved_goal == goal:
                return run_id
    return reserved[0][0] if reserved else ""


def reserve_runs(root: Path, count: int, goals: tuple[str, ...] | list[str] = ()) -> list[str]:
    """Reserve run ids for children about to be delegated.

    A child's instructions are written when its validator calls delegate_task,
    before the child exists, yet they must name its outputs/rM folder. So the
    ids are reserved here and travel in a Loam-forced [loam-run-id=rM] goal tag;
    ``adopt_run`` moves each one to "started" once the child really begins. A
    run that never starts stays "reserved" and is closed as dropped later.

    ``goals`` records what each run was reserved for. Reserving is idempotent in
    those goals: a run already reserved for a goal is handed back rather than
    duplicated, because the reservation happens inside request middleware that
    may run more than once for one dispatch (hypothesis_orchestration.
    _reserve_child_runs).
    """

    with _locked(root):
        if not goals:
            return [_reserve_run_unlocked(root) for _ in range(count)]
        free = dict(_reserved_runs_unlocked(root))
        taken: set[str] = set()
        run_ids: list[str] = []
        for goal in list(goals)[:count]:
            reuse = next((run for run, g in free.items() if g == goal and run not in taken), "")
            if reuse:
                taken.add(reuse)
                run_ids.append(reuse)
            else:
                run_ids.append(_reserve_run_unlocked(root, goal))
        return run_ids


def start_run(root: Path, run_id: str, child_session_id: str, goal: str, why: str = "") -> str:
    """Mark a reserved run as started and log its plan."""

    with _locked(root):
        return _start_run_unlocked(root, run_id, child_session_id, goal, why)


def allocate_run(root: Path, child_session_id: str, goal: str, why: str = "") -> str:
    """Reserve and start the next run in one step, for a child that arrived without a reserved id."""

    with _locked(root):
        run_id = _reserve_run_unlocked(root, goal)
        return _start_run_unlocked(root, run_id, child_session_id, goal, why)


def adopt_run(root: Path, run_id: str, child_session_id: str, goal: str, why: str = "") -> str:
    """Give a starting child a run, whatever state its reserved id is in.

    The id travels in a goal tag through a model and a dispatch rewrite, and the
    child may be dispatched more than once, so three things can arrive:

    * the reserved id, still reserved -- the normal case; start it.
    * the reserved id, already started BY THIS SESSION -- the same start
      replayed (Hermes re-emitted the callback); hand back the same run instead
      of failing, which used to leave the child with no run at all and so with
      every write and every skill read blocked.
    * an id that is taken or missing -- a re-dispatch of the same task, or a
      tag that did not survive. Adopt the run reserved for this goal, so the
      child lands in the folder its own instructions name; only when nothing is
      reserved is a new run created. Either way the substitution is logged, so
      the record shows which run the child was told about and which it got.
    """

    with _locked(root):
        if run_id:
            if not _RUN_ID.fullmatch(run_id):
                raise ValueError(f"not a run id: {run_id!r}")
            run = read_json(root / "runs" / run_id / RUN_FILE)
            state = _text(run.get("state"))
            if state == "reserved":
                return _start_run_unlocked(root, run_id, child_session_id, goal, why)
            if state == "started" and _text(run.get("child_session_id")) == child_session_id:
                return run_id
            instead = f"run {run_id} is {state or 'missing'}, not reserved"
        else:
            instead = "the child's goal carried no [loam-run-id] tag"
        adopted = _match_reserved_unlocked(root, goal)
        what = f"adopted reserved run {adopted}" if adopted else "no run was reserved, so one was created"
        if not adopted:
            adopted = _reserve_run_unlocked(root, goal)
        _append_log_unlocked(
            root,
            "note",
            {
                "run": adopted,
                "what": f"{what}: {instead}",
                "actor": {"role": "code-generator", "session_id": child_session_id},
            },
        )
        return _start_run_unlocked(root, adopted, child_session_id, goal, why)


def on_agent_started(
    payload: Mapping[str, Any],
    role: str,
    plan_id: str,
    hypothesis_id: str,
    run_id: str = "",
    goal: str = "",
    statement: str = "",
) -> dict[str, str]:
    """Workspace side of ``record_start``: the fields to add to the agent_started row.

    Never raises. The agent_started row is what tool_policy uses to identify a
    session, so a workspace failure must not lose it; instead the row carries
    ``workspace_error`` and the failure is reported on stderr.
    """

    if settings(payload) is None:
        return {}
    # A code-generator needs its hypothesis; a reasoner is plan-level and has none.
    if role not in {"hypothesis-validator", "code-generator", "reasoner"}:
        return {}
    if role in {"hypothesis-validator", "code-generator"} and not hypothesis_id:
        return {}
    session_id = _text(payload.get("child_session_id"))
    try:
        if role == "reasoner":
            return {"workspace": str(ensure_reasoner(payload, plan_id, session_id))}
        if role == "hypothesis-validator":
            root = ensure_workspace(payload, plan_id, hypothesis_id, statement, session_id)
            _launch_seed(payload, plan_id, hypothesis_id, root)
            return {"workspace": str(root)}
        root = workspace_root(payload, plan_id, hypothesis_id)
        why = f"delegated by the hypothesis-validator for {hypothesis_id}"
        run = adopt_run(root, run_id, session_id, goal, why)
        return {"workspace": str(root), "run": run}
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        print(f"loam workspace: {role} start failed: {message}", file=sys.stderr)
        return {"workspace_error": message}


# --- enforcement (pre_tool_call) ------------------------------------------------

# Hermes file tools, by what they do to their `path` argument.
_READ_TOOLS = {"read_file", "search_files"}
# HPC MCP tools that only report state; always allowed.
_HPC_STATUS = {"list_tools", "list_batches", "batch_status"}
_HPC_TOOLS = ("run_code", "run_batch", "register_tool", "inspect_filesystem", *sorted(_HPC_STATUS))
_PATCH_FILE = re.compile(r"^\*\*\* (?:Update|Add|Delete) File:\s*(.+?)\s*$", re.MULTILINE)
_PATCH_MOVE = re.compile(r"^\*\*\* Move File:\s*(.+?)\s*->\s*(.+?)\s*$", re.MULTILINE)
# Shell commands whose path arguments are all written (or removed), and those
# whose LAST path argument is the destination.
_SHELL_WRITES_ALL = {"tee", "touch", "mkdir", "rm", "rmdir", "truncate", "chmod", "chown"}
_SHELL_WRITES_LAST = {"cp", "mv", "ln", "rsync", "install"}
_SHELL_REDIRECTS = {">", ">>", "&>", "2>", ">|"}
_SHELL_SEPARATORS = {";", "&&", "||", "|", "&"}


def _hpc_function(tool_name: str) -> str:
    """The HPC MCP function a Hermes tool name refers to (``mcp_hpc_run_code`` -> ``run_code``)."""

    for name in _HPC_TOOLS:
        if tool_name == name or tool_name.endswith("_" + name):
            return name
    return ""


def _within(path: str, root: str) -> list[str] | None:
    """Path components of ``path`` below ``root``, or None when it is not inside it."""

    try:
        relative = os.path.relpath(path, root)
    except ValueError:
        return None
    if relative == ".":
        return []
    if relative == os.pardir or relative.startswith(os.pardir + os.sep):
        return None
    return relative.split(os.sep)


def _layout_problem(parts: list[str], run_id: str) -> str | None:
    """Why writing at ``parts`` (relative to the workspace root) breaks the layout, or None."""

    own = f"src/, outputs/{run_id}/ or scratch/{run_id}/"
    if not parts:
        return f"write into {own}, not the workspace top level"
    head = parts[0]
    if head in RECORD_FILES or head == "runs":
        return f"{head} is a Loam record file; only Loam hooks write it"
    if head == "data":
        return f"data/ holds read-only inputs; write derived files to outputs/{run_id}/"
    if head == "src":
        return None
    if head in {"outputs", "scratch"}:
        if run_id and len(parts) > 1 and parts[1] == run_id:
            return None
        other = parts[1] if len(parts) > 1 else ""
        return f"{head}/{other} is not your run; you may write only {own} (other runs are frozen)"
    return f"{head} is not part of the workspace layout; write into {own}"


def batch_root_for(execution: Mapping[str, Any] | None) -> str:
    """Where the HPC server keeps batches: execution.batch_root, else <workdir_root>/../batches.

    The server's HPC_BATCH_ROOT defaults to <HPC_BASE_DIR>/batches, and
    workdir_root is normally <HPC_BASE_DIR>/work, hence the default.
    """

    if not isinstance(execution, Mapping):
        return ""
    configured, workdir_root = _text(execution.get("batch_root")), _text(execution.get("workdir_root"))
    if configured:
        return os.path.normpath(configured)
    return os.path.join(os.path.dirname(os.path.normpath(workdir_root)), "batches") if workdir_root else ""


def batch_prefix(hypothesis_id: str, run_id: str) -> str:
    """The prefix a run's batch ids must start with.

    Batch ids are global on the HPC server, so this names the owner. Long
    hypothesis ids are shortened with a digest to stay within the server's
    64-character batch id limit.
    """

    short = hypothesis_id if len(hypothesis_id) <= 24 else f"{hypothesis_id[:15]}-{hashlib.sha256(hypothesis_id.encode()).hexdigest()[:8]}"
    return f"{short}-{run_id}-"


def batch_labels(plan_id: str, hypothesis_id: str, run_id: str) -> dict[str, str]:
    """The labels a run's batches must carry; the server refuses to resume a batch under other labels."""

    return {"plan_id": plan_id, "hypothesis_id": hypothesis_id, "run": run_id}


class _Scope:
    """The places one code-generator may touch, locally and on HPC."""

    def __init__(self, payload: Mapping[str, Any], start: Mapping[str, Any]):
        config = settings(payload) or {}
        self.root = Path(_text(start.get("workspace")))
        self.real_root = os.path.realpath(self.root)
        self.base = Path(_text(config.get("root"))).expanduser()
        self.real_base = os.path.realpath(self.base)
        self.run_id = _text(start.get("run"))
        self.plan_id, self.hypothesis_id = _text(start.get("plan_id")), _text(start.get("hypothesis_id"))
        # Completed outputs of OTHER runs this call reads (this hypothesis's or
        # another's); capture_code records them as the run's inputs.
        self.reads: list[dict[str, str]] = []
        # Read-only campaign reference data (workspace.shared_inputs) this call
        # reads in place rather than from data/; also recorded as inputs.
        self.shared_reads: list[dict[str, str]] = []
        execution = payload.get("loam_execution")
        execution = execution if isinstance(execution, Mapping) else {}
        aliases = execution.get("workdir_aliases")
        # Longest alias first, so a more specific spelling wins.
        self.aliases = sorted(
            ((_text(a).rstrip("/"), _text(c).rstrip("/")) for a, c in (aliases.items() if isinstance(aliases, Mapping) else ())),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )
        workdir_root = _text(execution.get("workdir_root"))
        self.remote_base = self._canonical(os.path.normpath(workdir_root)) if workdir_root else ""
        self.remote_root = os.path.join(self.remote_base, self.plan_id, self.hypothesis_id) if self.remote_base else ""
        batch_root = batch_root_for(execution)
        self.batch_root = self._canonical(batch_root) if batch_root else ""
        # Canonicalized here, after the aliases exist, so an HPC spelling of a
        # shared path matches whichever form a child writes.
        shared = config.get("shared_inputs")
        configured_shared = tuple(
            _text(item).rstrip("/") for item in (shared if isinstance(shared, (list, tuple)) else ())
            if _text(item).startswith("/")
        )
        # Two spellings, because the two branches compare differently: a local
        # path is matched after os.path.realpath (so a symlinked /var does not
        # look foreign), an HPC path after alias canonicalization (the hook
        # cannot resolve a path on another machine).
        self.shared = tuple(self._canonical(item) for item in configured_shared)
        self.shared_real = tuple(os.path.realpath(item) for item in configured_shared)
        # This run's own batches, as recorded by capture_code.
        run = read_json(self.root / "runs" / self.run_id / RUN_FILE) if self.run_id else {}
        self.batches = {b for b in run.get("batches", []) if isinstance(b, str)}

    def _canonical(self, path: str) -> str:
        """Rewrite an aliased HPC spelling (/lus/flare/projects/X) to its canonical one (/flare/X)."""

        for alias, canonical in self.aliases:
            if path == alias or path.startswith(alias + "/"):
                return canonical + path[len(alias):]
        return path

    # local paths -------------------------------------------------------------
    def local_write(self, raw: str) -> str | None:
        if not raw:
            return None
        path = Path(raw).expanduser()
        if not path.is_absolute():
            return f"use an absolute path under {self.root} (got relative path {raw!r})"
        return self._local(os.path.realpath(path), raw, writing=True)

    def local_read(self, raw: str) -> str | None:
        path = Path(raw).expanduser() if raw else None
        if path is None or not path.is_absolute():
            return None
        return self._local(os.path.realpath(path), raw, writing=False)

    def _local(self, real: str, raw: str, writing: bool) -> str | None:
        parts = _within(real, self.real_root)
        if parts is None:
            foreign = _within(real, self.real_base)
            if foreign is not None:
                return self._other_run(foreign, raw, writing)
            problem = self._shared_path(real, raw, writing, self.shared_real)
            if problem is not None:
                return problem or None
            return f"{raw} is outside this hypothesis's workspace {self.root}" if writing else None
        return self._own(parts, raw, writing)

    def _shared_path(self, path: str, raw: str, writing: bool, roots: tuple[str, ...] | None = None) -> str | None:
        """Verdict for a path under a configured shared_inputs root, or None if it is not one.

        Returns "" for an allowed read (distinct from None, "not a shared
        path"), so a caller can tell the two apart. Shared reference data is
        read-only to everyone: a write there is refused even in the one
        direction the layout would otherwise ignore, because it is the single
        piece of state every hypothesis in the campaign sees.
        """

        for index, base in enumerate(self.shared if roots is None else roots):
            if _within(path, base) is None:
                continue
            if writing:
                return f"{raw} is shared campaign input, which is read-only; write into outputs/{self.run_id}/"
            # Record the configured spelling, not the resolved one, so the
            # record names the path the campaign declared.
            self.shared_reads.append({"shared": self.shared[index], "path": raw if os.path.isabs(raw) else path})
            return ""
        return None

    def _own(self, parts: list[str], raw: str, writing: bool) -> str | None:
        """A path inside this hypothesis's folder; ``parts`` are relative to its root.

        Writes follow the layout. Reads are free, except another run's
        outputs/ or scratch/: those get the same rule as another hypothesis's.
        """

        if writing:
            return _layout_problem(parts, self.run_id)
        if len(parts) > 1 and parts[0] in {"outputs", "scratch"} and parts[1] != self.run_id:
            return self._other_run([self.plan_id, self.hypothesis_id, *parts], raw, writing=False)
        return None

    def _other_run(self, parts: list[str], raw: str, writing: bool) -> str | None:
        """A path belonging to ANOTHER run; ``parts`` are relative to the workspaces root.

        That run may be in another hypothesis (never writable) or an earlier run
        of this one (writes already refused by the layout). Either way only its
        finished results are readable, Pando-style (a run consuming
        ``output:eN/rM``): ``outputs/rK/`` of a run that is ``completed`` and
        whose output manifest is verified. Never scratch/, a run still in
        progress, a failed or dropped one, or one whose hashes are unknown. An
        allowed read is collected in ``self.reads`` so it is recorded as an input.
        """

        own = parts[:2] == [self.plan_id, self.hypothesis_id]
        if writing:
            return f"{raw} belongs to another hypothesis's workspace, which is read-only to you"
        legal = f"only outputs/rK/ of {'this' if own else 'another'} hypothesis's completed, verified runs may be read"
        if len(parts) < 4 or parts[2] != "outputs" or not _RUN_ID.fullmatch(parts[3]):
            return f"{raw}: {legal}"
        plan_id, hypothesis_id, run_id = parts[0], parts[1], parts[3]
        run = read_json(self.base / plan_id / hypothesis_id / "runs" / run_id / RUN_FILE)
        if run.get("state") != "completed":
            return f"{raw}: run {run_id} of {hypothesis_id} is {run.get('state') or 'unknown'}, not completed; {legal}"
        if run.get("manifest_status") not in VERIFIED_MANIFESTS:
            return (
                f"{raw}: run {run_id} of {hypothesis_id} has no verified output manifest yet "
                f"({run.get('manifest_status') or 'none'}); {legal}"
            )
        self.reads.append(
            {"plan_id": plan_id, "hypothesis_id": hypothesis_id, "run": run_id, "path": "/".join(parts[2:])}
        )
        return None

    # remote (HPC) paths --------------------------------------------------------
    def remote(self, raw: str, writing: bool, required: bool = False, label: str = "path") -> str | None:
        if not self.remote_root:
            return None
        if not raw:
            if required:
                return f"pass {label}={self.remote_root}/scratch/{self.run_id} (your run's HPC folder)"
            return None
        path = self._canonical(os.path.normpath(raw if os.path.isabs(raw) else os.path.join(self.remote_root, raw)))
        in_batches = _within(path, self.batch_root) if self.batch_root else None
        if in_batches is not None:
            return self._batch_path(in_batches, raw, writing)
        parts = _within(path, self.remote_root)
        if parts is None:
            foreign = _within(path, self.remote_base)
            if foreign is not None:
                # The HPC folder mirrors the local layout, so the local run.json
                # says whether that remote run completed.
                return self._other_run(foreign, raw, writing)
            problem = self._shared_path(path, raw, writing)
            if problem is not None:
                return problem or None
            return f"{label} {raw} is outside this hypothesis's HPC workspace {self.remote_root}" if required else None
        return self._own(parts, raw, writing)

    def _batch_path(self, parts: list[str], raw: str, writing: bool) -> str | None:
        """A path in the server's shared batch folder; ``parts`` are relative to it.

        Batches of every run of every hypothesis share that folder. A child may
        read only its own run's batches (and their registry entries); another
        run's batch results are read from that run's outputs/rK/, where they
        were harvested, checked and hashed.
        """

        if writing:
            return f"{raw} is in the shared batch folder; write into outputs/{self.run_id}/ or scratch/{self.run_id}/"
        batch_id = parts[0] if parts else ""
        if batch_id == "registry":
            batch_id = parts[1][: -len(".json")] if len(parts) > 1 and parts[1].endswith(".json") else ""
        if batch_id and batch_id in self.batches:
            return None
        return (
            f"{raw}: batch {batch_id or '(all)'} is not one of your run's batches; "
            "read another run's batch results from its outputs/rK/ instead"
        )

    def _spellings(self, path: str) -> set[str]:
        """``path`` and its aliased spellings."""

        spellings = {path}
        for alias, canonical in self.aliases:
            if path == canonical or path.startswith(canonical + "/"):
                spellings.add(alias + path[len(canonical):])
        return spellings

    def _code_bases(self) -> set[str]:
        """Every spelling of the workspaces roots that code might use."""

        bases = {self.real_base, str(self.base)} | (self._spellings(self.remote_base) if self.remote_base else set())
        return bases - {""}

    def code(self, text: str) -> str | None:
        """Check paths into other runs' folders named in code, locally or on HPC.

        Code can't be analysed for what it does with a path, so each one is held
        to the read rule: allowed only for a completed, verified run's outputs.
        That covers another hypothesis's runs and this hypothesis's earlier ones.
        """

        for base in self._code_bases():
            pattern = re.escape(base.rstrip("/")) + r"/([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+)((?:/[^\s'\"(),;\]]*)?)"
            for match in re.finditer(pattern, text):
                parts = [match.group(1), match.group(2), *(p for p in match.group(3).split("/") if p)]
                if parts[:2] == [self.plan_id, self.hypothesis_id]:
                    problem = self._own(parts[2:], match.group(0), writing=False)
                    where = "another run of this hypothesis"
                else:
                    problem = self._other_run(parts, match.group(0), writing=False)
                    where = "another hypothesis's workspace"
                if problem:
                    return f"code refers to {where}: {problem}"
        # Shared reference data named in a program is the normal way a remote
        # child reads it (there is no path argument to inspect), so it is
        # collected here for provenance. Reads there are always allowed, so
        # nothing is returned; a write cannot be told from a read in code.
        for base in self.shared:
            for spelling in self._spellings(base):
                pattern = re.escape(spelling.rstrip("/")) + r"(?:/[^\s'\"(),;\]]*)?"
                for match in re.finditer(pattern, text):
                    self._shared_path(self._canonical(match.group(0)), match.group(0), writing=False)
        for base in self._spellings(self.batch_root) if self.batch_root else ():
            pattern = re.escape(base.rstrip("/")) + r"/([A-Za-z0-9._-]+)((?:/[^\s'\"(),;\]]*)?)"
            for match in re.finditer(pattern, text):
                parts = [match.group(1), *(p for p in match.group(2).split("/") if p)]
                problem = self._batch_path(parts, match.group(0), writing=False)
                if problem:
                    return f"code refers to a batch: {problem}"
        return None

    # shell ---------------------------------------------------------------------
    def shell(self, command: str, workdir: str) -> str | None:
        if not workdir:
            return f"pass workdir={self.root}/scratch/{self.run_id} so the command runs inside your workspace"
        problem = self.local_write(workdir)
        if problem:
            return f"workdir: {problem}"
        for token in _shell_write_targets(command):
            path = Path(token).expanduser()
            path = path if path.is_absolute() else Path(workdir) / path
            problem = self._local(os.path.realpath(path), token, writing=True)
            if problem:
                return problem
        for token in _shell_tokens(command):
            if "/" in token:
                problem = self.local_read(token if os.path.isabs(token) else str(Path(workdir) / token))
                if problem:
                    return problem
        return None


def _shell_tokens(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        return list(lexer)
    except ValueError:  # unbalanced quotes: fall back to whitespace
        return command.split()


def _shell_write_targets(command: str) -> list[str]:
    """Paths a shell command line writes: redirect targets and the write commands' arguments.

    Conservative by design. It cannot see what a script writes (`python x.py`);
    the stop-time audit catches that. It only needs to stop the obvious shell
    edits of record files, frozen runs and other hypotheses.
    """

    targets: list[str] = []
    segment: list[str] = []
    tokens = iter(_shell_tokens(command))
    for token in tokens:
        if token in _SHELL_REDIRECTS:
            target = next(tokens, "")  # the token after a redirect is what it writes
            if target:
                targets.append(target)
        elif token in _SHELL_SEPARATORS:
            targets.extend(_segment_targets(segment))
            segment = []
        else:
            segment.append(token)
    targets.extend(_segment_targets(segment))
    return targets


def _segment_targets(segment: list[str]) -> list[str]:
    """Paths one simple command (no separators, no redirects) writes."""

    if not segment:
        return []
    name, rest = os.path.basename(segment[0]), segment[1:]
    if name in _SHELL_WRITES_ALL:
        return [token for token in rest if not token.startswith("-")]
    if name in _SHELL_WRITES_LAST:
        return [token for token in rest if not token.startswith("-")][-1:]
    if name == "sed" and any(token.startswith("-i") or token == "--in-place" for token in rest):
        # The edit script is the first operand unless given with -e/-f.
        files, script_given, skip = [], False, False
        for token in rest:
            if skip:
                skip = False
            elif token in {"-e", "-f", "--expression", "--file"}:
                script_given, skip = True, True
            elif not token.startswith("-"):
                files.append(token)
        return files if script_given else files[1:]
    return []


def _arguments(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    value = payload.get("arguments")
    return value if isinstance(value, Mapping) else {}


def _violation(payload: Mapping[str, Any], scope: _Scope) -> str | None:
    tool = _text(payload.get("tool_name"))
    args = _arguments(payload)
    if tool == "write_file":
        return scope.local_write(_text(args.get("path")))
    if tool == "patch":
        text = _text(args.get("patch"))
        paths = [_text(args.get("path"))] if _text(args.get("mode")) != "patch" else [
            *_PATCH_FILE.findall(text), *(p for pair in _PATCH_MOVE.findall(text) for p in pair)
        ]
        return next((problem for raw in paths if (problem := scope.local_write(raw))), None)
    if tool in _READ_TOOLS:
        return scope.local_read(_text(args.get("path")))
    if tool == "terminal":
        return scope.shell(_text(args.get("command")), _text(args.get("workdir")))
    if tool == "execute_code":
        return scope.code(_text(args.get("code")))
    function = _hpc_function(tool)
    if function == "run_code":
        return scope.remote(_text(args.get("workdir")), writing=True, required=True, label="workdir") or scope.code(
            _text(args.get("payload"))
        )
    if function == "run_batch":
        problem = _batch_identity(args, scope)
        if problem:
            return problem
    if function == "batch_status":
        batch_id = _text(args.get("batch_id"))
        if batch_id not in scope.batches:
            return f"batch {batch_id or '(none)'} is not one of run {scope.run_id}'s batches"
        return None
    if function == "list_batches":
        if _text(args.get("hypothesis_id")) != scope.hypothesis_id or _text(args.get("plan_id")) not in {"", scope.plan_id}:
            return f"pass hypothesis_id={scope.hypothesis_id!r} (and plan_id={scope.plan_id!r}) to list only this hypothesis's batches"
        return None
    if function in {"run_batch", "register_tool"}:
        for key in ("script_path", "inputs_path"):
            if _text(args.get(key)):
                problem = scope.remote(_text(args.get(key)), writing=False, required=True, label=key)
                if problem:
                    return problem
        return None
    if function == "inspect_filesystem":
        return scope.remote(_text(args.get("path")), writing=False)
    return None


def _batch_identity(args: Mapping[str, Any], scope: _Scope) -> str | None:
    """A run_batch call must name its owner: a prefixed batch_id and matching labels.

    The batch id is how Loam records the batch in run.json (the hook sees the
    call, not its result), and the labels make the server itself refuse to
    resume the batch under another hypothesis.
    """

    prefix = batch_prefix(scope.hypothesis_id, scope.run_id)
    labels = batch_labels(scope.plan_id, scope.hypothesis_id, scope.run_id)
    batch_id = _text(args.get("batch_id"))
    if not batch_id.startswith(prefix) or len(batch_id) == len(prefix):
        return f"pass batch_id='{prefix}<name>' so the batch is recorded as part of run {scope.run_id}"
    given = args.get("labels")
    given = given if isinstance(given, Mapping) else {}
    if any(_text(given.get(key)) != value for key, value in labels.items()):
        return f"pass labels={json.dumps(labels)} (other keys may be added)"
    return None


def fs_policy(payload: Mapping[str, Any], start: Mapping[str, Any] | None) -> dict[str, str] | None:
    """Block a code-generator tool call that writes outside its run's place in the layout.

    ``start`` is the session's agent_started row. Only code-generators are
    governed: validators may already call nothing but delegate_task, and
    non-campaign sessions have no row. With ``strict: false`` a violation is
    logged but allowed.
    """

    config = settings(payload)
    if config is None or not isinstance(start, Mapping) or _text(start.get("role")) != "code-generator":
        return None
    tool = _text(payload.get("tool_name"))
    # A skill read is read-only, and it is the one call a child in trouble most
    # needs: a frozen run or a failed workspace must not be what stops it from
    # re-reading the campaign's recipe.
    reading_only = tool in _READ_TOOLS or tool in SKILL_READ_TOOLS or _hpc_function(tool) in _HPC_STATUS
    if not _text(start.get("workspace")) or not _text(start.get("run")):
        if reading_only:
            return None
        reason = f"this child has no workspace run ({_text(start.get('workspace_error')) or 'not allocated'})"
        return _verdict(config, None, start, tool, reason)
    scope = _Scope(payload, start)
    run = read_json(scope.root / "runs" / scope.run_id / RUN_FILE)
    if run.get("state") != "started" and not reading_only:
        reason = f"run {scope.run_id} is {run.get('state') or 'missing'}; a finished run is frozen"
        return _verdict(config, scope.root, start, tool, reason)
    reason = _violation(payload, scope)
    return _verdict(config, scope.root, start, tool, reason) if reason else None


def _verdict(
    config: Mapping[str, Any], root: Path | None, start: Mapping[str, Any], tool: str, reason: str
) -> dict[str, str] | None:
    """Log the violation in the hypothesis's history, then block unless strict is off."""

    if root is not None:
        try:
            append_log(
                root,
                "violation",
                {
                    "run": _text(start.get("run")),
                    "what": f"{tool}: {reason}",
                    "actor": {"role": "code-generator", "session_id": _text(start.get("child_session_id"))},
                    "blocked": config.get("strict") is not False,
                },
            )
        except Exception as exc:
            print(f"loam workspace: could not log violation: {type(exc).__name__}: {exc}", file=sys.stderr)
    if config.get("strict") is False:
        return None
    return {"action": "block", "message": f"Loam workspace: {reason}"}


# --- provenance (pre_tool_call, after the policy allowed the call) -------------


def capture_code(payload: Mapping[str, Any], start: Mapping[str, Any] | None) -> None:
    """Record what an allowed code-generator call runs and reads, in the run's record.

    Code passed inline (HPC run_code ``payload``, local execute_code ``code``) is
    saved as src/rM_NN.py, so the exact program behind a result is kept even
    when it ran remotely. Calls that point at files (write_file, run_batch,
    register_tool) are recorded by path, with a hash when the content is in the
    call. Completed outputs of other hypotheses that the call reads become the
    run's ``inputs``, with the source file's hash when its manifest has one.
    Best effort; the caller isolates failures.
    """

    if settings(payload) is None or not isinstance(start, Mapping) or _text(start.get("role")) != "code-generator":
        return
    root, run_id = Path(_text(start.get("workspace"))), _text(start.get("run"))
    if not _text(start.get("workspace")) or not _RUN_ID.fullmatch(run_id):
        return
    scope = _Scope(payload, start)
    _violation(payload, scope)  # the policy already allowed this call; this collects scope.reads
    entry = _call_entry(payload)
    if entry is None and not scope.reads and not scope.shared_reads:
        return
    with _locked(root):
        path = root / "runs" / run_id / RUN_FILE
        run = read_json(path)
        if run.get("state") != "started":
            return
        if entry is not None:
            code = entry.pop("code", "")
            if code:
                # Numbered from the highest existing index, not the count: with
                # a count, removing one captured program makes the next capture
                # overwrite a program that was kept.
                used = [
                    int(match.group(1))
                    for path in (root / "src").glob(f"{run_id}_*.py")
                    if (match := re.fullmatch(rf"{re.escape(run_id)}_(\d+)\.py", path.name))
                ]
                number = max(used, default=0) + 1
                source = root / "src" / f"{run_id}_{number:02d}.py"
                source.write_text(code + ("" if code.endswith("\n") else "\n"), encoding="utf-8")
                entry["src"] = f"src/{source.name}"
            run["calls"] = [*run.get("calls", []), entry]
            if _hpc_function(entry["tool"]) == "run_batch" and entry.get("batch_id"):
                run["batches"] = sorted({*run.get("batches", []), entry["batch_id"]})
            if entry.get("skill") and entry["skill"] not in run.get("skills", []):
                run["skills"] = sorted({*run.get("skills", []), entry["skill"]})
                _append_log_unlocked(
                    root,
                    "note",
                    {
                        "run": run_id,
                        "what": f"reads campaign skill {entry['skill']}"
                        + ("" if entry.get("hashed") else " (version not pinned: skill not found locally)"),
                        "skill": {k: v for k, v in entry.items() if k in ("skill", "sha256", "bytes", "hashed", "skill_file")},
                        "actor": {"role": "code-generator", "session_id": _text(start.get("child_session_id"))},
                    },
                )
        inputs = list(run.get("inputs", []))
        known = {(i.get("plan_id"), i.get("hypothesis_id"), i.get("run"), i.get("path")) for i in inputs}
        for read in scope.reads:
            key = (read["plan_id"], read["hypothesis_id"], read["run"], read["path"])
            if key in known:
                continue
            known.add(key)
            item = {**read, **_source_hash(scope.base, read)}
            inputs.append(item)
            _append_log_unlocked(
                root,
                "note",
                {
                    "run": run_id,
                    "what": f"reads completed output {read['hypothesis_id']}/{read['path']} (plan {read['plan_id']})",
                    "input": item,
                    "actor": {"role": "code-generator", "session_id": _text(start.get("child_session_id"))},
                },
            )
        for read in scope.shared_reads:
            key = (None, None, None, read["path"])
            if key in known:
                continue
            known.add(key)
            item = {**read, **_local_input_hash(read["path"])}
            inputs.append(item)
            _append_log_unlocked(
                root,
                "note",
                {
                    "run": run_id,
                    "what": f"reads shared campaign input {read['path']}",
                    "input": item,
                    "actor": {"role": "code-generator", "session_id": _text(start.get("child_session_id"))},
                },
            )
        run["inputs"] = inputs
        _write_json(path, run)


# --- declared decisions (loam_record_decision) --------------------------------------

# Per-field cap on a recorded decision. A decision is a high-level choice, not a
# transcript; a model that pastes an essay here is told, in effect, to summarize.
MAX_DECISION_CHARS = 2000
DECISION_TOOL = "loam_record_decision"


def _decision_entry(args: Mapping[str, Any], session_id: str, role: str) -> dict[str, Any] | None:
    """One inline decision record, or None when there is no `what` to record."""

    what = _text(args.get("what"))
    if not what:
        return None
    raw_alts = args.get("alternatives")
    alternatives = (
        [a[:MAX_DECISION_CHARS] for item in raw_alts if (a := _text(item))]
        if isinstance(raw_alts, (list, tuple))
        else []
    )
    try:
        confidence: float | None = float(args["confidence"]) if args.get("confidence") is not None else None
    except (TypeError, ValueError):
        confidence = None
    entry: dict[str, Any] = {
        "ts": _timestamp(),
        "what": what[:MAX_DECISION_CHARS],
        "actor": {"role": role, "session_id": session_id},
    }
    if _text(args.get("why")):
        entry["why"] = _text(args.get("why"))[:MAX_DECISION_CHARS]
    if alternatives:
        entry["alternatives"] = alternatives
    if _text(args.get("expected")):
        entry["expected"] = _text(args.get("expected"))[:MAX_DECISION_CHARS]
    if confidence is not None:
        entry["confidence"] = confidence
    return entry


def record_decision(payload: Mapping[str, Any], start: Mapping[str, Any] | None) -> bool:
    """Append one declared decision inline to the caller's Pando record. Returns whether it stuck.

    Linked into the structure, not kept beside it: a code-generator's decisions
    live in its ``runs/rM/run.json`` ``decisions`` list, a validator's in its
    ``experiment.json`` ``decisions`` list, each also chained into ``log.jsonl``.
    The decision is the only record of WHY a tier was chosen -- the model emits
    no reasoning text and this deployment's provider returns no reasoning tokens
    (see the trace capture) -- so it is captured from a tool the agent calls,
    never scraped. Best effort and never raising into the tool handler: a
    provenance write must not fail the agent's turn.

    The reasoner has no hypothesis folder yet, so its decisions are handled with
    the plan-level record added in a later group; until then a reasoner decision
    returns False rather than being misfiled into a hypothesis.
    """

    if settings(payload) is None or not isinstance(start, Mapping):
        return False
    role = _text(start.get("role"))
    session_id = _text(start.get("child_session_id")) or _text(payload.get("session_id"))
    entry = _decision_entry(_arguments(payload), session_id, role)
    if entry is None:
        return False
    root_text = _text(start.get("workspace"))
    if not root_text:
        return False
    root = Path(root_text)
    log_fields = {key: value for key, value in entry.items() if key != "ts"}
    if role == "code-generator":
        run_id = _text(start.get("run"))
        if not _RUN_ID.fullmatch(run_id or ""):
            return False
        with _locked(root):
            path = root / "runs" / run_id / RUN_FILE
            run = read_json(path)
            if run.get("state") != "started":
                return False  # a frozen run takes no more decisions
            run["decisions"] = [*run.get("decisions", []), entry]
            _write_json(path, run)
            _append_log_unlocked(root, "decision", {"run": run_id, **log_fields})
        return True
    if role == "hypothesis-validator":
        with _locked(root):
            path = root / EXPERIMENT_FILE
            experiment = read_json(path)
            if experiment.get("state") != "open":
                return False
            experiment["decisions"] = [*experiment.get("decisions", []), entry]
            _write_json(path, experiment)
            _append_log_unlocked(root, "decision", log_fields)
        return True
    if role == "reasoner":
        with _locked(root):
            path = root / REASONER_FILE
            record = read_json(path)
            if record.get("state") != "open":
                return False
            record["decisions"] = [*record.get("decisions", []), entry]
            _write_json(path, record)
            _append_log_unlocked(root, "decision", log_fields)
        return True
    return False


def record_dispatch(
    root: Path, run_id: str, validator_session: str, requested_sha256: str, instruction_sha256: str
) -> None:
    """Link a reserved run to the validator that dispatched it, at dispatch time.

    The validator↔code-generator handoff, made explicit and checkable. Written
    before the child starts, onto two ends of one edge:

    * the reserved ``run.json`` gets ``dispatched_by`` (validator session, and
      the two hashes), and
    * the validator's ``experiment.json`` gets a ``children`` entry for the run.

    ``instruction_sha256`` is the hash of the canonical goal+context the child
    actually receives; ``requested_sha256`` is the validator's own ask before
    Loam augmented it with the layout, campaign and hypothesis blocks. They
    differ whenever Loam rewrote the task -- which is the norm, not the
    exception -- so the pair is what lets the validator's trace (the ask) and
    the child's trace (what ran) be reconciled rather than silently diverge.
    Idempotent: the dispatch middleware may run more than once for one round.
    """

    rewritten = requested_sha256 != instruction_sha256
    edge = {
        "validator_session": validator_session,
        "requested_sha256": requested_sha256,
        "instruction_sha256": instruction_sha256,
        "rewritten": rewritten,
    }
    with _locked(root):
        run_path = root / "runs" / run_id / RUN_FILE
        run = read_json(run_path)
        if run:
            run["dispatched_by"] = {**edge, "at": _timestamp()}
            _write_json(run_path, run)
        experiment_path = root / EXPERIMENT_FILE
        experiment = read_json(experiment_path)
        if experiment:
            others = [c for c in experiment.get("children", []) if isinstance(c, Mapping) and c.get("run") != run_id]
            child = {"run": run_id, "instruction_sha256": instruction_sha256, "rewritten": rewritten}
            experiment["children"] = sorted([*others, child], key=lambda c: c.get("run", ""))
            _write_json(experiment_path, experiment)
            if run_id not in {c.get("run") for c in others}:
                _append_log_unlocked(
                    root,
                    "note",
                    {"run": run_id,
                     "what": f"dispatched by validator {validator_session[:12] or '(unknown)'}"
                             + (" (instruction rewritten by Loam)" if rewritten else " (instruction verbatim)")},
                )


def _call_entry(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    """The run.json ``calls`` entry for a tool call, or None for calls not worth recording.

    Inline code is returned under ``code`` for the caller to save to src/.
    """

    tool, args = _text(payload.get("tool_name")), _arguments(payload)
    function = _hpc_function(tool)
    code = _text(args.get("payload")) if function == "run_code" else _text(args.get("code")) if tool == "execute_code" else ""
    entry: dict[str, Any] = {"tool": tool, "ts": _timestamp()}
    if code:
        entry.update(code=code, sha256=hashlib.sha256(code.encode("utf-8")).hexdigest())
        if function == "run_code" and _text(args.get("workdir")):
            entry["workdir"] = _text(args.get("workdir"))
    elif tool == "write_file":
        entry.update(path=_text(args.get("path")), sha256=hashlib.sha256(_text(args.get("content")).encode("utf-8")).hexdigest())
    elif tool in SKILL_READ_TOOLS:
        # Which reviewed recipe a run followed is part of how its result was
        # produced, so it belongs in the run's own record and not only in the
        # campaign-wide tool-call ledger. The skill's hash pins the version:
        # editing a skill afterwards cannot silently rewrite what a finished
        # run was told to do.
        name = _text(args.get("name"))
        if not name:
            return None  # skills_list with no category: a browse, not a read
        entry.update(skill=name, **_skill_digest(name, _text(args.get("file_path"))))
    elif function in {"run_batch", "register_tool"}:
        # The call's own `tool` argument names a registered HPC tool; keep it
        # apart from entry["tool"], the Hermes tool that was called.
        fields = {"script_path": "script_path", "inputs_path": "inputs_path", "tool": "registered_tool",
                  "name": "name", "batch_id": "batch_id"}
        entry.update({out: _text(args.get(key)) for key, out in fields.items() if _text(args.get(key))})
    else:
        return None
    return entry


def _declared_skills(payload: Mapping[str, Any]) -> list[str]:
    """The skills this campaign names for its children (loam_campaign.skills)."""

    campaign = payload.get("loam_campaign")
    campaign = campaign if isinstance(campaign, Mapping) else {}
    raw = campaign.get("skills")
    return [name for item in (raw if isinstance(raw, (list, tuple)) else []) if (name := _text(item))]


def _skill_digest(name: str, file_path: str = "") -> dict[str, Any]:
    """Hash the SKILL.md a child read, so the record pins the recipe's version.

    The skill library is local to the harness, so the hook can read it:
    ``$HERMES_HOME/skills/<category>/<name>/SKILL.md``, or the linked file a
    ``skill_view(file_path=...)`` asked for. Best effort -- an unresolvable name
    is recorded unhashed rather than dropped, because knowing a run followed
    some version of a recipe is still worth more than knowing nothing.
    """

    home = _text(os.environ.get("HERMES_HOME"))
    if not home or not _SEGMENT.fullmatch(name):
        return {"hashed": False}
    library = Path(home).expanduser() / "skills"
    try:
        matches = sorted(library.glob(f"*/{name}/SKILL.md")) or sorted(library.glob(f"{name}/SKILL.md"))
    except OSError:
        return {"hashed": False}
    if not matches:
        return {"hashed": False}
    target = matches[0]
    if file_path:
        candidate = target.parent / file_path
        # Keep the read inside the skill's own folder; a traversing file_path is
        # recorded by name only rather than followed.
        if _within(os.path.realpath(candidate), os.path.realpath(target.parent)) is None:
            return {"skill_file": file_path, "hashed": False}
        target = candidate
    try:
        sha256, size = _file_digest(target)
    except OSError:
        return {"hashed": False}
    digest: dict[str, Any] = {"sha256": sha256, "bytes": size, "hashed": True}
    if file_path:
        digest["skill_file"] = file_path
    return digest


def _local_input_hash(path: str) -> dict[str, Any]:
    """Hash a shared input the hook can see; a remote-only one is recorded unhashed.

    A shared path may be on HPC, where the hook has no access, so the hash is
    best effort. ``hashed: false`` says the record names the file but does not
    pin its bytes -- weaker provenance than a copied data/ input, which is the
    trade shared_inputs makes for not duplicating large reference data.
    """

    file = Path(path)
    try:
        if file.is_file():
            sha256, size = _file_digest(file)
            return {"sha256": sha256, "bytes": size, "hashed": True}
    except OSError:
        pass
    return {"hashed": False}


def _source_hash(base: Path, read: Mapping[str, str]) -> dict[str, str]:
    """The source run's recorded sha256 for a read file, when its manifest lists it."""

    run = read_json(base / read["plan_id"] / read["hypothesis_id"] / "runs" / read["run"] / RUN_FILE)
    for item in run.get("manifest", []):
        if isinstance(item, Mapping) and item.get("path") == read["path"] and item.get("sha256"):
            return {"sha256": str(item["sha256"])}
    return {}


# --- lifecycle at stop (subagent_stop) -----------------------------------------

# Bound on the child/validator report copied into a log entry.
MAX_RESULT_CHARS = 4_000
# The body may be empty: a run with no output files is a valid, empty manifest.
_MANIFEST_BLOCK = re.compile(r"^[ \t]*LOAM_MANIFEST[ \t]*\n(.*?)^[ \t]*END_LOAM_MANIFEST", re.DOTALL | re.MULTILINE)
_SHA256 = re.compile(r"[0-9a-f]{64}")


def run_state(child_status: str) -> str:
    """A run's terminal state from Hermes's child_status, read as ledger._observation_status does."""

    if child_status == "completed":
        return "completed"
    if child_status in {"interrupted", "cancelled"}:
        return "dropped"
    return "failed"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_manifest(root: Path, run_id: str) -> list[dict[str, Any]]:
    outputs = root / "outputs" / run_id
    return [
        {"path": path.relative_to(root).as_posix(), "sha256": _sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(outputs.rglob("*"))
        if path.is_file()
    ]


def manifest_program(outputs_dir: str) -> str:
    """The Python program that prints a LOAM_MANIFEST block for one run's outputs.

    Given to a remote child to run as its last call (hypothesis_orchestration.
    _layout_block), and run by .loam/reconcile/verify_workspace.py on HPC when a
    child didn't. One program, so both produce what remote_manifest parses.
    Hashes in 1 MB chunks so large outputs are fine.
    """

    return "\n".join((
        "import hashlib, json, os",
        f"root = {outputs_dir!r}",
        "print('LOAM_MANIFEST')",
        "for folder, _, files in sorted(os.walk(root)):",
        "    for name in sorted(files):",
        "        path = os.path.join(folder, name)",
        "        digest = hashlib.sha256()",
        "        with open(path, 'rb') as handle:",
        "            for chunk in iter(lambda: handle.read(1 << 20), b''):",
        "                digest.update(chunk)",
        "        print(json.dumps({'path': os.path.relpath(path, root), 'sha256': digest.hexdigest(), "
        "'bytes': os.path.getsize(path)}))",
        "print('END_LOAM_MANIFEST')",
    ))


def harvest_program(batch_root: str, outputs_dir: str) -> str:
    """The Python that copies one finished batch's results into a run's outputs.

    Each row folder's top level holds the promoted outputs of its successful
    attempt plus its stdout/stderr/status; failed attempts stay under
    attempt_*/, and lease.json / done are bookkeeping, so those are skipped.
    The child sets ``batch_id`` and runs it with run_code after batch_status
    reports the batch complete. Lands in <outputs_dir>/<batch_id>/run_<id>/.

    Re-runnable: copying a directory tolerates one already being there, because
    the retry after a partial harvest is a documented step and copytree's
    default would make it raise FileExistsError instead.
    """

    return "\n".join((
        "import os, shutil",
        "batch_id = 'SET_YOUR_BATCH_ID'",
        f"source = os.path.join({batch_root!r}, batch_id)",
        f"target = os.path.join({outputs_dir!r}, batch_id)",
        "for row in sorted(os.listdir(source)):",
        "    if not row.startswith('run_'):",
        "        continue",
        "    for name in sorted(os.listdir(os.path.join(source, row))):",
        "        if name.startswith('attempt_') or name in ('lease.json', 'done'):",
        "            continue",
        "        src, dst = os.path.join(source, row, name), os.path.join(target, row, name)",
        "        os.makedirs(os.path.dirname(dst), exist_ok=True)",
        "        if os.path.isdir(src):",
        "            shutil.copytree(src, dst, dirs_exist_ok=True)",
        "        else:",
        "            shutil.copy2(src, dst)",
        "print('harvested', batch_id, 'into', target)",
    ))


def remote_manifest(summary: str, run_id: str, prefix: str = "") -> tuple[list[dict[str, Any]], str]:
    """Parse the LAST ``LOAM_MANIFEST`` block out of text: (entries, status).

    On HPC the hook cannot list the run's files, so the child is told (see
    hypothesis_orchestration._layout_block) to print this block. Status is
    ``recorded``, ``missing`` (no block) or ``invalid`` (a malformed entry, or a
    path that leaves the folder it describes). Nothing is guessed.

    ``prefix`` is what the printed paths are relative to, as the record spells
    it: ``outputs/<run_id>`` for a run's outputs (the default, from ``run_id``),
    and empty for a seeded data/, whose entries are recorded relative to data/.
    """

    blocks = _MANIFEST_BLOCK.findall(summary)
    if not blocks:
        return [], "missing"
    entries: list[dict[str, Any]] = []
    for line in blocks[-1].splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            relative = os.path.normpath(str(item["path"]))
            digest, size = str(item["sha256"]), int(item["bytes"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return [], "invalid"
        if os.path.isabs(relative) or relative.split(os.sep)[0] == os.pardir or not _SHA256.fullmatch(digest):
            return [], "invalid"
        where = prefix if prefix else (f"outputs/{run_id}" if run_id else "")
        entries.append({"path": f"{where}/{relative}" if where else relative, "sha256": digest, "bytes": size})
    return entries, "recorded"


def finalize_run(root: Path, run_id: str, child_status: str, summary: str, remote: bool) -> dict[str, Any] | None:
    """Close a code-generator's run: terminal state, output manifest, a ``result`` entry.

    From here on the run is frozen: fs_policy refuses every write to a run that
    is not ``started``, and the validator-stop audit re-hashes local outputs
    against this manifest. Returns the updated run record, or None when the run
    was not started (already finalized, or never began).
    """

    with _locked(root):
        path = root / "runs" / run_id / RUN_FILE
        run = read_json(path)
        if run.get("state") != "started":
            return None
        if remote:
            manifest, manifest_status = remote_manifest(summary, run_id)
        else:
            manifest, manifest_status = _local_manifest(root, run_id), "recorded"
        run.update(
            state=run_state(child_status),
            child_status=child_status,
            ended_at=_timestamp(),
            manifest=manifest,
            manifest_status=manifest_status,
        )
        _write_json(path, run)
        _append_log_unlocked(
            root,
            "result",
            {
                "run": run_id,
                "state": run["state"],
                "results": summary[:MAX_RESULT_CHARS],
                "artifacts": [item["path"] for item in manifest],
                "manifest_status": manifest_status,
                "actor": {"role": "code-generator", "session_id": _text(run.get("child_session_id"))},
            },
        )
        return run


def close_workspace(root: Path, completion: Mapping[str, Any], remote: bool) -> list[str]:
    """Mirror the validator's completion into the workspace, then audit it.

    The completion signal itself stays the ``validator_completed`` row that
    hypothesis_orchestration.record_stop writes; ``completion`` is that row,
    copied here, never re-decided. Runs still ``reserved`` (the child never
    started) or ``started`` (it never stopped) are closed as ``dropped``.
    Returns the audit's problems, each also logged as a ``violation``.
    Idempotent: a second call on a closed workspace does nothing.
    """

    with _locked(root):
        experiment = read_json(root / EXPERIMENT_FILE)
        if experiment.get("state") != "open":
            return []
        for run_id in experiment.get("runs", []):
            path = root / "runs" / run_id / RUN_FILE
            run = read_json(path)
            if run.get("state") not in {"reserved", "started"}:
                continue
            reason = "child never started" if run["state"] == "reserved" else "validator finished before this child stopped"
            run.update(state="dropped", ended_at=_timestamp(), dropped_reason=reason)
            _write_json(path, run)
            _append_log_unlocked(root, "note", {"run": run_id, "what": f"run {run_id} dropped: {reason}"})
        experiment.update(
            state="closed",
            closed_at=_timestamp(),
            validator_status=_text(completion.get("status")),
            evidence_score=completion.get("evidence_score"),
            evidence_score_status=completion.get("evidence_score_status"),
        )
        _write_json(root / EXPERIMENT_FILE, experiment)
        _append_log_unlocked(
            root,
            "report",
            {
                "what": "hypothesis-validator finished",
                "status": _text(completion.get("status")),
                "evidence_score": completion.get("evidence_score"),
                "evidence_score_status": completion.get("evidence_score_status"),
                "results": _text(completion.get("summary"))[:MAX_RESULT_CHARS],
                "actor": {"role": "hypothesis-validator", "session_id": _text(completion.get("child_session_id"))},
            },
        )
        problems = _audit_unlocked(root, experiment, remote)
        for problem in problems:
            _append_log_unlocked(root, "violation", {"what": f"audit: {problem}", "blocked": False})
        return problems


def _audit_unlocked(root: Path, experiment: Mapping[str, Any], remote: bool) -> list[str]:
    """What the pre-tool checks could not prevent: tampering, edits after freeze, strays.

    Observe and report only; nothing is repaired or removed.
    """

    problems = [f"log.jsonl {problem}" for problem in verify_log(root)]
    runs = [run_id for run_id in experiment.get("runs", []) if isinstance(run_id, str)]
    for entry in sorted(root.iterdir()):
        transient = entry.name.startswith(".") and entry.name.endswith(".tmp")
        if entry.name not in LAYOUT_DIRS and entry.name not in RECORD_FILES and not transient:
            problems.append(f"stray top-level entry {entry.name}")
    for folder in ("runs", "outputs", "scratch"):
        for entry in sorted((root / folder).iterdir()) if (root / folder).is_dir() else []:
            if entry.name not in runs:
                problems.append(f"{folder}/{entry.name} is not a run of this hypothesis")
    for run_id in runs:
        run = read_json(root / "runs" / run_id / RUN_FILE)
        if run.get("state") != "completed" or remote or run.get("manifest_status") != "recorded":
            continue  # remote outputs live on HPC; only local bytes can be re-hashed
        recorded = {item["path"]: item.get("sha256") for item in run.get("manifest", []) if isinstance(item, Mapping)}
        present = {path.relative_to(root).as_posix(): path for path in (root / "outputs" / run_id).rglob("*") if path.is_file()}
        for relative, digest in recorded.items():
            if relative not in present:
                problems.append(f"{relative} was removed after run {run_id} finished")
            elif _sha256_file(present[relative]) != digest:
                problems.append(f"{relative} changed after run {run_id} finished")
        for relative in sorted(set(present) - set(recorded)):
            problems.append(f"{relative} was added after run {run_id} finished")
    return problems


# --- per-agent trace capture (state.db -> trace.jsonl) ------------------------------

TRACE_FILE = "trace.jsonl"
# Columns the trace keeps from Hermes's messages table, in a stable order. The
# reasoning_* columns are read even though this deployment's provider leaves them
# empty: a provider that returns reasoning fills them with no change here.
_TRACE_COLUMNS = (
    "role", "content", "tool_name", "tool_calls", "tool_call_id",
    "reasoning", "reasoning_content", "finish_reason", "token_count", "timestamp",
)


def trace_db_path(payload: Mapping[str, Any]) -> Path | None:
    """Hermes's state.db for this campaign, or None when it cannot be located.

    HERMES_HOME is set per campaign by run-campaign and inherited by the hook
    subprocess; state.db lives at its root (hermes_state.DEFAULT_DB_PATH). A
    payload override wins, for tests and for a non-standard home.
    """

    override = _text((settings(payload) or {}).get("trace_db"))
    home = override or _text(payload.get("hermes_home")) or _text(os.environ.get("HERMES_HOME"))
    if not home:
        return None
    candidate = Path(home).expanduser()
    db = candidate if candidate.name == "state.db" else candidate / "state.db"
    return db if db.is_file() else None


def read_trace(db_path: Path, session_id: str, full: bool) -> list[dict[str, Any]]:
    """One session's turns from state.db, oldest first, normalized. Best effort -> [] on any error.

    A skill read's body is replaced with a short marker unless ``full``: it is
    already hashed under the run's ``skills`` (run.json), so the default trace is
    the reasoning/tool spine, not a second copy of the skill library.
    """

    import sqlite3

    if not session_id:
        return []
    try:
        # Read-only: the live campaign may still hold the write lock (WAL).
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5.0)
    except sqlite3.Error:
        return []
    rows: list[dict[str, Any]] = []
    try:
        connection.row_factory = sqlite3.Row
        cursor = connection.execute(
            f"SELECT {', '.join(_TRACE_COLUMNS)} FROM messages "
            "WHERE session_id = ? ORDER BY timestamp, id",
            (session_id,),
        )
        for index, record in enumerate(cursor, start=1):
            rows.append(_trace_row(index, record, full))
    except sqlite3.Error:
        return []
    finally:
        connection.close()
    return rows


def _trace_row(seq: int, record: Mapping[str, Any], full: bool) -> dict[str, Any]:
    role = _text(record["role"])
    tool = _text(record["tool_name"])
    entry: dict[str, Any] = {"seq": seq, "role": role}
    timestamp = record["timestamp"]
    if isinstance(timestamp, (int, float)):
        entry["ts"] = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
    content = _text(record["content"])
    if content:
        if not full and role == "tool" and tool in SKILL_READ_TOOLS:
            # The body is already pinned by its sha256 under run.json `skills`.
            entry["content"] = f"<{tool} body omitted; {len(content)} chars, hashed under skills>"
        else:
            entry["content"] = content
    for key in ("tool_name", "tool_call_id", "tool_calls", "reasoning", "reasoning_content", "finish_reason"):
        value = _text(record[key])
        if value:
            entry[key] = value
    return entry


def write_trace(dest_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Write trace.jsonl into ``dest_dir`` and return its {path, sha256, bytes, rows} pointer.

    ``path`` is relative to the workspace root (dest_dir's parent chain), so the
    pointer reads like a manifest entry. None when there is nothing to write.
    """

    if not rows:
        return None
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / TRACE_FILE
    body = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    data = body.encode("utf-8")
    tmp = target.with_name(f".{target.name}.tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "rows": len(rows)}


def capture_trace(payload: Mapping[str, Any], start: Mapping[str, Any]) -> bool:
    """Copy a stopped agent's conversation out of state.db into its workspace dir.

    Linked into the record like everything else: the trace file is referenced by
    a {path, sha256, bytes, rows} pointer on the hub (run.json for a
    code-generator, experiment.json for a validator) and announced in the chain.
    The reasoner is plan-level and handled with its own record in a later group.
    Best effort: a provenance copy must never raise into the stop path.
    """

    db_path = trace_db_path(payload)
    session_id = _text(start.get("child_session_id"))
    if db_path is None or not session_id:
        return False
    root = Path(_text(start.get("workspace")))
    role = _text(start.get("role"))
    full = (settings(payload) or {}).get("trace_full") is True
    spec = _trace_destination(root, role, _text(start.get("run")))
    if spec is None:
        return False
    dest, where, log_extra = spec
    return _persist_trace(root, dest, where, log_extra, session_id, db_path, full)


def _trace_destination(root: Path, role: str, run_id: str) -> tuple[Path, Path, dict[str, Any]] | None:
    """(trace dir, hub file, log fields) for a role, or None when it has no record."""

    if role == "code-generator":
        if not _RUN_ID.fullmatch(run_id or ""):
            return None
        return root / "runs" / run_id, root / "runs" / run_id / RUN_FILE, {"run": run_id}
    if role == "hypothesis-validator":
        return root / "validator", root / EXPERIMENT_FILE, {}
    if role == "reasoner":
        # root IS the reasoner dir; its trace sits in it, pointer on reasoner.json.
        return root, root / REASONER_FILE, {}
    return None


def _persist_trace(
    root: Path, dest: Path, where: Path, log_extra: dict[str, Any],
    session_id: str, db_path: Path, full: bool,
) -> bool:
    """Read a session's trace, write trace.jsonl, and link its pointer into the hub record."""

    pointer = write_trace(dest, read_trace(db_path, session_id, full))
    if pointer is None:
        return False
    pointer["path"] = (dest / TRACE_FILE).relative_to(root).as_posix()
    with _locked(root):
        record = read_json(where)
        if not record:
            return False
        record["trace"] = pointer
        _write_json(where, record)
        _append_log_unlocked(root, "trace", {**log_extra, "trace": pointer})
    return True


def backfill_traces(root: Path, db_path: Path, full: bool) -> list[str]:
    """Fill any MISSING trace in one hypothesis workspace, for the validator and each run.

    The stop-time capture (capture_trace) covers an agent that stopped cleanly;
    an agent whose subagent_stop never fired -- a crash, an Argo drop -- leaves
    its trace uncaptured though its turns are still in state.db. This re-reads
    them from the session ids the records already hold. Returns what it filled.
    """

    filled: list[str] = []
    experiment = read_json(root / EXPERIMENT_FILE)
    if experiment and not experiment.get("trace"):
        session_id = _text(experiment.get("validator_session_id"))
        if session_id and _persist_trace(root, root / "validator", root / EXPERIMENT_FILE, {}, session_id, db_path, full):
            filled.append("validator")
    runs_dir = root / "runs"
    for run_path in sorted(runs_dir.iterdir()) if runs_dir.is_dir() else []:
        run_id = run_path.name
        if not _RUN_ID.fullmatch(run_id):
            continue
        run = read_json(run_path / RUN_FILE)
        if not run or run.get("trace"):
            continue
        session_id = _text(run.get("child_session_id"))
        if session_id and _persist_trace(root, run_path, run_path / RUN_FILE, {"run": run_id}, session_id, db_path, full):
            filled.append(run_id)
    return filled


def backfill_reasoner_trace(reasoner_dir: Path, db_path: Path, full: bool) -> bool:
    """Fill a missing reasoner trace from the session id its record holds."""

    record = read_json(reasoner_dir / REASONER_FILE)
    if not record or record.get("trace"):
        return False
    session_id = _text(record.get("session_id"))
    return bool(session_id and _persist_trace(reasoner_dir, reasoner_dir, reasoner_dir / REASONER_FILE, {}, session_id, db_path, full))


def on_agent_stopped(
    payload: Mapping[str, Any], start: Mapping[str, Any], completion: Mapping[str, Any] | None
) -> list[str]:
    """Workspace side of subagent_stop. Returns audit problems (validator stop only).

    ``start`` is the child's agent_started row; ``completion`` is the row
    record_stop just wrote for it. Raises on failure; subagent-stop.py isolates
    this step so it can never drop the completion rows.
    """

    if settings(payload) is None or not _text(start.get("workspace")):
        return []
    root = Path(_text(start.get("workspace")))
    execution = payload.get("loam_execution")
    remote = isinstance(execution, Mapping) and _text(execution.get("mode")) == "remote_mcp"
    role = _text(start.get("role"))
    if role == "code-generator" and _text(start.get("run")):
        run_id = _text(start.get("run"))
        run = finalize_run(
            root, run_id, _text(payload.get("child_status")) or "error", _text(payload.get("child_summary")), remote
        )
        # After finalize_run, so the trace pointer lands on the frozen run record.
        _capture_trace_safe(payload, start)
        if remote and run is not None and run.get("state") == "completed" and run.get("manifest_status") not in VERIFIED_MANIFESTS:
            _launch_verifier(payload, start, root, ["fill", "--run", run_id])
        if remote and run is not None and run.get("batches"):
            _launch_verifier(payload, start, root, ["batches", "--run", run_id])
        return []
    if role == "hypothesis-validator" and completion is not None:
        problems = close_workspace(root, completion, remote)
        _capture_trace_safe(payload, start)
        if remote:
            _launch_verifier(payload, start, root, ["recheck"])
        return problems
    if role == "reasoner":
        close_reasoner(root, _text(payload.get("child_summary")))
        _capture_trace_safe(payload, start)
        return []
    return []


def _capture_trace_safe(payload: Mapping[str, Any], start: Mapping[str, Any]) -> None:
    """capture_trace, isolated: a trace copy must never fail the stop path."""

    try:
        capture_trace(payload, start)
    except Exception as exc:
        print(f"loam workspace: trace capture failed: {type(exc).__name__}: {exc}", file=sys.stderr)


VERIFIER = Path(__file__).resolve().parents[1] / "reconcile" / "verify_workspace.py"


def _launch_verifier(payload: Mapping[str, Any], start: Mapping[str, Any], root: Path, args: list[str]) -> None:
    """Start verify_workspace.py detached, so the stop hook returns at once.

    The hook has a 15 s budget and no MCP access; hashing files on HPC needs
    both. Unconfigured (no verifier_python / verifier_mcp_url) -> a note
    saying the run stays unverified, nothing launched.
    """

    config = settings(payload) or {}
    python, url = _text(config.get("verifier_python")), _text(config.get("verifier_mcp_url"))
    execution = payload.get("loam_execution")
    execution = execution if isinstance(execution, Mapping) else {}
    workdir_root = _text(execution.get("workdir_root"))
    if not (python and url and workdir_root):
        append_log(root, "note", {"what": f"verifier not configured; {' '.join(args)} not run (run stays as recorded)"})
        return
    remote_root = os.path.join(os.path.normpath(workdir_root), _text(start.get("plan_id")), _text(start.get("hypothesis_id")))
    command = [
        python, str(VERIFIER), *args,
        "--workspace", str(root),
        "--remote-root", remote_root,
        "--mcp-url", url,
        "--tool", _text(execution.get("function_name")) or "run_code",
    ]
    state_dir = Path(_text(payload.get("loam_state_dir")) or root.parent.parent)
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "verify-workspace.log").open("ab") as log:
        subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, close_fds=True
        )
    append_log(root, "note", {"what": f"verifier launched: {' '.join(args)}"})


def _launch_seed(payload: Mapping[str, Any], plan_id: str, hypothesis_id: str, root: Path) -> None:
    """Launch the verifier's ``seed`` for a staged HPC inputs directory, once per hypothesis.

    Only for remote execution: with a local workdir the hook copies the inputs
    itself in ensure_workspace. Skipped once experiment.json records a remote
    seed, so a validator restarting on an existing workspace does not re-copy.
    """

    config = settings(payload) or {}
    source = _text(config.get("remote_inputs_dir"))
    execution = payload.get("loam_execution")
    execution = execution if isinstance(execution, Mapping) else {}
    if not source or _text(execution.get("mode")) != "remote_mcp":
        return
    recorded = read_json(root / EXPERIMENT_FILE).get("inputs", [])
    if any(isinstance(item, Mapping) and item.get("how") == "copied_on_hpc" for item in recorded):
        return
    _launch_verifier(
        payload,
        {"plan_id": plan_id, "hypothesis_id": hypothesis_id},
        root,
        ["seed", "--source", source, "--max-bytes", str(input_cap(payload))],
    )


# --- verifier side (called from .loam/reconcile/verify_workspace.py) ---------------


def record_verified_manifest(root: Path, run_id: str, entries: list[dict[str, Any]]) -> bool:
    """Store a manifest the verifier hashed on HPC, for a completed run that lacked one."""

    with _locked(root):
        path = root / "runs" / run_id / RUN_FILE
        run = read_json(path)
        if run.get("state") != "completed" or run.get("manifest_status") in VERIFIED_MANIFESTS:
            return False
        run.update(manifest=entries, manifest_status="recorded_by_loam", verified_at=_timestamp())
        _write_json(path, run)
        _append_log_unlocked(
            root,
            "note",
            {"run": run_id, "what": f"output manifest recorded by the Loam verifier ({len(entries)} files)",
             "artifacts": [item["path"] for item in entries]},
        )
        return True


def manifest_changes(recorded: list[dict[str, Any]], fresh: list[dict[str, Any]], run_id: str) -> list[str]:
    """How a fresh hashing of a run's outputs differs from its recorded manifest."""

    before = {item.get("path"): item.get("sha256") for item in recorded if isinstance(item, Mapping)}
    after = {item.get("path"): item.get("sha256") for item in fresh if isinstance(item, Mapping)}
    problems = [f"{path} was removed after run {run_id} finished" for path in sorted(set(before) - set(after))]
    problems += [f"{path} changed after run {run_id} finished" for path in sorted(set(before) & set(after)) if before[path] != after[path]]
    problems += [f"{path} was added after run {run_id} finished" for path in sorted(set(after) - set(before))]
    return problems


def record_batch_check(root: Path, run_id: str, unfinished: list[str]) -> None:
    """Store which of a run's batches were still unfinished when the run closed.

    The run stays ``completed`` (Loam's completion signals are unchanged);
    the flag and a violation per batch make the incomplete harvest visible.
    """

    with _locked(root):
        path = root / "runs" / run_id / RUN_FILE
        run = read_json(path)
        if not run:
            # Writing here would create a run.json holding nothing but the
            # batch fields, which reads like a run that never existed.
            _append_log_unlocked(
                root, "violation",
                {"run": run_id, "what": f"batch check found no record for run {run_id}", "blocked": False},
            )
            return
        run.update(batches_unfinished=unfinished, batches_checked_at=_timestamp())
        _write_json(path, run)
        for batch_id in unfinished:
            _append_log_unlocked(
                root, "violation",
                {"run": run_id, "what": f"batch {batch_id} was unfinished when run {run_id} closed", "blocked": False},
            )


def log_violations(root: Path, problems: list[str], run_id: str = "") -> None:
    with _locked(root):
        for problem in problems:
            _append_log_unlocked(root, "violation", {"run": run_id, "what": problem, "blocked": False})
