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
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

EXPERIMENT_FILE = "experiment.json"
LOG_FILE = "log.jsonl"
LOCK_FILE = ".workspace.lock"
RUN_FILE = "run.json"
LAYOUT_DIRS = ("src", "runs", "outputs", "scratch", "data")
# Hook-owned top-level files. With LAYOUT_DIRS this is the whole legal top level.
RECORD_FILES = (EXPERIMENT_FILE, LOG_FILE, LOCK_FILE)
LOG_KINDS = ("plan", "result", "note", "report", "violation")
# The hash a first log line chains to.
GENESIS_SHA256 = "0" * 64
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


def ensure_workspace(
    payload: Mapping[str, Any],
    plan_id: str,
    hypothesis_id: str,
    statement: str = "",
    validator_session_id: str = "",
) -> Path:
    """Create the hypothesis folder and its open experiment.json; idempotent."""

    root = workspace_root(payload, plan_id, hypothesis_id)
    with _locked(root):
        for name in LAYOUT_DIRS:
            (root / name).mkdir(parents=True, exist_ok=True)
        experiment = root / EXPERIMENT_FILE
        if experiment.is_file():
            return root
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
    return root


def _next_run_id(root: Path) -> str:
    numbers = [
        int(match.group(1))
        for path in (root / "runs").iterdir()
        if path.is_dir() and (match := _RUN_ID.fullmatch(path.name))
    ] if (root / "runs").is_dir() else []
    return f"r{max(numbers, default=0) + 1}"


def _reserve_run_unlocked(root: Path) -> str:
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
            "goal": "",
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


def reserve_runs(root: Path, count: int) -> list[str]:
    """Reserve run ids for children about to be delegated.

    A child's instructions are written when its validator calls delegate_task,
    before the child exists, yet they must name its outputs/rM folder. So the
    ids are reserved here and travel in a Loam-forced [loam-run-id=rM] goal tag;
    start_run moves each one to "started" once the child really begins. A run
    that never starts stays "reserved" and is closed as dropped later.
    """

    with _locked(root):
        return [_reserve_run_unlocked(root) for _ in range(count)]


def start_run(root: Path, run_id: str, child_session_id: str, goal: str, why: str = "") -> str:
    """Mark a reserved run as started and log its plan."""

    with _locked(root):
        return _start_run_unlocked(root, run_id, child_session_id, goal, why)


def allocate_run(root: Path, child_session_id: str, goal: str, why: str = "") -> str:
    """Reserve and start the next run in one step, for a child that arrived without a reserved id."""

    with _locked(root):
        run_id = _reserve_run_unlocked(root)
        return _start_run_unlocked(root, run_id, child_session_id, goal, why)


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

    if settings(payload) is None or role not in {"hypothesis-validator", "code-generator"} or not hypothesis_id:
        return {}
    session_id = _text(payload.get("child_session_id"))
    try:
        if role == "hypothesis-validator":
            root = ensure_workspace(payload, plan_id, hypothesis_id, statement, session_id)
            return {"workspace": str(root)}
        root = workspace_root(payload, plan_id, hypothesis_id)
        why = f"delegated by the hypothesis-validator for {hypothesis_id}"
        if run_id:
            run = start_run(root, run_id, session_id, goal, why)
        else:
            run = allocate_run(root, session_id, goal, why)
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


class _Scope:
    """The places one code-generator may touch, locally and on HPC."""

    def __init__(self, payload: Mapping[str, Any], start: Mapping[str, Any]):
        config = settings(payload) or {}
        self.root = Path(_text(start.get("workspace")))
        self.real_root = os.path.realpath(self.root)
        self.base = Path(_text(config.get("root"))).expanduser()
        self.real_base = os.path.realpath(self.base)
        self.run_id = _text(start.get("run"))
        # Completed outputs of OTHER hypotheses this call reads; capture_code
        # records them as the run's inputs.
        self.reads: list[dict[str, str]] = []
        execution = payload.get("loam_execution")
        workdir_root = _text(execution.get("workdir_root")) if isinstance(execution, Mapping) else ""
        self.remote_base = os.path.normpath(workdir_root) if workdir_root else ""
        plan_id, hypothesis_id = _text(start.get("plan_id")), _text(start.get("hypothesis_id"))
        self.remote_root = os.path.join(self.remote_base, plan_id, hypothesis_id) if self.remote_base else ""

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
                return self._foreign(foreign, raw, writing)
            return f"{raw} is outside this hypothesis's workspace {self.root}" if writing else None
        return _layout_problem(parts, self.run_id) if writing else None

    def _foreign(self, parts: list[str], raw: str, writing: bool) -> str | None:
        """A path in ANOTHER hypothesis's workspace; ``parts`` are relative to the base.

        Other hypotheses are never writable. Their finished results are
        readable, Pando-style (a run consuming ``output:eN/rM``): only
        ``outputs/rK/`` of a run whose state is ``completed``, never scratch/,
        a run still in progress, or a failed or dropped one. An allowed read is
        collected in ``self.reads`` so it is recorded as an input.
        """

        if writing:
            return f"{raw} belongs to another hypothesis's workspace, which is read-only to you"
        legal = "only outputs/rK/ of another hypothesis's completed runs may be read"
        if len(parts) < 4 or parts[2] != "outputs" or not _RUN_ID.fullmatch(parts[3]):
            return f"{raw}: {legal}"
        plan_id, hypothesis_id, run_id = parts[0], parts[1], parts[3]
        state = read_json(self.base / plan_id / hypothesis_id / "runs" / run_id / RUN_FILE).get("state")
        if state != "completed":
            return f"{raw}: run {run_id} of {hypothesis_id} is {state or 'unknown'}, not completed; {legal}"
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
        path = os.path.normpath(raw if os.path.isabs(raw) else os.path.join(self.remote_root, raw))
        parts = _within(path, self.remote_root)
        if parts is None:
            foreign = _within(path, self.remote_base)
            if foreign is not None:
                # The HPC folder mirrors the local layout, so the local run.json
                # says whether that remote run completed.
                return self._foreign(foreign, raw, writing)
            return f"{label} {raw} is outside this hypothesis's HPC workspace {self.remote_root}" if required else None
        return _layout_problem(parts, self.run_id) if writing else None

    def code(self, text: str) -> str | None:
        """Check paths into other hypotheses' workspaces named in code, locally or on HPC.

        Code can't be analysed for what it does with a path, so each one is held
        to the read rule: allowed only for a completed run's outputs.
        """

        for base in {self.real_base, str(self.base), self.remote_base} - {""}:
            pattern = re.escape(base.rstrip("/")) + r"/([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+)((?:/[^\s'\"(),;\]]*)?)"
            for match in re.finditer(pattern, text):
                if os.path.join(base, match.group(1), match.group(2)) in {self.real_root, str(self.root), self.remote_root}:
                    continue
                parts = [match.group(1), match.group(2), *(p for p in match.group(3).split("/") if p)]
                problem = self._foreign(parts, match.group(0), writing=False)
                if problem:
                    return f"code refers to another hypothesis's workspace: {problem}"
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
    reading_only = tool in _READ_TOOLS or _hpc_function(tool) in _HPC_STATUS
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
    if entry is None and not scope.reads:
        return
    with _locked(root):
        path = root / "runs" / run_id / RUN_FILE
        run = read_json(path)
        if run.get("state") != "started":
            return
        if entry is not None:
            code = entry.pop("code", "")
            if code:
                number = sum(1 for _ in (root / "src").glob(f"{run_id}_*.py")) + 1
                source = root / "src" / f"{run_id}_{number:02d}.py"
                source.write_text(code + ("" if code.endswith("\n") else "\n"), encoding="utf-8")
                entry["src"] = f"src/{source.name}"
            run["calls"] = [*run.get("calls", []), entry]
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
        run["inputs"] = inputs
        _write_json(path, run)


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
    elif function in {"run_batch", "register_tool"}:
        entry.update({key: _text(args.get(key)) for key in ("script_path", "inputs_path", "tool", "name") if _text(args.get(key))})
    else:
        return None
    return entry


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
_MANIFEST_BLOCK = re.compile(r"LOAM_MANIFEST[ \t]*\n(.*?)\n[ \t]*END_LOAM_MANIFEST", re.DOTALL)
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


def remote_manifest(summary: str, run_id: str) -> tuple[list[dict[str, Any]], str]:
    """Parse the child's LAST ``LOAM_MANIFEST`` block from its report: (entries, status).

    On HPC the hook cannot list the run's files, so the child is told (see
    hypothesis_orchestration._layout_block) to print this block. Status is
    ``recorded``, ``missing`` (no block) or ``invalid`` (a malformed entry, or a
    path that leaves outputs/rM). Nothing is guessed.
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
        entries.append({"path": f"outputs/{run_id}/{relative}", "sha256": digest, "bytes": size})
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
        finalize_run(
            root,
            _text(start.get("run")),
            _text(payload.get("child_status")) or "error",
            _text(payload.get("child_summary")),
            remote,
        )
        return []
    if role == "hypothesis-validator" and completion is not None:
        return close_workspace(root, completion, remote)
    return []
