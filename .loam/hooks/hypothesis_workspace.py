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
        self.real_base = os.path.realpath(Path(_text(config.get("root"))).expanduser())
        self.run_id = _text(start.get("run"))
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
            if _within(real, self.real_base) is not None:
                return f"{raw} belongs to another hypothesis's workspace"
            return f"{raw} is outside this hypothesis's workspace {self.root}" if writing else None
        return _layout_problem(parts, self.run_id) if writing else None

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
            if _within(path, self.remote_base) is not None:
                return f"{raw} belongs to another hypothesis's HPC workspace"
            return f"{label} {raw} is outside this hypothesis's HPC workspace {self.remote_root}" if required else None
        return _layout_problem(parts, self.run_id) if writing else None

    def code(self, text: str) -> str | None:
        """Block code that names another hypothesis's workspace, locally or on HPC."""

        for base in {self.real_base, str(self.root.parent.parent), self.remote_base} - {""}:
            pattern = re.escape(base.rstrip("/")) + r"/([A-Za-z0-9._-]+)/([A-Za-z0-9._-]+)"
            for match in re.finditer(pattern, text):
                where = os.path.join(base, match.group(1), match.group(2))
                if where not in {self.real_root, str(self.root), self.remote_root}:
                    return f"code refers to another hypothesis's workspace ({where})"
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
    """Record what code an allowed code-generator call runs, in the run's record.

    Code passed inline (HPC run_code ``payload``, local execute_code ``code``) is
    saved as src/rM_NN.py, so the exact program behind a result is kept even
    when it ran remotely. Calls that point at files (write_file, run_batch,
    register_tool) are recorded by path, with a hash when the content is in the
    call. Best effort; the caller isolates failures.
    """

    if settings(payload) is None or not isinstance(start, Mapping) or _text(start.get("role")) != "code-generator":
        return
    root, run_id = Path(_text(start.get("workspace"))), _text(start.get("run"))
    if not _text(start.get("workspace")) or not _RUN_ID.fullmatch(run_id):
        return
    tool, args = _text(payload.get("tool_name")), _arguments(payload)
    function = _hpc_function(tool)
    code = _text(args.get("payload")) if function == "run_code" else _text(args.get("code")) if tool == "execute_code" else ""
    entry: dict[str, Any] = {"tool": tool, "ts": _timestamp()}
    if code:
        entry["sha256"] = hashlib.sha256(code.encode("utf-8")).hexdigest()
        if function == "run_code" and _text(args.get("workdir")):
            entry["workdir"] = _text(args.get("workdir"))
    elif tool == "write_file":
        entry.update(path=_text(args.get("path")), sha256=hashlib.sha256(_text(args.get("content")).encode("utf-8")).hexdigest())
    elif function in {"run_batch", "register_tool"}:
        entry.update({key: _text(args.get(key)) for key in ("script_path", "inputs_path", "tool", "name") if _text(args.get(key))})
    else:
        return
    with _locked(root):
        path = root / "runs" / run_id / RUN_FILE
        run = read_json(path)
        if run.get("state") != "started":
            return
        if code:
            number = sum(1 for _ in (root / "src").glob(f"{run_id}_*.py")) + 1
            source = root / "src" / f"{run_id}_{number:02d}.py"
            source.write_text(code + ("" if code.endswith("\n") else "\n"), encoding="utf-8")
            entry["src"] = f"src/{source.name}"
        run["calls"] = [*run.get("calls", []), entry]
        _write_json(path, run)
