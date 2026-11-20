"""Tools that touch the world: run code, read results, fetch records.

Restriction is by construction: an agent can only call the tools it was given.
Resilience is by wrapping: every tool runs inside `Resilient`, so a failure or timeout comes back
to the model as a failed tool result it can work around, never an exception that ends its run.
"""

from __future__ import annotations

import asyncio
import contextlib
import ipaddress
import os
import signal
import socket
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import httpx
from pydantic_ai import FunctionToolset, ModelRetry, RunContext
from pydantic_ai.exceptions import AgentRunError, ApprovalRequired, CallDeferred, ToolFailed
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, DynamicToolset, WrapperToolset

from .config import Config
from .memory import Memory, NoMemory
from .workspace import Workspace

# Tool names on loam's hpc-deployment MCP server, split by who may call them.
# This is an ALLOWLIST and `execution_tools` filters on it, so a tool the server
# exposes but that is missing here is absent from the agent's toolset entirely --
# however loudly a prompt or skill recommends it. Add both together.
#
# `list_tools` is read-only discovery, so every role gets it: a validator plans
# better when it can see which stages an earlier round already proved.
# `register_tool` freezes a script, smoke-runs it and declares its `expects`
# output contract, so only the code-generator may call it.
MCP_READ = {'inspect_filesystem', 'batch_status', 'list_batches', 'pool_status',
            'list_tools'}
MCP_FULL = MCP_READ | {'write_files', 'run_batch', 'run_code', 'register_tool'}


@dataclass
class Deps:
    """Per-hypothesis context shared by a validator, its code-generators and its reviewer."""

    cfg: Config
    skills: dict[str, Path]
    ws: Workspace
    round: int
    hypothesis: str
    run_dir: str = ''          # set only for a code-generator: the one directory it may write to
    code_calls: int = 0
    code_results: list = field(default_factory=list)   # every CodeResult, for the round record
    shell_executed: bool = False
    submitted_batches: set[str] = field(default_factory=set)
    executed_batches: set[str] = field(default_factory=set)
    memory: Memory = field(default_factory=NoMemory)   # recall only, for the code-generator prompt


@dataclass
class Resilient(WrapperToolset[Deps]):
    """Bounds every tool call by `timeout` and turns any failure into ToolFailed: the model sees the
    error as the tool's result, and it does not spend the run's retry budget."""

    timeout: float = 1800

    async def call_tool(self, name, tool_args, ctx, tool):
        try:
            async with asyncio.timeout(self.timeout):
                return await super().call_tool(name, tool_args, ctx, tool)
        except (ToolFailed, CallDeferred, ApprovalRequired, AgentRunError):
            raise
        except ModelRetry as exc:  # raised by MCP tool errors; retries are counted and eventually fatal
            raise ToolFailed(exc.message) from exc
        except TimeoutError as exc:
            raise ToolFailed(f'{name} timed out after {self.timeout:.0f}s') from exc
        except Exception as exc:
            raise ToolFailed(f'{name} failed: {exc!r}') from exc


@dataclass
class OwnedMCP(WrapperToolset[Deps]):
    """Confine MCP calls to this hypothesis's workspace, and save a receipt for every batch submitted,
    so its validator can poll it later, even if the code-generator that submitted it crashed."""

    async def call_tool(self, name, tool_args, ctx, tool):
        args = dict(tool_args)
        run_id = PurePosixPath(ctx.deps.run_dir).name
        if name == 'batch_status' and args.get('batch_id') not in ctx.deps.ws.batch_ids():
            raise ToolFailed(f"batch {args.get('batch_id')!r} does not belong to this hypothesis")
        if name == 'list_batches':
            args['hypothesis_id'] = ctx.deps.ws.hypothesis_id
        if name == 'inspect_filesystem' and args.get('path'):
            args['path'] = str(_remote_path(ctx.deps.ws.root, args['path'], allow_root=True))
        if name == 'write_files':
            args['files'] = {str(_remote_path(ctx.deps.run_dir, path)): text
                             for path, text in args.get('files', {}).items()}
        if name == 'run_batch':
            for key in ('script_path', 'inputs_path'):
                if args.get(key):
                    args[key] = str(_remote_path(ctx.deps.run_dir, args[key]))
            args['labels'] = {**(args.get('labels') or {}),
                              'hypothesis_id': ctx.deps.ws.hypothesis_id, 'run_id': run_id}
        if name == 'run_code':
            args['workdir'] = ctx.deps.run_dir
        result = await super().call_tool(name, args, ctx, tool)
        if name == 'run_batch' and isinstance(result, dict) and result.get('batch_id'):
            batch_id = str(result['batch_id'])
            ctx.deps.ws.add_batch(run_id, batch_id)
            ctx.deps.submitted_batches.add(batch_id)
        elif name == 'list_batches' and isinstance(result, list):
            _adopt_batches(ctx.deps, result)
        elif name == 'batch_status' and isinstance(result, dict):
            batch_id = str(args.get('batch_id'))
            labels = result.get('labels')
            same_batch = str(result.get('batch_id')) == batch_id
            same_hypothesis = (isinstance(labels, dict)
                               and labels.get('hypothesis_id') == ctx.deps.ws.hypothesis_id)
            if same_batch and same_hypothesis and result.get('complete') is True and result.get('ok') is True:
                ctx.deps.executed_batches.add(batch_id)
        return result

def _adopt_batches(deps: Deps, batches: list) -> None:
    """Record remotely discovered receipts only when their labels identify one of our runs."""
    for batch in batches:
        if not isinstance(batch, dict) or not batch.get('batch_id'):
            continue
        labels = batch.get('labels')
        if not isinstance(labels, dict) or labels.get('hypothesis_id') != deps.ws.hypothesis_id:
            continue
        run_id = labels.get('run_id')
        if not isinstance(run_id, str):
            continue
        runs = {path.stem: path for path in deps.ws.records.joinpath('runs').glob('r*.json')}
        if run_id in runs:
            deps.ws.add_batch(run_id, str(batch['batch_id']))


def _remote_path(root: str, path: str, *, allow_root: bool = False) -> PurePosixPath:
    """Resolve a model-supplied remote path inside one owned workspace directory."""
    base = PurePosixPath(root)
    candidate = PurePosixPath(path)
    target = candidate if candidate.is_absolute() else base / candidate
    outside = target != base and base not in target.parents
    if (target == base and not allow_root) or outside or '..' in target.parts:
        raise ToolFailed(f'{path} is outside {base}')
    return target


# --- local mode: reads anywhere in the hypothesis workspace, writes only in the run dir ---

def _inside(root: str, path: str) -> Path:
    base = Path(root).resolve()
    target = (base / path).resolve()
    if not target.is_relative_to(base):
        raise ToolFailed(f'{path} is outside {base}')
    return target


def read_file(ctx: RunContext[Deps], path: str, max_chars: int = 20000) -> str:
    """Read a file; path relative to the hypothesis workspace (e.g. 'outputs/r001-01/result.json')."""
    return _inside(ctx.deps.ws.root, path).read_text(errors='replace')[:max_chars]


def list_dir(ctx: RunContext[Deps], path: str = '.') -> list[str]:
    """List files under a directory of the hypothesis workspace, recursively."""
    root = _inside(ctx.deps.ws.root, path)
    return sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file())


def write_files(ctx: RunContext[Deps], files: dict[str, str]) -> str:
    """Write several files at once into your run directory: {relative_path: contents}."""
    for path, text in files.items():
        target = _inside(ctx.deps.run_dir, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
    return f'wrote {len(files)} file(s) in {ctx.deps.run_dir}: {", ".join(files)}'


async def run_shell(ctx: RunContext[Deps], command: str) -> str:
    """Run a bash command with your run directory as cwd; a nonzero exit is a failed tool call."""
    proc = await asyncio.create_subprocess_exec(
        'bash', '-lc', command, cwd=ctx.deps.run_dir, start_new_session=True,
        stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        out, err = await proc.communicate()
    finally:
        if proc.returncode is None:  # timed out or cancelled: kill bash AND everything it started
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()
    result = (f'exit={proc.returncode}\nstdout:\n{out.decode(errors="replace")[-8000:]}\n'
              f'stderr:\n{err.decode(errors="replace")[-4000:]}')
    if proc.returncode:
        raise ToolFailed(result)
    ctx.deps.shell_executed = True
    return result


# --- shared -----------------------------------------------------------------

async def _validate_public_https(request: httpx.Request) -> None:
    """Reject non-HTTPS and destinations that resolve outside the public Internet."""
    parsed = urlsplit(str(request.url))
    if parsed.scheme != 'https' or not parsed.hostname:
        raise ToolFailed('fetch_url permits only HTTPS URLs with a hostname')
    try:
        addresses = await asyncio.get_running_loop().run_in_executor(
            None, lambda: socket.getaddrinfo(parsed.hostname, parsed.port or 443,
                                             type=socket.SOCK_STREAM))
    except socket.gaierror as exc:
        raise ToolFailed(f'cannot resolve {parsed.hostname!r}') from exc
    resolved = {item[4][0].split('%', 1)[0] for item in addresses}
    try:
        public = resolved and all(ipaddress.ip_address(address).is_global for address in resolved)
    except ValueError:
        public = False
    if not public:
        raise ToolFailed(f'fetch_url refused non-public destination {parsed.hostname!r}')


async def fetch_url(url: str, max_chars: int = 20000) -> str:
    """HTTPS GET a public URL and return the body as text."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=60, trust_env=False,
                                 event_hooks={'request': [_validate_public_https]}) as client:
        response = await client.get(url)
    return f'HTTP {response.status_code}\n{response.text[:max_chars]}'


SIDE_EFFECTS = (MCP_FULL - MCP_READ) | {'write_files', 'run_shell'}


async def prepare_execution_tools(ctx: RunContext[Deps], tools: list[ToolDefinition]) -> list[ToolDefinition]:
    """Run side-effecting tools one at a time within an agent turn, so a submission sees the files
    written just before it; hypotheses still run in parallel."""
    return [replace(tool, sequential=True) if tool.name in SIDE_EFFECTS else tool for tool in tools]


def execution_tools(cfg: Config, full: bool) -> AbstractToolset[Deps]:
    """Execution tools for one role: full (code-generator) or read-only (validator, reviewer).

    In mcp mode each agent run opens its own connection, so a dropped session fails only the
    calls of the run that held it, and the next run reconnects.
    """
    if cfg.execution.mode in {'mcp', 'hpc'}:
        from pydantic_ai.mcp import MCPToolset

        allowed = MCP_FULL if full else MCP_READ
        return OwnedMCP(DynamicToolset(lambda ctx: MCPToolset(
            cfg.execution.mcp_url, read_timeout=cfg.timeouts.tool
        ).filtered(lambda ctx, tool: tool.name in allowed), per_run_step=False)).prepared(prepare_execution_tools)
    return FunctionToolset([read_file, list_dir] + ([write_files, run_shell] if full else [])).prepared(prepare_execution_tools)


async def check_mcp_tools(cfg: Config) -> None:
    """Fail once at startup, not on every agent run, if the MCP server lacks a tool the roles use."""
    from pydantic_ai.mcp import MCPToolset

    async with MCPToolset(cfg.execution.mcp_url) as server:
        missing = MCP_FULL - {tool.name for tool in await server.list_tools()}
    if missing:
        raise RuntimeError(f'MCP server is missing required tools: {", ".join(sorted(missing))}')
