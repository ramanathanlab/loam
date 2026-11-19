"""Campaign configuration: one YAML file -> one validated Config."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal['validator', 'code_generator', 'reviewer', 'reasoner']


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ModelSpec(ConfigModel):
    """One agent's model. With base_url: any OpenAI-compatible endpoint.
    Without: a pydantic-ai model string such as 'anthropic:claude-opus-5-5' or 'test'."""

    model: str
    base_url: str | None = None
    api_key_env: str | None = None

    def api_key(self, *, custom_endpoint: bool = False) -> str | None:
        key = os.environ.get(self.api_key_env) if self.api_key_env else None
        return key or ('none' if custom_endpoint else None)

    def build(self):
        if not self.base_url:
            return self.model
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider

        return OpenAIChatModel(
            self.model,
            provider=OpenAIProvider(base_url=self.base_url, api_key=self.api_key(custom_endpoint=True)),
        )

    def jnana(self, max_tokens: int = 32768):
        """The same endpoint as a Jnana LLMConfig (import deferred: Jnana is on jnana_root)."""
        from jnana.protognosis.core.multi_llm_config import LLMConfig

        provider, _, name = self.model.rpartition(':') if not self.base_url else ('openai', '', self.model)
        return LLMConfig(provider=provider or 'openai', model=name,
                         api_key=self.api_key(custom_endpoint=self.base_url is not None),
                         base_url=self.base_url, max_tokens=max_tokens,
                         model_adapter={'json_max_tokens': max_tokens})

PositiveInt = Annotated[int, Field(gt=0)]
PositiveFloat = Annotated[float, Field(gt=0)]
NonEmptyStr = Annotated[str, Field(min_length=1)]


class HPCAllocation(ConfigModel):
    account: NonEmptyStr
    queue: NonEmptyStr
    walltime: str
    nodes: PositiveInt
    init_blocks: Annotated[int, Field(ge=0)]
    min_blocks: Annotated[int, Field(ge=0)]
    max_blocks: PositiveInt
    max_idletime: PositiveInt
    endpoint_options: dict[str, Any] = {}

    @model_validator(mode='after')
    def validate_allocation(self):
        if not re.fullmatch(r'\d{2}:[0-5]\d:[0-5]\d', self.walltime):
            raise ValueError('execution.hpc.allocation.walltime must have format HH:MM:SS')
        if not self.min_blocks <= self.init_blocks <= self.max_blocks:
            raise ValueError('execution.hpc allocation blocks must satisfy min_blocks <= init_blocks <= max_blocks')
        reserved = {
            'account', 'queue', 'walltime', 'nodes', 'nodes_per_block', 'init_blocks',
            'min_blocks', 'max_blocks', 'max_idletime', 'available_accelerators',
            'max_workers_per_node', 'base_dir',
        }
        overlap = reserved.intersection(self.endpoint_options)
        if overlap:
            raise ValueError(f'execution.hpc.allocation.endpoint_options cannot override typed fields: {", ".join(sorted(overlap))}')
        return self


class HPCDeployment(ConfigModel):
    endpoint_id: NonEmptyStr
    campaign_id: NonEmptyStr
    system: Literal['aurora', 'polaris']
    exchange_url: NonEmptyStr
    base_dir: str
    worker_code_dir: str
    inspect_roots: list[str]
    pool_size: PositiveInt
    server_script: Path
    host: NonEmptyStr
    port: Annotated[int, Field(ge=1, le=65535)]
    startup_timeout: PositiveFloat
    # A multi-user endpoint forks the per-config user endpoint on demand, and
    # the first submit against a cold one is refused with 422
    # ENDPOINT_NOT_ONLINE (which the Globus SDK does not retry). The server
    # retries just that case for cold_endpoint_timeout seconds, starting at
    # cold_endpoint_backoff and doubling. These bound uep startup only -- the
    # PBS queue is waited out separately by the cluster agent's first probe.
    # Float literals, not ints: pydantic does not validate defaults, so an int
    # default would render as '300' while a YAML-supplied 300 renders '300.0'.
    cold_endpoint_timeout: PositiveFloat = 300.0
    cold_endpoint_backoff: PositiveFloat = 5.0
    allocation: HPCAllocation

    @model_validator(mode='after')
    def validate_deployment(self):
        for name in ('base_dir', 'worker_code_dir'):
            _validate_remote_path(name, getattr(self, name))
        for root in self.inspect_roots:
            _validate_remote_path('inspect_roots', root)
        exchange = urlsplit(self.exchange_url)
        if exchange.scheme not in {'http', 'https'} or not exchange.netloc:
            raise ValueError('execution.hpc.exchange_url must be an HTTP(S) URL')
        return self

    def environment(self) -> dict[str, str]:
        accelerators = 12 if self.system == 'aurora' else 4
        allocation = self.allocation
        endpoint_config = {
            **allocation.endpoint_options,
            'account': allocation.account,
            'queue': allocation.queue,
            'walltime': allocation.walltime,
            'nodes_per_block': allocation.nodes,
            'init_blocks': allocation.init_blocks,
            'min_blocks': allocation.min_blocks,
            'max_blocks': allocation.max_blocks,
            'max_idletime': allocation.max_idletime,
            'available_accelerators': accelerators,
            'max_workers_per_node': accelerators,
            'base_dir': self.worker_code_dir,
        }
        return {
            'HPC_GLOBUS_ENDPOINT_ID': self.endpoint_id,
            'HPC_CAMPAIGN_ID': self.campaign_id,
            'HPC_SYSTEM': self.system,
            'ACADEMY_EXCHANGE_URL': self.exchange_url,
            'HPC_BASE_DIR': self.base_dir,
            'HPC_LOAM_HOME': self.worker_code_dir,
            'HPC_INSPECT_ROOTS': ':'.join(self.inspect_roots),
            'HPC_POOL_SIZE': str(self.pool_size),
            'HPC_COLD_ENDPOINT_TIMEOUT': str(self.cold_endpoint_timeout),
            'HPC_COLD_ENDPOINT_BACKOFF': str(self.cold_endpoint_backoff),
            'HPC_ENDPOINT_CONFIG_JSON': json.dumps(endpoint_config, sort_keys=True, separators=(',', ':')),
            'HPC_NNODES': str(allocation.nodes),
            'HPC_MCP_TRANSPORT': 'http',
            'HPC_MCP_HOST': self.host,
            'HPC_MCP_PORT': str(self.port),
        }


def _validate_remote_path(name: str, value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if '\\' in value or not path.is_absolute() or '..' in path.parts:
        raise ValueError(f'execution.hpc.{name} must be an absolute POSIX path without ..')
    return path


class Execution(ConfigModel):
    mode: Literal['local', 'mcp', 'hpc'] = 'local'
    mcp_url: str | None = None
    workdir: str = 'work'
    setup: str = ''
    hpc: HPCDeployment | None = None

    @model_validator(mode='after')
    def validate_mode(self):
        if self.mode == 'mcp':
            url = urlsplit(self.mcp_url or '')
            if url.scheme not in {'http', 'https'} or not url.netloc:
                raise ValueError('execution.mcp_url must be an HTTP(S) URL in MCP mode')
            if self.hpc is not None:
                raise ValueError('execution.hpc is only valid in HPC mode')
            _validate_remote_path('workdir', self.workdir)
        elif self.mode == 'hpc':
            if self.mcp_url is not None:
                raise ValueError('execution.mcp_url is derived and must not be set in HPC mode')
            if self.hpc is None:
                raise ValueError('execution.hpc is required in HPC mode')
            workdir = _validate_remote_path('workdir', self.workdir)
            base_dir = PurePosixPath(self.hpc.base_dir)
            if workdir != base_dir and base_dir not in workdir.parents:
                raise ValueError('execution.workdir must be under execution.hpc.base_dir')
        elif self.hpc is not None or self.mcp_url is not None:
            raise ValueError('execution.hpc and execution.mcp_url are not valid in local mode')
        return self


class Timeouts(ConfigModel):
    """Seconds. A tool or model request that runs out is reported to the agent; an agent run that runs
    out ends that one thread (or code-generator call), never the round."""

    tool: PositiveFloat = 1800              # one tool call: a shell command, an MCP call, a fetch
    model_request: PositiveFloat = 600      # one LLM request, including the SDK's own retries
    validator: PositiveFloat = 4 * 3600     # one validator run, including the code-generators it delegates to
    code_generator: PositiveFloat = 3600
    reviewer: PositiveFloat = 1800
    reasoner: PositiveFloat = 1800


class TournamentConfig(ConfigModel):
    jnana_root: Path
    generation: ModelSpec
    ranking: ModelSpec
    initial_hypotheses: PositiveInt = 6
    new_per_round: PositiveInt = 3       # generated each later round from the reasoner's brief
    gen_batch_size: PositiveInt = 1      # hypotheses requested PER LLM call (Jnana batch_size);
                                         # >1 means ceil(count/gen_batch_size) larger calls, fewer in flight
    top_k: PositiveInt = 3               # hypotheses validated per round
    # Pairing scheme. 'swiss' (the default) plays `swiss_rounds` sequential
    # rounds, re-pairing on the current Elo between them. A Swiss round's
    # pairings are DISJOINT -- every hypothesis plays exactly once per round --
    # so the whole pool gets ranked and the round's matches may run in parallel.
    # 'random' plays a single batch of `matches` pairs that may overlap, so some
    # hypotheses can end up barely ranked and the matches must stay serial.
    scheme: Literal['swiss', 'random'] = 'swiss'
    swiss_rounds: PositiveInt = 3        # swiss only (Jnana's `rounds`); "deep Swiss" is just more
    workers: PositiveInt = 1             # matches run in parallel within one swiss round
    matches: PositiveInt = 10            # RANDOM SCHEME ONLY; Jnana ignores it when scheme is swiss
    max_seconds: PositiveFloat = 600

    @model_validator(mode='after')
    def validate_tournament(self):
        # Mirrors Jnana's own check, so an invalid combination fails at config
        # load instead of part-way through a round that has already spent
        # allocation: random pairs can share a hypothesis, so parallel matches
        # would race on its Elo (read-modify-write).
        if self.workers > 1 and self.scheme != 'swiss':
            raise ValueError('tournament.workers > 1 is only supported by the swiss scheme')
        return self


class Config(ConfigModel):
    goal: str                    # path to a markdown goal file, or the goal text itself
    rules: list[Path] = []       # optional campaign rule files, appended to every agent's instructions
    state_dir: Path = Path('state')
    skills_dirs: list[Path] = []
    models: dict[Role, ModelSpec]
    execution: Execution = Execution()
    tournament: TournamentConfig
    rounds: PositiveInt = 1
    workspace_inputs: Path | None = None   # local dir copied into every hypothesis's data/
    concurrency: PositiveInt = 4         # validators running at once
    max_code_calls: PositiveInt = 4      # code-generator delegations per validator
    max_requests: PositiveInt = 200      # LLM requests per agent run (pydantic-ai's default of 50 is too low)
    model_retries: PositiveInt = 8       # SDK retries per LLM request on 429/5xx/connection errors, with backoff
    timeouts: Timeouts = Timeouts()

    @model_validator(mode='after')
    def validate_campaign(self):
        if self.execution.mode != 'local' and self.workspace_inputs is not None:
            raise ValueError('workspace_inputs is not supported in MCP or HPC mode')
        required = {'validator', 'code_generator', 'reviewer', 'reasoner'}
        if set(self.models) != required:
            missing = required - set(self.models)
            raise ValueError(f'models must define exactly all four roles; missing: {", ".join(sorted(missing))}')
        return self

    def goal_text(self) -> str:
        path = Path(self.goal)
        return path.read_text() if path.suffix == '.md' and path.is_file() else self.goal

    def rules_text(self) -> str:
        return '\n\n'.join(p.read_text() for p in self.rules)


def load(path: str | Path) -> Config:
    """Load a YAML config; relative paths resolve against the config file's directory."""
    path = Path(path).resolve()
    cfg = Config.model_validate(yaml.safe_load(path.read_text()))
    here = path.parent

    def fix(p: Path) -> Path:
        p = p.expanduser()
        return (p if p.is_absolute() else here / p).resolve()

    cfg.state_dir = fix(cfg.state_dir)
    cfg.skills_dirs = [fix(p) for p in cfg.skills_dirs]
    cfg.rules = [fix(p) for p in cfg.rules]
    cfg.workspace_inputs = cfg.workspace_inputs and fix(cfg.workspace_inputs)
    cfg.tournament.jnana_root = fix(cfg.tournament.jnana_root)
    if cfg.goal.endswith('.md'):
        cfg.goal = str(fix(Path(cfg.goal)))
    if cfg.execution.mode == 'local':
        cfg.execution.workdir = str(fix(Path(cfg.execution.workdir)))
    if cfg.execution.mode == 'hpc':
        assert cfg.execution.hpc is not None
        cfg.execution.hpc.server_script = fix(cfg.execution.hpc.server_script)
        if not cfg.execution.hpc.server_script.is_file():
            raise ValueError(f'execution.hpc.server_script does not exist: {cfg.execution.hpc.server_script}')
    return cfg
