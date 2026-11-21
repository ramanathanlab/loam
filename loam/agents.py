"""The four LLM agents. Each gets its own model, its prompt, and only the tools its role allows."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

from pydantic_ai import Agent, FunctionToolset, RunContext, Tool
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.usage import UsageLimits

from . import execution, memory, skills
from .config import Config, Role
from .execution import Deps, Resilient
from .history import bounded_history
from .schemas import Brief, CodeResult, Review, ValidationReport

PROMPTS = Path(__file__).parent / 'prompts'


@dataclass
class Agents:
    cfg: Config
    validator: Agent[Deps, ValidationReport]
    code_generator: Agent[Deps, CodeResult]
    reviewer: Agent[Deps, Review]
    reasoner: Agent[Deps, Brief]

    async def run(self, role: Role, prompt: str, deps: Deps | None = None):
        """Run one agent under its role's timeout and request budget; returns its typed output."""
        result = await getattr(self, role).run(
            prompt, deps=deps, timeout=getattr(self.cfg.timeouts, role),
            usage_limits=UsageLimits(request_limit=self.cfg.max_requests))
        return result.output


def read_skill(ctx: RunContext[Deps], name: str, file: str | None = None) -> str:
    """Read a skill's SKILL.md, or one of its bundled files (e.g. 'scripts/run.py')."""
    return skills.read(ctx.deps.skills, name, file)


def build(cfg: Config) -> Agents:
    agents: Agents  # bound below; `delegate` reaches the code-generator through it

    def make(role: Role, output_type, *, tools=(), execution_tools=None, unwrapped=(),
             sees_skills=False, sees_execution=False):
        def campaign_context(ctx: RunContext[Deps | None]) -> str:
            parts = [f'CAMPAIGN GOAL:\n{cfg.goal_text()}']
            if cfg.rules:
                parts.append(f'CAMPAIGN RULES:\n{cfg.rules_text()}')
            if sees_skills and ctx.deps:
                parts.append(skills.catalog(ctx.deps.skills))
            if sees_execution and ctx.deps:
                ex = cfg.execution
                parts.append(f'EXECUTION: mode={ex.mode}\nHYPOTHESIS WORKSPACE (read): {ctx.deps.ws.root} '
                             '-- outputs/<run>/ holds every earlier run\'s results, data/ the campaign inputs.'
                             '\nPROVENANCE: read anything here, but a CLAIM may cite only a path inside a run '
                             'opened for this hypothesis, or a batch that run submitted. A directory you '
                             'created yourself, or another round\'s output, is discarded whatever it contains.'
                             + (f'\nSYSTEM (HPC_SYSTEM): {ex.hpc.system} -- when a skill\'s SKILL.md lists '
                                f'`systems:`, follow its references/{ex.hpc.system}.md; if it does not list '
                                f'{ex.hpc.system}, the skill is unsupported here.'
                                if ex.mode == 'hpc' and ex.hpc else '')
                             + (f'\nYOUR RUN DIRECTORY (write only here): {ctx.deps.run_dir}' if ctx.deps.run_dir else '')
                             + (f'\nENVIRONMENT SETUP:\n{ex.setup}' if ex.setup else ''))
            return '\n\n'.join(parts)

        # Every tool that touches the world is wrapped, so its failures reach the model, not the run.
        toolsets = [Resilient(t, timeout=cfg.timeouts.tool)
                    for t in [FunctionToolset(list(tools)), execution_tools] if t is not None]
        return Agent(cfg.models[role].build(cfg.rate_limits), name=role, output_type=output_type, deps_type=Deps,
                     instructions=[(PROMPTS / f'{role}.md').read_text(), campaign_context],
                     tools=unwrapped, toolsets=toolsets, retries=3,
                     capabilities=([ProcessHistory(lambda messages: bounded_history(messages, cfg.max_history_bytes))]
                                   if cfg.max_history_bytes is not None else []),
                     model_settings={'max_retries': cfg.model_retries,
                                     'request_timeout': cfg.timeouts.model_request})

    async def delegate(ctx: RunContext[Deps], instruction: str) -> str:
        """Delegate one action to a code-generator agent, which writes and RUNS the code.

        The instruction must be complete: question, inputs/identifiers, designed parameters,
        expected outputs, acceptance gate, and `skill: <name>` for each step.
        """
        if ctx.deps.code_calls >= cfg.max_code_calls:
            return 'Code-generator budget exhausted. Poll pending batches yourself, then conclude.'
        ctx.deps.code_calls += 1
        run_id, run_dir = ctx.deps.ws.open_run(ctx.deps.round, instruction)
        prompt = (f'HYPOTHESIS UNDER TEST:\n{ctx.deps.hypothesis}\n\nRUN {run_id}. INSTRUCTION:\n{instruction}'
                  + await memory.recall_block(ctx.deps.memory, instruction, kinds=['procedural']))
        child_deps = replace(ctx.deps, run_dir=run_dir, shell_executed=False,
                             submitted_batches=set(), executed_batches=set())
        try:
            result = await agents.run('code_generator', prompt, child_deps)
        except Exception as exc:  # crashed, timed out, or out of requests: reported to the validator, not fatal
            ctx.deps.ws.close_run(run_id, None, error=repr(exc))
            ctx.deps.code_results.append({'run_id': run_id, 'instruction': instruction,
                                          'status': 'crashed', 'error': repr(exc)})
            return (f'code_generator for run {run_id} ended without a result: {exc!r}. '
                    f'Anything it wrote is under outputs/{run_id}/.')
        if cfg.execution.mode in {'mcp', 'hpc'}:
            observed = ('executed' if child_deps.executed_batches else
                        'submitted' if child_deps.submitted_batches else 'blocked')
        else:
            observed = 'executed' if child_deps.shell_executed else 'blocked'
        result = result.model_copy(update={'status': observed})
        ctx.deps.ws.close_run(run_id, result)
        ctx.deps.code_results.append({'run_id': run_id, 'instruction': instruction, **result.model_dump()})
        return json.dumps({'run_id': run_id, 'outputs': f'outputs/{run_id}/', **result.model_dump()}, indent=1)

    agents = Agents(
        cfg=cfg,
        # `delegate` is not wrapped by Resilient: it is bounded by the code-generator's own timeout.
        validator=make('validator', ValidationReport, tools=[read_skill, execution.fetch_url],
                       execution_tools=execution.execution_tools(cfg, full=False),
                       unwrapped=[Tool(delegate, name='code_generator')], sees_skills=True, sees_execution=True),
        code_generator=make('code_generator', CodeResult, tools=[read_skill, execution.fetch_url],
                            execution_tools=execution.execution_tools(cfg, full=True),
                            sees_skills=True, sees_execution=True),
        reviewer=make('reviewer', Review, tools=[read_skill, execution.fetch_url],
                      execution_tools=execution.execution_tools(cfg, full=False)),
        reasoner=make('reasoner', Brief),
    )
    return agents
