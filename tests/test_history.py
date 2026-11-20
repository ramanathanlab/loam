"""Conversation history: agents keep the task and recent tool exchanges within a byte budget.

Ported from the moeen branch's tests/test_history.py.
"""

import asyncio
from pathlib import Path

from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import (
    InstructionBaselinePart,
    InstructionPart,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from loam.agents import build
from loam.execution import Deps, prepare_execution_tools
from loam.history import bounded_history, model_visible_size
from loam.workspace import Workspace
from tests.test_round import H, make_cfg


def exchange(index, length):
    key = f'call-{index}'
    return [ModelResponse(parts=[ToolCallPart('inspect', {'index': index}, key)]),
            ModelRequest(parts=[ToolReturnPart('inspect', str(index) * length, key)])]


def tool_ids(messages):
    calls = {p.tool_call_id for m in messages if isinstance(m, ModelResponse)
             for p in m.parts if isinstance(p, ToolCallPart)}
    returns = {p.tool_call_id for m in messages if isinstance(m, ModelRequest)
               for p in m.parts if isinstance(p, ToolReturnPart)}
    return calls, returns


def test_history_keeps_initial_task_and_complete_latest_tool_exchanges():
    messages = [ModelRequest(parts=[UserPromptPart('Initial hypothesis and gate')])]
    messages += exchange(1, 14000) + exchange(2, 14000) + exchange(3, 2000)
    original = ModelMessagesTypeAdapter.dump_json(messages)
    result = bounded_history(messages, 9000)
    assert len(ModelMessagesTypeAdapter.dump_json(result)) <= 9000
    assert result[0].parts[0].content == 'Initial hypothesis and gate'
    calls, returns = tool_ids(result)
    assert calls == returns == {'call-3'}
    assert ModelMessagesTypeAdapter.dump_json(messages) == original


def test_oversized_latest_tool_output_is_explicitly_shortened():
    messages = [ModelRequest(parts=[UserPromptPart('Measure real files')]), *exchange(1, 100000)]
    result = bounded_history(messages, 9000)
    assert len(ModelMessagesTypeAdapter.dump_json(result)) <= 9000
    assert 'SHORTENED' in result[-1].parts[0].content
    assert len(messages[-1].parts[0].content) == 100000


def test_real_agent_loop_accepts_compacted_tool_protocol():
    requests = 0
    saw_shortened = False

    def model(messages, info):
        nonlocal requests, saw_shortened
        calls, _ = tool_ids(messages)
        for m in messages:
            if isinstance(m, ModelRequest):
                for p in m.parts:
                    if isinstance(p, ToolReturnPart):
                        assert p.tool_call_id in calls
                        saw_shortened |= 'SHORTENED' in str(p.content)
        requests += 1
        return ModelResponse(parts=[ToolCallPart('inspect', {})] if requests < 5 else [TextPart('done')])

    def inspect():
        return 'actual verbose tool output ' * 5000

    agent = Agent(FunctionModel(model), tools=[inspect],
                  capabilities=[ProcessHistory(lambda messages: bounded_history(messages, 9000))])
    result = asyncio.run(agent.run('Keep the initial task and tool-call associations'))
    assert result.output == 'done' and requests == 5 and saw_shortened


def test_sdk_instruction_snapshots_do_not_evict_useful_tool_history():
    instructions = 'Keep the real hypothesis, tool contracts and scientific gates. ' * 400
    messages = [ModelRequest(parts=[UserPromptPart('Validate the original hypothesis')], instructions=instructions)]
    for index in range(12):
        messages.extend(exchange(index, 500))
        messages[-1].instructions = instructions
        messages[-1].parts = [*messages[-1].parts, InstructionBaselinePart(instructions=[InstructionPart(content=instructions)])]
    assert len(ModelMessagesTypeAdapter.dump_json(messages)) > 500000
    assert model_visible_size(messages) < 96000
    assert bounded_history(messages, 96000) is messages


def test_dependent_file_creation_and_submission_in_one_response_are_sequential(tmp_path):
    calls = 0
    destination = tmp_path / 'inputs.jsonl'

    async def write_files():
        await asyncio.sleep(.02)
        destination.write_text('real candidate inputs')
        return 'written'

    async def run_batch():
        assert destination.read_text() == 'real candidate inputs'
        return 'submitted after write'

    def model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[ToolCallPart('write_files', {}), ToolCallPart('run_batch', {})]
                             if calls == 1 else [TextPart('done')])

    tools = FunctionToolset([write_files, run_batch]).prepared(prepare_execution_tools)
    assert asyncio.run(Agent(FunctionModel(model), toolsets=[tools]).run('Write then submit')).output == 'done'


def test_campaign_agents_see_bounded_history_when_configured(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.max_history_bytes = 40000
    ws = Workspace(cfg, H)
    (Path(ws.root) / 'big.txt').write_text('x' * 15000)
    sizes, notes = [], 0

    def respond(messages, info: AgentInfo):
        nonlocal notes
        sizes.append(model_visible_size(messages))
        notes += any('CONTEXT WINDOW NOTE' in str(getattr(p, 'content', ''))
                     for m in messages if isinstance(m, ModelRequest) for p in m.parts)
        if len(sizes) < 6:
            return ModelResponse(parts=[ToolCallPart('read_file', {'path': 'big.txt'})])
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, {'verdicts': []})])

    agents = build(cfg)
    with agents.reviewer.override(model=FunctionModel(respond)):
        asyncio.run(agents.run('reviewer', 'Review the claims.', Deps(cfg, {}, ws, 1, 'h')))
    assert max(sizes) <= 40000 and notes
