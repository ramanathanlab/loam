import asyncio

from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel

from loam.history import bounded_history


def exchange(index, length):
    key = f'call-{index}'
    return [ModelResponse(parts=[ToolCallPart('inspect', {'index': index}, key)]),
            ModelRequest(parts=[ToolReturnPart('inspect', str(index) * length, key)])]


def test_history_keeps_initial_task_and_complete_latest_tool_exchanges():
    messages = [ModelRequest(parts=[UserPromptPart('Initial hypothesis and gate')])]
    messages += exchange(1, 14000) + exchange(2, 14000) + exchange(3, 2000)
    original = ModelMessagesTypeAdapter.dump_json(messages)
    result = bounded_history(messages, 9000)
    assert len(ModelMessagesTypeAdapter.dump_json(result)) <= 9000
    assert result[0].parts[0].content == 'Initial hypothesis and gate'
    calls = {p.tool_call_id for m in result if isinstance(m, ModelResponse) for p in m.parts if isinstance(p, ToolCallPart)}
    returns = {p.tool_call_id for m in result if isinstance(m, ModelRequest) for p in m.parts if isinstance(p, ToolReturnPart)}
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
        calls = {p.tool_call_id for m in messages if isinstance(m, ModelResponse) for p in m.parts if isinstance(p, ToolCallPart)}
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
    from pydantic_ai.messages import InstructionBaselinePart, InstructionPart
    from loam.history import model_visible_size
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
    from pydantic_ai import FunctionToolset
    from loam.execution import prepare_execution_tools
    calls=0
    destination=tmp_path/'inputs.jsonl'
    async def write_files():
        await asyncio.sleep(.02)
        destination.write_text('real candidate inputs')
        return 'written'
    async def run_batch():
        assert destination.read_text()=='real candidate inputs'
        return 'submitted after write'
    def model(messages,info):
        nonlocal calls
        calls+=1
        return ModelResponse(parts=[ToolCallPart('write_files',{}),ToolCallPart('run_batch',{})]
                             if calls==1 else [TextPart('done')])
    tools=FunctionToolset([write_files,run_batch]).prepared(prepare_execution_tools)
    assert asyncio.run(Agent(FunctionModel(model),toolsets=[tools]).run('Write then submit')).output=='done'
