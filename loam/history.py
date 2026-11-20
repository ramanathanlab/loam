"""Bound model context without changing on-disk execution/evidence records."""
from dataclasses import replace
import json
import logging

from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, ModelResponse, ToolReturnPart, UserPromptPart


def model_visible_size(messages):
    """Count current instructions once, without SDK-only history snapshots.

    ModelRequest.instructions is retained on *every* request for tracing,
    but the OpenAI adapter sends only the current instructions. Instruction
    baselines are SDK bookkeeping and are never sent to the model.
    """
    current=next((m.instructions for m in reversed(messages)
                  if isinstance(m,ModelRequest) and m.instructions is not None),'')
    visible=[replace(m,instructions=None,metadata=None,
                     parts=[p for p in m.parts if getattr(p,'part_kind',None)!='instruction-baseline'])
             if isinstance(m,ModelRequest) else m for m in messages]
    return len(ModelMessagesTypeAdapter.dump_json(visible))+len(current.encode())


def bounded_history(messages, max_bytes=96000):
    """Keep the initial task and a suffix of whole model/tool exchanges.

    A suffix starts at an assistant response, so retained tool results still
    have their corresponding calls. Large tool returns are explicitly marked
    as shortened; agents must reread compact artifacts rather than infer data.
    Original scripts, receipts, result files and claim ledgers are untouched.
    """
    size = model_visible_size
    before = size(messages)
    def finish(history):
        logging.getLogger('loam.history').info('Bounded model-visible history from %d to %d bytes; execution records unchanged', before, size(history))
        return history
    if before <= max_bytes:
        return messages
    starts = [i for i, m in enumerate(messages) if isinstance(m, ModelResponse)]
    if not starts:
        return messages
    prefix = list(messages[:starts[0]])
    note = UserPromptPart('CONTEXT WINDOW NOTE: Older exchanges were removed from this prompt. '
                          'Their files and owned batch receipts remain on disk. Use list_batches and '
                          'read compact summaries to recover facts; do not repeat completed science '
                          'or infer results from shortened tool output.')
    if prefix and isinstance(prefix[-1], ModelRequest) and not any(
            isinstance(p, UserPromptPart) and isinstance(p.content, str) and p.content.startswith('CONTEXT WINDOW NOTE:')
            for p in prefix[-1].parts):
        prefix[-1] = replace(prefix[-1], parts=[*prefix[-1].parts, note])
    chosen = prefix + list(messages[starts[-1]:])
    for start in starts:
        candidate = prefix + list(messages[start:])
        if size(candidate) <= max_bytes:
            return finish(candidate)
    # A single verbose response can exceed the whole budget. Keep call IDs
    # and explicit truncation notices, not an apparently complete JSON value.
    for cap in (12000, 4000, 1000, 200):
        shortened = []
        for message in chosen:
            if not isinstance(message, ModelRequest):
                shortened.append(message)
                continue
            parts = []
            for part in message.parts:
                if isinstance(part, ToolReturnPart):
                    content = part.content if isinstance(part.content, str) else json.dumps(part.content, default=str)
                    if len(content) > cap:
                        part = replace(part, content='[TOOL OUTPUT SHORTENED FOR CONTEXT; reread the owned compact artifact]\n' + content[:cap])
                parts.append(part)
            shortened.append(replace(message, parts=parts))
        if size(shortened) <= max_bytes:
            return finish(shortened)
    # A huge original task or tool-call arguments cannot be silently rewritten.
    raise ValueError('The initial task and latest tool exchange exceed the configured history budget; use a fresh, narrower delegation')
