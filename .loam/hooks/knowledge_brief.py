"""Ground Jnana's first hypothesis generation in an MCP knowledge graph (OKN by default).

Before Jnana generates from the research goal, a bounded tool-calling loop on
the tournament's own model queries the knowledge-graph MCP server and writes a
short brief: relevant graphs, entities with identifiers, facts with the graph
they came from, and gaps. The tournament appends the brief to Jnana's goal.

Advisory like the tournament itself: any failure returns no brief and Jnana
generates from the plain goal. Briefs are cached per (goal, server, model).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping

DEFAULT_URL = "https://apps.okn.us/okn-mcp-dev/mcp"
# The research subset of OKN's tools; transcript, diagram and spatial tools
# only cost context here.
DEFAULT_TOOLS = (
    "list_kgs", "describe_kg", "get_schema", "probe_namespaces", "find_crosswalks",
    "get_join_strategy", "list_crosswalks", "find_context_sources", "sparql_query",
    "expand_ontology_term", "get_kg_version",
)
CACHE_FILE = "knowledge-brief.json"
HEADER = ("Background gathered automatically from the {name} knowledge graphs before generation. "
          "Facts carry [tool: graph] provenance; it is incomplete, and anything marked "
          "[background, unverified] or lacking provenance was NOT confirmed against a graph and "
          "must not be treated as established. Use it to ground mechanisms and identifiers, and "
          "treat an unsourced identifier as something to verify rather than to build on:")
_RESULT_CHARS = 20000
_CALL_TIMEOUT_S = 90
# A plain "...[truncated]" tail reads as "scroll for more" and the model repeats
# the same call until its budget is gone; say outright that there is no more.
_TRUNCATED = ("\n...[result truncated here; there is no way to fetch the rest and repeating this "
              "call returns the same thing. Work from what is above.]")

SYSTEM = """You ground a hypothesis-generation step in knowledge graphs reached through the tools.
Research the goal below with the tools, then write the brief. Follow the server's workflow:
pick relevant graphs with list_kgs, call get_schema before writing SPARQL, keep queries small
(LIMIT), and stop querying once you have enough to ground hypotheses.

QUERY COST, learned from real runs against this server. Queries are cut off after
120 s, and a timeout costs you part of your budget and returns nothing:
- NEVER scan all labels. `?s ?p ?label . FILTER(CONTAINS(LCASE(STR(?label)), "x"))`,
  and STRSTARTS/regex over unbounded labels, always time out here, even with LIMIT
  and even scoped to one graph. Searching by name this way does not work; it is the
  main way these runs fail, and it looks like "the graph has no data on X" when the
  data is there under an id you did not look up.
- Enter by IDENTIFIER instead. Name the exact terms in a `VALUES` block
  (`VALUES ?t { obo:MONDO_0020499 obo:NCBITaxon_3052225 } ?t rdfs:label ?l .`) and
  they return in under a second; so do `expand_ontology_term` on a CURIE and edge
  queries out from a known IRI. If you do not know an id, derive it from
  `probe_namespaces`/`find_crosswalks`/`get_schema` rather than scanning for it,
  and if you cannot, say so in section 4 instead of burning the budget.
- Declare every PREFIX you use (rdfs, obo, biolink, up, skos…). An undeclared
  prefix is an HTTP 400, a wasted call.
- NEVER invent a numeric CURIE. Guessing `MONDO_00…`/`NCBITaxon_…` digits for a
  name you know returns an unrelated term (a real run guessed three and got
  "obsolete dracunculiasis"), and an id you did not look up is also exactly what
  the evidence rule below forbids. Use the ids given to you below, ids returned by
  a tool, or none at all.
- A TIMEOUT IS NOT AN ABSENCE, and never a finding. If a query times out, you
  learned nothing about the data: do not write that the term, edge or hierarchy is
  missing, do not call it a gap or a contradiction, and never contrast it with
  something another query did return (a real run reported "Hendra is not under its
  parent, Nipah is" when both queries simply timed out). Say "query timed out, not
  determined". Only an EXECUTED query that returned zero rows is evidence of
  absence, and say which query it was.
- Hierarchy traversal is expensive here: `subClassOf*` and `expand_ontology_term`
  on a large ontology such as MONDO often time out even for one subject with a
  LIMIT. Prefer asking for the specific terms you care about in a VALUES block
  over walking the tree.
- Each graph is a different subject area. A graph that lacks one layer of the
  question (say molecular interactions) may still hold another (disease terms,
  phenotypes, taxonomy); report the layer you DID find rather than nothing.

Then reply WITHOUT tool calls with a brief of at most ~500 words:
1. Relevant graphs and why.
2. Key entities with identifiers (e.g. NCBI Gene, MONDO, CHEBI ids).
3. Concrete facts or relationships the queries returned, each with its graph and counts.
4. Gaps, contradictions or sparse areas worth forming hypotheses about.

EVIDENCE RULE, the point of this step. Every identifier, name, number and
relationship in sections 2 and 3 must be COPIED from a tool result in this
conversation. Do not supply an id, taxon, gene, accession or edge from your own
knowledge, not even one you are confident about, and do not reconstruct one you
expect the graph to hold: a wrong id presented as graph-derived is worse than no
id. Write each such line as `<fact> [<tool>: <graph>]` so it can be traced. When
you looked and the graph did not have it, say "not found in <graph>" and put it
in section 4; when you know something from background knowledge and want to note
it, mark it `[background, unverified]`. A short brief of traceable facts is the
goal; a long one padded with recalled detail is a failure.

Server notes:
{instructions}"""


def _key(goal: str, url: str, model: str, seeds: tuple[str, ...] = ()) -> str:
    return hashlib.sha256(json.dumps([goal, url, model, list(seeds)]).encode()).hexdigest()[:16]


def _describe(exc: BaseException) -> str:
    """Flatten the cause chain: anyio nests real errors inside ExceptionGroups."""

    parts, seen = [], set()

    def walk(error: BaseException) -> None:
        if id(error) in seen:
            return
        seen.add(id(error))
        nested = list(getattr(error, "exceptions", None) or [])
        if nested:
            for item in nested:
                walk(item)
            return
        parts.append(f"{type(error).__name__}: {error}".strip().rstrip(":"))
        if error.__cause__ is not None:
            walk(error.__cause__)

    walk(exc)
    return "; ".join(parts)[:300] or type(exc).__name__


def _read_cache(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _tool_text(result: Any) -> str:
    parts = []
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        parts.append(text if isinstance(text, str) else str(item))
    text = "\n".join(parts) or json.dumps(getattr(result, "structuredContent", None) or {})
    if getattr(result, "isError", False):
        text = "TOOL ERROR: " + text
    return text if len(text) <= _RESULT_CHARS else text[:_RESULT_CHARS] + _TRUNCATED


def _openai_tools(tools: list, allowed: set[str]) -> list[dict]:
    return [
        {"type": "function", "function": {
            "name": tool.name,
            "description": (tool.description or "")[:1500],
            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
        }}
        for tool in tools if not allowed or tool.name in allowed
    ]


async def _research(goal: str, url: str, allowed: set[str], llm: Mapping[str, Any],
                    max_calls: int, seeds: tuple[str, ...] = ()) -> tuple[str, int, list[str]]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=llm["base_url"], api_key=llm["api_key"] or "local-no-auth")
    async with streamablehttp_client(url) as (read, write, _), ClientSession(read, write) as session:
        init = await session.initialize()
        tools = _openai_tools((await session.list_tools()).tools, allowed)
        task = f"Research goal:\n{goal}"
        if seeds:
            # Entry points the operator supplies: the server has no working
            # name->id search, so without these the model has to guess.
            task += ("\n\nVerified identifiers to start from (already checked; use them in VALUES "
                     "blocks and expand outward):\n" + "\n".join(f"- {s}" for s in seeds))
        messages: list[dict] = [
            # replace(), not format(): the prompt contains literal SPARQL braces.
            {"role": "system", "content": SYSTEM.replace("{instructions}", (init.instructions or "")[:4000])},
            {"role": "user", "content": task},
        ]
        calls, used, seen, nudged = 0, [], set(), False
        while True:
            budget_left = calls < max_calls
            response = await client.chat.completions.create(
                model=llm["model"], messages=messages, max_tokens=3000,
                **({"tools": tools, "tool_choice": "auto"} if budget_left else {}))
            message = response.choices[0].message
            if not budget_left or not message.tool_calls:
                text = (message.content or "").strip()
                # A reasoning model can spend the whole budget thinking and
                # answer with empty content; ask once, plainly, for the text.
                if not text and not nudged:
                    nudged = True
                    messages.append({"role": "user", "content": (
                        "Your last reply had no text. Write the brief now as plain prose, no tool calls.")})
                    continue
                return text, calls, used
            messages.append(message.model_dump(exclude_none=True))
            for call in message.tool_calls:
                calls += 1
                used.append(call.function.name)
                signature = (call.function.name, call.function.arguments or "{}")
                if signature in seen:
                    # Repeating a call verbatim is a loop, not progress.
                    text = (f"Already called {call.function.name} with these arguments; the result is "
                            "earlier in this conversation and has not changed. Use it, or call a "
                            "different tool, or write the brief.")
                else:
                    try:
                        arguments = json.loads(call.function.arguments or "{}")
                        result = await session.call_tool(call.function.name, arguments,
                                                         read_timeout_seconds=timedelta(seconds=_CALL_TIMEOUT_S))
                        text = _tool_text(result)
                    except Exception as exc:  # noqa: BLE001 - report to the model and keep going
                        text = f"TOOL ERROR: {type(exc).__name__}: {exc}"
                    seen.add(signature)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
            if calls >= max_calls:
                messages.append({"role": "user", "content": "Tool budget used. Write the brief now, without tool calls."})


def brief_for(goal: str, knowledge: Mapping[str, Any], llm: Mapping[str, Any],
              cache_dir: Path) -> dict[str, Any]:
    """Return {status, brief, tool_calls, tools_used, cached, seconds}; brief is "" unless status is ok."""

    url = str(knowledge.get("mcp_url") or DEFAULT_URL)
    name = str(knowledge.get("name") or "OKN")
    tools = knowledge.get("tools")
    allowed = {str(t) for t in tools} if isinstance(tools, (list, tuple)) else set(DEFAULT_TOOLS)
    max_calls = max(1, int(knowledge.get("max_tool_calls") or 12))
    timeout = max(1, int(knowledge.get("timeout_seconds") or 900))
    max_chars = max(500, int(knowledge.get("max_brief_chars") or 4000))
    raw_seeds = knowledge.get("seed_terms")
    seeds = tuple(str(s) for s in raw_seeds) if isinstance(raw_seeds, (list, tuple)) else ()

    cache_path = cache_dir / CACHE_FILE
    key = _key(goal, url, str(llm["model"]), seeds)
    cached = _read_cache(cache_path).get(key)
    if cached and cached.get("brief"):
        return {**cached, "cached": True}

    started = time.time()
    try:
        text, calls, used = asyncio.run(asyncio.wait_for(
            _research(goal, url, allowed, llm, max_calls, seeds), timeout))
        status = "ok" if text else "empty"
    except Exception as exc:  # noqa: BLE001 - advisory: never block generation
        text, calls, used, status = "", 0, [], f"failed: {_describe(exc)}"
    entry = {"status": status, "brief": "", "tool_calls": calls, "tools_used": used,
             "seconds": round(time.time() - started, 1)}
    if status != "ok":
        return {**entry, "cached": False}
    entry.update(brief=HEADER.format(name=name) + "\n" + text[:max_chars], mcp_url=url, goal=goal[:500])
    # Re-read: another tournament may have cached a different goal meanwhile.
    cache = _read_cache(cache_path)
    cache[key] = entry
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {**entry, "cached": False}
