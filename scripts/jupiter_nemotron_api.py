"""The private chat API shared by the Nemotron Slurm job and the login relay."""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import os
import re
import ssl
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

MODEL = "nemotron-3.5-super-vl"
CHECKPOINT = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
PROBE_TOKEN = "LOAM_NEMOTRON_READY"
ROUTES = {("GET", "/v1/models"), ("POST", "/v1/chat/completions")}


def safe_backend_error(status: int, body: bytes) -> dict:
    """Keep context-overflow recovery actionable without forwarding internals.

    Both relay hops apply this filter. Only a known error code and numeric
    context counts survive; arbitrary backend text, headers and paths do not.
    """
    fallback = {"error": {"message": f"Nemotron API returned HTTP {status}"}}
    if status != 400 or len(body) > 16 * 1024:
        return fallback
    try:
        data = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return fallback
    error = data.get("error", data) if isinstance(data, dict) else None
    if not isinstance(error, dict):
        return fallback
    message = error.get("message", "")
    if not isinstance(message, str):
        return fallback
    context = re.search(r"maximum context length is\s+(\d{1,10})\s+tokens", message, re.I)
    count = re.search(r"prompt contains\s+(?:at least\s+)?(\d{1,10})\s+input tokens", message, re.I)
    if context and count and int(count[1]) > int(context[1]):
        qualifier = "at least " if re.search(r"prompt contains\s+at least", message, re.I) else ""
        detail = (f"This model's maximum context length is {int(context[1])} tokens. "
                  f"However, your prompt contains {qualifier}{int(count[1])} input tokens, "
                  "which exceeds this limit even without any output tokens. "
                  "Please reduce the length of the input prompt.")
    elif error.get("code") == "context_length_exceeded":
        detail = "Maximum context length exceeded. Please reduce the length of the input prompt."
    else:
        return fallback
    return {"error": {"message": detail, "type": "invalid_request_error",
                      "code": "context_length_exceeded", "param": "input_tokens"}}


def atomic_json(path: Path, value: dict) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def private_address(address: str) -> str:
    ip = ipaddress.IPv4Address(address)
    if not any(ip in ipaddress.IPv4Network(net) for net in
               ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")):
        raise ValueError("Nemotron requires an internal Jupiter IPv4 address")
    return str(ip)


def readiness(attempt: Path, job_id: str) -> dict:
    record = read_json(attempt / "ready.json")
    request = read_json(attempt / "request.json")
    if (record["job_id"] != job_id or record["attempt"] != attempt.name
            or record["model"] != MODEL
            or record["checkpoint_revision"] != request["checkpoint_revision"]
            or record["image_digest"] != request["image_digest"]
            or record["expires_at"] <= time.time()):
        raise RuntimeError("Nemotron API readiness record is stale or mismatched")
    url = urlsplit(record["base_url"])
    if (url.scheme != "https" or url.username or url.password or url.path != "/v1"
            or not url.port or url.query or url.fragment):
        raise ValueError("Invalid Nemotron private API URL")
    private_address(url.hostname or "")
    return record


async def completion_check(session, base: str, token: str = "", *, tls=None, timeout=120) -> None:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": f"Reply exactly: {PROBE_TOKEN}"}],
        "max_tokens": 128,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with session.post(base + "/chat/completions", json=body, headers=headers,
                            ssl=tls, timeout=timeout, allow_redirects=False) as response:
        response.raise_for_status()
        result = await response.json()
    if result.get("model") != MODEL or PROBE_TOKEN not in (
        result["choices"][0]["message"].get("content") or ""
    ):
        raise RuntimeError("Nemotron API did not return the expected model and warm-up response")


async def streaming_tool_check(session, base: str, token: str) -> None:
    """Exercise streamed tool arguments and the follow-up chat turn."""
    tool = {
        "type": "function",
        "function": {"name": "skill_view", "description": "Read an installed skill",
                     "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                                    "required": ["name"]}},
    }
    headers = {"Authorization": f"Bearer {token}"}
    messages = [{"role": "user", "content": "Use skill_view to read the hpc-deployment skill."}]
    body = {"model": MODEL, "messages": messages, "tools": [tool], "stream": True,
            "tool_choice": {"type": "function", "function": {"name": "skill_view"}},
            "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 512}
    call_id = name = arguments = ""
    async with session.post(base + "/chat/completions", json=body, headers=headers,
                            allow_redirects=False) as response:
        response.raise_for_status()
        async for line in response.content:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if chunk.get("model") != MODEL:
                raise RuntimeError("Nemotron API streaming model mismatch")
            for choice in chunk.get("choices", []):
                for call in choice.get("delta", {}).get("tool_calls", []):
                    if call.get("index", 0) != 0:
                        raise RuntimeError("Nemotron API returned extra tool calls")
                    call_id += call.get("id") or ""
                    function = call.get("function") or {}
                    name += function.get("name") or ""
                    arguments += function.get("arguments") or ""
    if not call_id or name != "skill_view" or json.loads(arguments) != {"name": "hpc-deployment"}:
        raise RuntimeError("Nemotron API did not stream a valid skill_view call")
    messages.extend([
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}]},
        {"role": "tool", "tool_call_id": call_id, "content": f"The test marker is {PROBE_TOKEN}."},
        {"role": "user", "content": "Return the test marker from the tool response."},
    ])
    body.update(stream=False, tool_choice="none")
    async with session.post(base + "/chat/completions", json=body, headers=headers,
                            allow_redirects=False) as response:
        response.raise_for_status()
        result = await response.json()
    if result.get("model") != MODEL or PROBE_TOKEN not in (
        result["choices"][0]["message"].get("content") or ""
    ):
        raise RuntimeError("Nemotron API did not use the tool result")


async def structured_completion_check(session, base: str, token: str) -> None:
    """Check the JSON response format used by Jnana generation and judging."""
    headers = {"Authorization": f"Bearer {token}"}
    body = {
        "model": MODEL, "max_tokens": 256,
        "messages": [{"role": "user", "content": 'Return only this JSON object: {"status":"ready"}'}],
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    async with session.post(base + "/chat/completions", json=body, headers=headers,
                            timeout=120, allow_redirects=False) as response:
        response.raise_for_status()
        result = await response.json()
    message = result["choices"][0]["message"]
    if result.get("model") != MODEL or json.loads(message.get("content") or "{}") != {"status": "ready"}:
        raise RuntimeError("Nemotron API did not return the expected structured completion")


def proxy_app(session, token: str, upstream, *, allowed_address: str | None = None) -> web.Application:
    """Forward only the two OpenAI routes that Hermes uses, including SSE."""
    async def forward(request: web.Request) -> web.StreamResponse:
        if (request.method, request.path) not in ROUTES:
            raise web.HTTPNotFound()
        if allowed_address and request.remote != allowed_address:
            raise web.HTTPForbidden()
        if not hmac.compare_digest(request.headers.get("Authorization", "").encode(),
                                   f"Bearer {token}".encode()):
            raise web.HTTPUnauthorized()
        data = await request.read() if request.method == "POST" else None
        if data is not None:
            try:
                model = json.loads(data)["model"]
            except (ValueError, KeyError, TypeError):
                raise web.HTTPBadRequest(text="Expected an OpenAI chat request") from None
            if model != MODEL:
                raise web.HTTPBadRequest(text="Unexpected Nemotron model")
        try:
            base, upstream_token, tls = await upstream()
            headers = {"Content-Type": "application/json"}
            if upstream_token:
                headers["Authorization"] = f"Bearer {upstream_token}"
            async with session.request(request.method, base + request.path.removeprefix("/v1"),
                                       data=data, headers=headers, ssl=tls,
                                       allow_redirects=False) as response:
                # Avoid forwarding backend internals, cookies, or redirects.
                if response.status >= 300:
                    error_body = bytearray()
                    async for chunk in response.content.iter_chunked(4096):
                        error_body.extend(chunk)
                        if len(error_body) > 16 * 1024:
                            break
                    return web.json_response(safe_backend_error(response.status, bytes(error_body)),
                                             status=response.status if response.status >= 400 else 502)
                reply = web.StreamResponse(status=response.status, headers={
                    "Content-Type": response.headers.get("Content-Type", "application/json"),
                    "Cache-Control": "no-cache",
                })
                await reply.prepare(request)
                async for chunk in response.content.iter_any():
                    await reply.write(chunk)
                await reply.write_eof()
                return reply
        except (aiohttp.ClientError, asyncio.TimeoutError, OSError, ValueError, KeyError, RuntimeError):
            if "reply" in locals() and reply.prepared:
                # The client must see a broken stream instead of a fabricated
                # successful completion if the allocation vanishes mid-request.
                if request.transport is not None:
                    request.transport.close()
                return reply
            return web.json_response({"error": {"message": "Nemotron API unavailable"}},
                                     status=503)

    app = web.Application(client_max_size=16 * 1024 * 1024)
    for method, path in ROUTES:
        app.router.add_route(method, path, forward)
    return app


def pinned_tls(certificate: Path) -> aiohttp.Fingerprint:
    import hashlib

    return aiohttp.Fingerprint(hashlib.sha256(ssl.PEM_cert_to_DER_cert(certificate.read_text())).digest())
