"""Local OpenAI-compatible proxy to the ALCF Inference Service that injects a fresh token.

ALCF access tokens last 48 h and Hermes / Jnana only take a fixed API key, so a
key pasted into their configs dies mid-campaign. Point them at this proxy
instead (base_url http://127.0.0.1:51910/v1, any api_key); every request is
forwarded with a current Bearer token from alcf-tokens, which refreshes it.

    alcf-tokens login inference            # once; re-login every 30 days
    python scripts/alcf_inference_proxy.py  # leave running

Environment:
  ALCF_PROXY_PORT     listen port on 127.0.0.1 (default 51910)
  ALCF_CLUSTER        sophia (default), metis, minerva or tara
Needs aiohttp and alcf-tokens.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

import aiohttp
from aiohttp import web
from alcf_tokens.auth import get_access_token

ROOT = "https://inference-api.alcf.anl.gov/resource_server"
CLUSTER = os.environ.get("ALCF_CLUSTER", "sophia")
# Chat/completions base per cluster (index.md "Available Clusters").
UPSTREAM = f"{ROOT}/{CLUSTER}/vllm/v1" if CLUSTER == "sophia" else f"{ROOT}/{CLUSTER}/api/v1"
TOKEN_TTL_S = 300  # re-ask alcf-tokens this often; it refreshes the 48 h token itself
# Hop-by-hop and auth headers are never forwarded in either direction.
_DROP = {"host", "authorization", "content-length", "transfer-encoding", "connection",
         "keep-alive", "content-encoding", "accept-encoding"}
log = logging.getLogger("alcf-proxy")


class Token:
    def __init__(self) -> None:
        self._value, self._at = "", 0.0
        self._lock = asyncio.Lock()

    async def get(self, force: bool = False) -> str:
        async with self._lock:
            if force or not self._value or time.time() - self._at > TOKEN_TTL_S:
                # Blocking file/network work; never run it on the event loop.
                self._value = await asyncio.get_running_loop().run_in_executor(
                    None, get_access_token, "inference")
                self._at = time.time()
            return self._value


async def models(request: web.Request) -> web.Response:
    """/v1/models from the cluster's model catalog; vLLM's own route is not exposed."""

    session: aiohttp.ClientSession = request.app["session"]
    token = await request.app["token"].get()
    async with session.get(f"{ROOT}/{CLUSTER}/models",
                           headers={"Authorization": f"Bearer {token}"}) as resp:
        if resp.status != 200:
            return web.Response(status=resp.status, body=await resp.read(),
                                content_type=resp.content_type)
        catalog = await resp.json()
    data = []
    for entry in catalog if isinstance(catalog, list) else []:
        window = (entry.get("capabilities") or {}).get("context_window_tokens")
        item = {"id": entry["id"], "object": "model", "owned_by": entry.get("cluster", CLUSTER)}
        if window:
            item.update(context_length=window, max_model_len=window)
        data.append(item)
    return web.json_response({"object": "list", "data": data})


async def forward(request: web.Request) -> web.StreamResponse:
    session: aiohttp.ClientSession = request.app["session"]
    body = await request.read()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _DROP}
    url = f"{UPSTREAM}/{request.match_info['tail']}"
    for attempt in (0, 1):
        token = await request.app["token"].get(force=attempt == 1)
        resp = await session.request(request.method, url, params=request.query, data=body,
                                     headers={**headers, "Authorization": f"Bearer {token}"})
        # One retry with a forced refresh if the token was rejected.
        if resp.status in (401, 403) and attempt == 0:
            resp.release()
            continue
        break
    out = web.StreamResponse(status=resp.status, headers={
        k: v for k, v in resp.headers.items() if k.lower() not in _DROP})
    await out.prepare(request)
    try:
        # Pass chunks through as they arrive so streamed (SSE) completions stay streamed.
        async for chunk in resp.content.iter_any():
            await out.write(chunk)
    finally:
        resp.release()
    await out.write_eof()
    return out


async def _session(app: web.Application):
    # Long reads: a cold model may sit in the cluster queue before answering.
    app["session"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=1800))
    app["token"] = Token()
    await app["token"].get()  # fail at startup, not on the first request, if not logged in
    yield
    await app["session"].close()


def make_app() -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.cleanup_ctx.append(_session)
    app.router.add_get("/v1/models", models)
    app.router.add_route("*", "/v1/{tail:.*}", forward)
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    port = int(os.environ.get("ALCF_PROXY_PORT", "51910"))
    log.info("forwarding http://127.0.0.1:%d/v1 -> %s", port, UPSTREAM)
    web.run_app(make_app(), host="127.0.0.1", port=port, access_log=None)
