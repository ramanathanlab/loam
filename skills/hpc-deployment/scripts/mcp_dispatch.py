"""Keep expected inspection failures separate from MCP worker/pool failures."""
from __future__ import annotations

from fastmcp.exceptions import ToolError


async def dispatch_action(pool, launch_one, action: str, *args, **kwargs):
    try:
        agent = await pool.next_agent(launch_one)
    except Exception as exc:
        raise ToolError(f"{action} worker unavailable: {type(exc).__name__}: {exc}") from exc
    try:
        return await agent.handle.action(action, *args, **kwargs)
    except Exception as exc:
        # These are answered worker requests, not evidence of a broken pool.
        # Keep security/input refusals explicit, without setting MCP isError.
        if action == 'inspect_filesystem' and isinstance(
            exc, (FileNotFoundError, PermissionError, ValueError, IsADirectoryError, NotADirectoryError)
        ):
            if isinstance(exc, FileNotFoundError):
                status = 'not_found'
            elif isinstance(exc, PermissionError):
                status = 'denied'
            else:
                status = 'invalid_request'
            result = {
                'status': status,
                'operation': args[0] if args else kwargs.get('operation'),
                'path': args[1] if len(args) > 1 else kwargs.get('path'),
                'message': f'{type(exc).__name__}: {exc}',
            }
            if status == 'not_found':
                result['exists'] = False
            return result
        raise ToolError(f"{action} failed: {type(exc).__name__}: {exc}") from exc
