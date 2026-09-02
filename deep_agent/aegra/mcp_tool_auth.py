"""Wrap MCP tools to raise LangGraph interrupts when OAuth is required."""

from __future__ import annotations

import inspect
import json
import time
from collections import defaultdict
from typing import Any

from langgraph.types import interrupt

from deep_agent.aegra.mcp_auth import NeedsAuthorization
from deep_agent.utils.pylogger import get_python_logger

logger = get_python_logger()

_AUTH_FAILURE_WINDOW = 300  # seconds
_AUTH_FAILURE_THRESHOLD = 3  # failures within window → escalate to ERROR
_auth_failure_times: dict[tuple[str, str], list[float]] = defaultdict(list)


def _track_auth_failure(user_id: str, mcp_name: str) -> bool:
    """Record an auth failure and return True if threshold exceeded.

    When the same (user, mcp) pair hits NeedsAuthorization or token
    rejection more than ``_AUTH_FAILURE_THRESHOLD`` times within
    ``_AUTH_FAILURE_WINDOW`` seconds, the caller should escalate to ERROR.
    """
    key = (user_id, mcp_name)
    now = time.monotonic()
    timestamps = _auth_failure_times[key]
    _auth_failure_times[key] = [t for t in timestamps if now - t < _AUTH_FAILURE_WINDOW]
    _auth_failure_times[key].append(now)
    return len(_auth_failure_times[key]) >= _AUTH_FAILURE_THRESHOLD


def _mcp_auth_interrupt_payload(exc: NeedsAuthorization) -> str:
    return json.dumps(
        {
            "type": "mcp_auth_required",
            "mcp_name": exc.mcp_name,
            "connect_url": exc.connect_url,
            "message": f"Connect to {exc.mcp_name} to use these tools",
        }
    )


def _extract_needs_authorization(exc: BaseException) -> NeedsAuthorization | None:
    """Unwrap a NeedsAuthorization buried inside an ExceptionGroup.

    Python 3.11+ TaskGroup wraps child exceptions in ExceptionGroup.  When
    NeedsAuthorization is raised inside a streamable-HTTP TaskGroup, the
    direct ``except NeedsAuthorization`` clause never fires.  This helper
    uses ``BaseExceptionGroup.subgroup`` to find the first match.
    """
    if isinstance(exc, NeedsAuthorization):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        group = exc.subgroup(NeedsAuthorization)
        if group and group.exceptions:
            first = group.exceptions[0]
            if isinstance(first, NeedsAuthorization):
                return first
    return None


def _fix_stringified_json_args(tool: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Parse stringified JSON args when the tool schema expects object/array.

    Some models (notably Gemini) serialize nested objects as JSON strings
    instead of proper dicts when calling tools with complex input schemas.
    Parses args whose schema type is ``object``, ``array``, or untyped
    (``Any`` — no ``type`` key in the schema property). Args explicitly
    typed as ``string`` are never modified.
    """
    schema_props: dict[str, Any] = {}
    try:
        schema_props = getattr(tool, "args", {}) or {}
    except Exception:
        logger.debug(
            "Failed to read args schema for tool '%s' — skipping JSON arg fix",
            getattr(tool, "name", "?"),
            exc_info=True,
        )
        return kwargs

    if not schema_props:
        return kwargs

    fixed = dict(kwargs)
    for key, value in fixed.items():
        if not isinstance(value, str):
            continue
        prop = schema_props.get(key, {})
        expected_type = prop.get("type", "")
        if expected_type == "string" or (
            isinstance(expected_type, list) and "string" in expected_type
        ):
            continue
        union_schemas = prop.get("anyOf", []) + prop.get("oneOf", [])
        if any("string" in str(s.get("type", "")) for s in union_schemas):
            continue
        stripped = value.strip()
        if stripped and stripped[0] in ("{", "["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, (dict, list)):
                    fixed[key] = parsed
                    logger.debug(
                        "Fixed stringified JSON arg '%s' for tool '%s'",
                        key,
                        getattr(tool, "name", "?"),
                    )
            except (json.JSONDecodeError, ValueError, RecursionError):
                pass
    return fixed


def wrap_mcp_tools_for_auth(tools: list[Any]) -> list[Any]:
    """Wrap MCP tools so ``NeedsAuthorization`` becomes a resumable interrupt."""
    wrapped: list[Any] = []
    for tool in tools:
        wrapped.append(_wrap_single_tool(tool))
    return wrapped


def _make_safe_ainvoke(target_tool: Any) -> Any:
    """Build an ainvoke wrapper that catches MCP errors for *target_tool*."""
    original_ainvoke = target_tool.ainvoke

    async def safe_ainvoke(tool_input: Any, config: Any = None, **kwargs: Any) -> Any:
        """Wrap ainvoke to catch auth interrupts and MCP errors."""
        from langchain_core.messages import ToolMessage
        from langgraph.errors import GraphBubbleUp

        try:
            return await original_ainvoke(tool_input, config, **kwargs)
        except NeedsAuthorization as exc:
            escalate = _track_auth_failure("_", exc.mcp_name)
            log = logger.error if escalate else logger.warning
            log(
                "MCP auth required for '%s' — interrupting run%s",
                exc.mcp_name,
                " [REPEATED — exceeds threshold, possible systemic issue]"
                if escalate
                else "",
            )
            interrupt(_mcp_auth_interrupt_payload(exc))
            return await original_ainvoke(tool_input, config, **kwargs)
        except GraphBubbleUp:
            raise
        except Exception as exc:
            auth_exc = _extract_needs_authorization(exc)
            if auth_exc is not None:
                escalate = _track_auth_failure("_", auth_exc.mcp_name)
                log = logger.error if escalate else logger.warning
                log(
                    "MCP auth required for '%s' (unwrapped from %s) — interrupting run%s",
                    auth_exc.mcp_name,
                    type(exc).__name__,
                    " [REPEATED — exceeds threshold, possible systemic issue]"
                    if escalate
                    else "",
                )
                interrupt(_mcp_auth_interrupt_payload(auth_exc))
                return await original_ainvoke(tool_input, config, **kwargs)

            tool_name = getattr(target_tool, "name", "unknown")
            tool_call_id = ""
            if isinstance(tool_input, dict):
                tool_call_id = str(tool_input.get("id", ""))
            logger.warning("MCP tool '%s' failed: %s", tool_name, exc)
            return ToolMessage(
                content=f"[TOOL_ERROR] {tool_name} failed: {exc}",
                name=tool_name,
                tool_call_id=tool_call_id,
                status="error",
            )

    return safe_ainvoke


def _wrap_single_tool(tool: Any) -> Any:
    coroutine = getattr(tool, "coroutine", None)
    func = getattr(tool, "func", None)

    if inspect.iscoroutinefunction(coroutine):

        async def wrapped_coroutine(**kwargs: Any) -> Any:
            auth_attempt = 0
            while True:
                try:
                    kwargs = _fix_stringified_json_args(tool, kwargs)
                    return await coroutine(**kwargs)
                except NeedsAuthorization as exc:
                    auth_exc: NeedsAuthorization = exc
                except BaseExceptionGroup as eg:
                    unwrapped = _extract_needs_authorization(eg)
                    if unwrapped is None:
                        raise
                    logger.warning(
                        "[%s] NeedsAuthorization unwrapped from %s (tool=%s)",
                        unwrapped.mcp_name,
                        type(eg).__name__,
                        getattr(tool, "name", "?"),
                    )
                    auth_exc = unwrapped
                auth_attempt += 1
                if auth_attempt > 1:
                    logger.error(
                        "[%s] MCP auth interrupt fired %d times in same tool call "
                        "— REAUTH LOOP DETECTED (tool=%s). "
                        "Investigate token store and MCP server-side session.",
                        auth_exc.mcp_name,
                        auth_attempt,
                        getattr(tool, "name", "?"),
                    )
                else:
                    logger.warning(
                        "[%s] MCP auth required — interrupting run (tool=%s)",
                        auth_exc.mcp_name,
                        getattr(tool, "name", "?"),
                    )
                interrupt(_mcp_auth_interrupt_payload(auth_exc))

        try:
            wrapped = tool.model_copy(update={"coroutine": wrapped_coroutine})
        except Exception:
            logger.warning(
                "model_copy failed for tool '%s' — patching coroutine directly",
                getattr(tool, "name", "?"),
            )
            tool.coroutine = wrapped_coroutine
            wrapped = tool
        object.__setattr__(wrapped, "ainvoke", _make_safe_ainvoke(wrapped))
        return wrapped

    if func is not None and inspect.isfunction(func):

        def wrapped_func(**kwargs: Any) -> Any:
            auth_attempt = 0
            while True:
                try:
                    kwargs = _fix_stringified_json_args(tool, kwargs)
                    return func(**kwargs)
                except NeedsAuthorization as exc:
                    auth_exc: NeedsAuthorization = exc
                except BaseExceptionGroup as eg:
                    unwrapped = _extract_needs_authorization(eg)
                    if unwrapped is None:
                        raise
                    logger.warning(
                        "[%s] NeedsAuthorization unwrapped from %s (tool=%s)",
                        unwrapped.mcp_name,
                        type(eg).__name__,
                        getattr(tool, "name", "?"),
                    )
                    auth_exc = unwrapped
                auth_attempt += 1
                if auth_attempt > 1:
                    logger.error(
                        "[%s] MCP auth interrupt fired %d times in same tool call "
                        "— REAUTH LOOP DETECTED (tool=%s). "
                        "Investigate token store and MCP server-side session.",
                        auth_exc.mcp_name,
                        auth_attempt,
                        getattr(tool, "name", "?"),
                    )
                else:
                    logger.warning(
                        "[%s] MCP auth required — interrupting run (tool=%s)",
                        auth_exc.mcp_name,
                        getattr(tool, "name", "?"),
                    )
                interrupt(_mcp_auth_interrupt_payload(auth_exc))

        try:
            wrapped = tool.model_copy(update={"func": wrapped_func})
        except Exception:
            logger.warning(
                "model_copy failed for tool '%s' — patching func directly",
                getattr(tool, "name", "?"),
            )
            tool.func = wrapped_func
            wrapped = tool
        object.__setattr__(wrapped, "ainvoke", _make_safe_ainvoke(wrapped))
        return wrapped

    object.__setattr__(tool, "ainvoke", _make_safe_ainvoke(tool))
    return tool
