#!/usr/bin/env python
import json
import os
import secrets
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple
import logging

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
import httpx

TAG_START = "<"
TAG_END = ">"

TOOL_CALL_OPEN = TAG_START + "tool_call" + TAG_END
TOOL_CALL_CLOSE = TAG_START + "/tool_call" + TAG_END

UPSTREAM = (
    f"{os.environ.get('LLAMA_CPP_URL', 'http://localhost')}:"
    f"{os.environ.get('LLAMA_CPP_PORT', '10000')}"
)

logging.basicConfig(
    filename="/home/fella/.local/state/agento/proxy.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

app = FastAPI()

_client = httpx.AsyncClient(timeout=None)


class ContentTypeMode(Enum):
    NORMAL = "normal"
    TOOL_CALL_PARSING = "tool_call_parsing"


@dataclass
class ContentTypeState:
    mode: ContentTypeMode = ContentTypeMode.NORMAL
    parser: Optional["_ToolCallParser"] = None
    tool_id: str = ""
    partial_line: str = ""
    tc_header_emitted: bool = False  # track across tokens
    prev_tc_closed: bool = False  # True after first (/tool_call) seen
    tc_index: int = 0  # tool_calls array index


class _ToolCallParser:
    """Parse fragmented tool call tokens, produce incremental JSON fragments.

    For add(lhs=10, rhs=200) the fragments concatenate to:
      {"lhs":"10","rhs":"200"}
    """

    def __init__(self) -> None:
        self._function_name = ""
        self._arguments: dict[str, str] = {}
        self._in_param = False
        self._current_param_name = ""
        self._param_value_chars: list[str] = []
        self._value_started = False
        self._buffer = ""
        self._has_close_tag = False
        self._fragments: list[str] = []

    def feed(self, text: str) -> None:
        self._buffer += text
        while True:
            consumed, found_close = self._try_consume()
            if consumed == 0:
                break
            self._buffer = self._buffer[consumed:]
            if found_close:
                self._has_close_tag = True
                return

    def _try_consume(self) -> tuple[int, bool]:
        buf = self._buffer
        if not buf:
            return 0, False

        # Handle non-tag text before any tag (e.g. newlines between tags)
        if buf[0] != TAG_START:
            next_tag = buf.find(TAG_START)
            if next_tag == -1:
                val = buf
                consumed = len(buf)
            else:
                val = buf[:next_tag]
                consumed = next_tag

            if self._in_param and val and not val.isspace():
                if not self._value_started:
                    self._value_started = True
                    self._fragments.append('"')
                self._fragments.append(val)
                self._param_value_chars.append(val)
            return consumed, False

        # buf[0] == "(": consume tag
        end = buf.find(TAG_END)
        if end == -1:
            return 0, False
        tag = buf[1:end]

        if tag == "/tool_call":
            self._fragments.append("}")
            return end + 1, True

        if tag.startswith("function="):
            self._function_name = tag[9:]  # len("function=") == 9
            self._in_param = False
            self._fragments.append("{")
            return end + 1, False

        if tag.startswith("parameter="):
            # Close previous param value if started
            if self._in_param and self._value_started:
                self._fragments.append('"')
            # Comma separator before all params except the first
            if self._arguments:
                self._fragments.append(",")
            param_name = tag[10:]  # len("parameter=") == 10
            self._current_param_name = param_name
            self._in_param = True
            self._param_value_chars = []
            self._value_started = False
            self._fragments.append(json.dumps(param_name) + ":")
            return end + 1, False

        if tag == "/parameter":
            if self._in_param and self._value_started:
                self._fragments.append('"')
            if self._in_param:
                self._arguments[self._current_param_name] = "".join(
                    self._param_value_chars
                ).strip()
            self._in_param = False
            self._param_value_chars = []
            self._value_started = False
            return end + 1, False

        if tag == "/function":
            self._in_param = False
            self._param_value_chars = []
            self._value_started = False
            return end + 1, False

        return end + 1, False

    def drain_fragments(self) -> list[str]:
        frags = self._fragments
        self._fragments = []
        return frags

    def reset(self) -> None:
        """Reset parser state for a new (function=...) block."""
        self._function_name = ""
        self._arguments = {}
        self._in_param = False
        self._current_param_name = ""
        self._param_value_chars = []
        self._value_started = False
        self._has_close_tag = False

    @property
    def has_close_tag(self) -> bool:
        return self._has_close_tag


# ---------------------------------------------------------------------------
# Content processing
# Returns (emitted_text, arg_fragments, is_finish)
# ---------------------------------------------------------------------------


def _process_content(
    text: str,
    state: ContentTypeState,
) -> Tuple[str, List[str], Optional[bool]]:
    if state.mode == ContentTypeMode.TOOL_CALL_PARSING:
        return _process_tool_call_mode(text, state)
    return _process_normal_mode(text, state)


def _process_normal_mode(
    text: str,
    state: ContentTypeState,
) -> Tuple[str, List[str], None]:
    buf = state.partial_line
    if buf:
        buf += text
    else:
        buf = text

    if buf == TOOL_CALL_OPEN:
        logging.info("Leaked tool call detected")
        state.mode = ContentTypeMode.TOOL_CALL_PARSING
        state.parser = _ToolCallParser()
        state.tool_id = "call_" + secrets.token_hex(16)
        state.partial_line = ""
        return ("", [], None)

    if len(buf) <= len(TOOL_CALL_OPEN) and buf.startswith(TOOL_CALL_OPEN[: len(buf)]):
        state.partial_line = buf
        return ("", [], None)

    state.partial_line = ""
    return (buf, [], None)


def _process_tool_call_mode(
    text: str,
    state: ContentTypeState,
) -> Tuple[str, List[str], Optional[bool]]:
    state.parser.feed(text)
    frags = state.parser.drain_fragments()
    if state.parser.has_close_tag:
        return ("", frags, True)
    return ("", frags, None)


# ---------------------------------------------------------------------------
# SSE payload helpers
# ---------------------------------------------------------------------------


def _sse_payload(sse_id, sse_object, delta, finish_reason=None):
    return {
        "id": sse_id or "",
        "object": sse_object or "chat.completion.chunk",
        "choices": [{"delta": delta, "index": 0, "finish_reason": finish_reason}],
    }


def _flush_payload(sse_id, sse_object, field, value):
    return _sse_payload(sse_id, sse_object, {field: value})


def _tc_header_payload(sse_id, sse_object, tc_index, tool_id, name, args_frag):
    return _sse_payload(
        sse_id,
        sse_object,
        {
            "tool_calls": [
                {
                    "index": tc_index,
                    "id": tool_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args_frag},
                }
            ]
        },
    )


def _tc_args_payload(sse_id, sse_object, tc_index, args_frag):
    return _sse_payload(
        sse_id,
        sse_object,
        {
            "tool_calls": [
                {
                    "index": tc_index,
                    "function": {"arguments": args_frag},
                }
            ]
        },
    )


def _finish_payload(sse_id, sse_object, reason="tool_calls"):
    return _sse_payload(sse_id, sse_object, {}, finish_reason=reason)


# ---------------------------------------------------------------------------
# Main streaming handler
# ---------------------------------------------------------------------------


async def _stream(url: str, body: bytes, headers: dict):
    states: dict[str, ContentTypeState] = {
        "content": ContentTypeState(),
        "reasoning_content": ContentTypeState(),
    }
    last_type: Optional[str] = None
    sse_id = None
    sse_object = None

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", url, content=body, headers=headers) as resp:
            resp.raise_for_status()

            async for line in resp.aiter_lines():
                if not line:
                    continue
                if line == "data: [DONE]":
                    yield line + "\n\n"
                    return
                if not line.startswith("data: "):
                    yield line + "\n\n"
                    continue

                payload = json.loads(line[6:])
                if sse_id is None:
                    sse_id = payload.get("id", "")
                    sse_object = payload.get("object", "chat.completion.chunk")

                if (
                    "choices" not in payload
                    or not payload["choices"]
                    or "delta" not in payload["choices"][0]
                ):
                    yield line + "\n\n"
                    continue

                delta = payload["choices"][0]["delta"]

                cur_type = None
                if delta.get("content") is not None:
                    cur_type = "content"
                elif delta.get("reasoning_content") is not None:
                    cur_type = "reasoning_content"

                if cur_type is not None:
                    # Flush partial_line on type switch
                    if cur_type != last_type and last_type is not None:
                        old = states[last_type]
                        if old.partial_line:
                            flush_p = _flush_payload(
                                sse_id, sse_object, last_type, old.partial_line
                            )
                            yield f"data: {json.dumps(flush_p)}\n\n"
                        states[cur_type] = ContentTypeState()

                    state = states[cur_type]
                    text = delta[cur_type] or ""
                    emitted, arg_frags, is_finish = _process_content(text, state)
                    states[cur_type] = state

                    # --- TOOL CALL MODE: emit arg fragments, skip original delta ---
                    if state.mode == ContentTypeMode.TOOL_CALL_PARSING:
                        fn_name = state.parser._function_name if state.parser else ""

                        # Detect new function after previous function's args
                        if (
                            state.tc_header_emitted
                            and arg_frags
                            and arg_frags[0] == "{"
                        ):
                            # Close previous function's arguments
                            yield f"data: {json.dumps(_tc_args_payload(
                                sse_id, sse_object, state.tc_index, "}")  )}\n\n"
                            # Reset for new function
                            state.parser.reset()
                            state.tc_index += 1
                            state.tool_id = "call_" + secrets.token_hex(16)
                            state.tc_header_emitted = False
                            state.prev_tc_closed = False

                        for frag in arg_frags:
                            if not state.tc_header_emitted:
                                p = _tc_header_payload(
                                    sse_id,
                                    sse_object,
                                    state.tc_index,
                                    state.tool_id,
                                    fn_name,
                                    frag,
                                )
                                state.tc_header_emitted = True
                            else:
                                p = _tc_args_payload(
                                    sse_id, sse_object, state.tc_index, frag
                                )
                            yield f"data: {json.dumps(p)}\n\n"

                        if is_finish:
                            state.prev_tc_closed = True
                            yield f"data: {json.dumps(_finish_payload(sse_id, sse_object))}\n\n"
                            yield "data: [DONE]\n\n"
                            return

                        last_type = cur_type
                        continue  # skip yielding original payload

                    # --- NORMAL MODE ---
                    if emitted:
                        delta[cur_type] = emitted
                    last_type = cur_type
                    # Skip suppressed tokens (empty emitted = marker buffering)
                    if not emitted:
                        continue

                yield f"data: {json.dumps(payload)}\n\n"


# ---------------------------------------------------------------------------
# Proxy endpoint
# ---------------------------------------------------------------------------


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
    response_model=None,
)
async def proxy(request: Request, path: str):
    body = await request.body()
    headers = dict(request.headers)
    headers.pop("host", None)

    url = f"{UPSTREAM}/{path}"

    if request.method == "POST" and path == "v1/chat/completions" and body:
        data = json.loads(body)
        if data.get("stream"):
            return StreamingResponse(
                _stream(url, body, headers),
                media_type="application/json",
            )

    resp = await _client.request(request.method, url, content=body, headers=headers)
    return JSONResponse(content=resp.json(), status_code=resp.status_code)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("LLAMA_CPP_PROXY_PORT", "12000")),
    )
