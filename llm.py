from dataclasses import dataclass, field
from typing import Optional
import json
import os
import requests
import sys

from tool import Tool


@dataclass
class ToolCall:
    function: str
    arguments: str
    id: str


@dataclass
class Response:
    content: str
    reasoning_content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


def open_log():
    # TODO: logging.info?
    path = f"/run/user/{os.getuid()}/messages.log"
    try:
        f = open(path, "w")
        print(f"LOGGING ALL MESSAGES TO {f}", file=sys.stderr)
        return f
    except Exception as e:
        print(f"Can't open logging file {path}")
        return None


LOG = open_log()


class LLM:
    """LLM is a wrapper around llama.cpp that eases generation and tool integration"""

    def __init__(self) -> None:
        self.url = f"http://localhost:{os.environ.get('LLAMA_CPP_PORT', 10000)}/v1/chat/completions"
        self.tools: dict[str, Tool] = {}
        self.tools_indentation = 1
        self.callback = LLMVerbose()

    def add_tool(self, tool: Tool):
        """Add tools, description is taken from docstring, description  and type of arguments is taken from their type info(Annotated[raw, description])"""
        assert tool.name not in self.tools
        self.tools[tool.name] = tool

    def generate(self, messages: list[dict], do_tool_calls=True) -> Response:
        """Generate response. If calls to be called, will recurisvely call itself and return last response"""

        assert isinstance(do_tool_calls, bool)
        payload = {
            "messages": messages,
            "stream": True,
        }

        if self.tools:
            payload["tool_choice"] = "auto"
            payload["tools"] = self._tool_listing_for_llm()

        response = requests.post(self.url, json=payload, stream=True)
        response.raise_for_status()
        if LOG:
            print("----------------", file=LOG)
            print(f"<<< {json.dumps(messages, ensure_ascii=False, indent=2)}", file=LOG)
            print("----------------", file=LOG)

        finish_reason = ""
        res = Response(content="", reasoning_content="")
        for bdata in response.iter_lines():
            sdata = bdata.decode().strip().removeprefix("data: ")
            if sdata == "[DONE]":  # posssible if we didn't recognize finish_reason
                print(
                    "Unexpected [DONE], check for possibly unsupported finish_reason in log",
                    file=sys.stderr,
                )
                break
            if not sdata:
                continue
            data = json.loads(sdata)
            data = data["choices"][0]
            if LOG:
                LOG.write(f">>> {sdata}\n")
            finish_reason = data["finish_reason"]
            if self.callback and finish_reason:
                self.callback({}, finish_reason=finish_reason)
            if finish_reason == "stop":
                break
            if finish_reason == "tool_calls":
                break
            data = data["delta"]
            assert finish_reason is None, f"unexpected {finish_reason=}"
            if self.callback:
                self.callback(data)

            if "content" in data:
                if (value := data["content"]) is not None:
                    res.content += value
                continue

            if (text := data.get("reasoning_content")) is not None:
                res.reasoning_content += text
                continue

            if (calls := data.get("tool_calls")) is not None:
                assert isinstance(calls, list)
                assert len(calls) == 1

                call = calls[0]
                assert "function" in call
                if "id" in call:
                    assert call["type"] == "function"
                    res.tool_calls.append(
                        ToolCall(
                            function=call["function"]["name"],
                            arguments=call["function"].get("arguments", ""),
                            id=call["id"],
                        )
                    )
                else:
                    res.tool_calls[-1].arguments += call["function"]["arguments"]
                continue

            print(
                f">>> WARNING skipped unexpected result {data} (raw: {sdata})",
                file=sys.stderr,
            )

        msg = self.msg_assistant(res.content, res.reasoning_content)
        if res.tool_calls:
            msg["tool_calls"] = []
            for tool in res.tool_calls:
                msg["tool_calls"].append(
                    {
                        "id": tool.id,
                        "type": "function",
                        "function": {
                            "name": tool.function,
                            "arguments": tool.arguments,
                        },
                    }
                )
        messages.append(msg)

        if finish_reason == "tool_calls" and do_tool_calls:
            for call in res.tool_calls:
                tool_callback = self.tools[call.function]
                print(f">>> FUNC: {call.function[:32]} ARGS: `{call.arguments[:200]}`")
                try:
                    args = json.loads(call.arguments)
                    result = tool_callback(**args)  # type: ignore
                except Exception as ex:
                    result = json.dumps({"TOOL CALL ERROR": str(ex)})

                if isinstance(result, (list, dict)):
                    result = json.dumps(
                        result, ensure_ascii=False, indent=self.tools_indentation
                    )

                assert isinstance(
                    result, str
                ), f"{type(result)} while str expected in {call}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.function,
                        "content": result,
                    }
                )

            return self.generate(messages, do_tool_calls)

        return res

    def msg_assistant(self, txt: str, reasoning: Optional[str] = None) -> dict:
        d = {"role": "assistant", "content": txt}
        if reasoning is not None:
            d["reasoning_content"] = reasoning
        return d

    def msg_system(self, txt: str):
        return {"role": "system", "content": txt}

    def msg_user(self, txt: str):
        return {"role": "user", "content": txt}

    def _tool_listing_for_llm(self) -> list:
        out = []

        for tool in self.tools.values():
            out.append(tool.llm_func_tool_info())
        return out


class LLMVerbose:
    """Print LLM communication to terminal"""

    CONTENT = "\x1b[37m"
    REASONING = "\x1b[33m"
    TOOL_CALL = "\x1b[32m"
    RESET = "\x1b[0m"

    def __init__(self) -> None:
        self.last_type = False
        self.data = []
        self.last = ""

    def __call__(self, data: dict, finish_reason=None):
        self.data.append(data)
        REASONING = "reasoning_content"
        CONTENT = "content"

        if finish_reason:
            print(self.RESET, end="" if self.last.endswith("\n") else "\n")
            return

        if (data.get("tool_calls")) is not None:
            self.last_type = "tool_calls"
            self.last = ""
            if text := data.get("arguments"):
                print(self.TOOL_CALL + text, flush=True, end="")
            return

        if (text := data.get(REASONING)) is not None:
            if self.last_type != REASONING:
                self.last_type = REASONING
                text = self.REASONING + text
            print(text, flush=True, end="")
            return

        if (text := data.get(CONTENT)) is not None:
            if self.last_type != CONTENT:
                if not self.last.endswith("\n"):
                    print()
                self.last_type = CONTENT
                text = self.CONTENT + text
            print(text, flush=True, end="")
            return
