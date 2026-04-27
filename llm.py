import copy
import json
import os
import requests
import sys
import traceback
from dataclasses import dataclass, field
from typing import Optional

import utilsql
from utils import name_tag, TEMP_DIR
from tool import Tool
from context import context_handler


def postprocess_tool_result(value, indent: int):
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False, indent=indent)
    notification = TEMP_DIR / "agento.notification"
    if notification.exists():
        try:
            txt = notification.read_text()
            if txt.strip():
                txt = txt.strip()
                txt = f"[SYSTEM EXTERNAL NOTIFICATION NOT RELATED TO TOOL CALL]\n{txt}"
                txt = f"{txt}\n[END OF NOTIFICATION]\n"
                txt = f"{txt}Original response from this point:\n"
                value = f"{txt}===\n{value}"
                notification.unlink()
        except Exception as e:
            print(f">>> Error during notification read/unlink: {e}")

    return value


@dataclass
class ToolCall:
    function: str
    arguments: str
    id: str

    def llm_func_call_info(self):
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.function,
                "arguments": self.arguments,
            },
        }


@dataclass
class FinishGeneration:
    """Special return type to stop generation loop and return value to caller."""

    value: str


@dataclass
class Response:
    content: str
    reasoning_content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class LlmInstace:
    llm: "LLM"
    messages: list[dict]


class LLM:
    """LLM is a wrapper around llama.cpp that eases generation and tool integration"""

    INSTANCES: list[LlmInstace] = []
    """ Keeps stack of LLM instances that are currently being run  """

    DEFAULT_SYSTEM = "You are a helpful ai assistant"

    def __init__(self) -> None:
        self.url = f"http://localhost:{os.environ.get('LLAMA_CPP_PORT', 10000)}/v1/chat/completions"
        self.tools: dict[str, Tool] = {}
        self.tools_indentation = 1
        self.callback = LLMVerbose()
        self.tool_calls_id = []
        self.llm_id = utilsql.llm_id()
        self.last_tool_call_num: Optional[int] = None

    def clone(self):
        # TODO: copy or not callback?
        llm = copy.deepcopy(self)
        llm.callback = self.callback
        return llm

    def add_tool(self, tool: Tool):
        """Add tools, description is taken from docstring, description  and type of arguments is taken from their type info(Annotated[raw, description])"""
        assert tool.name not in self.tools
        self.tools[tool.name] = tool

    def prepend_system_message(self, msgs: list[dict]):
        extra = []
        if msgs[0]["role"] == "system":
            sys = msgs[0]["content"]
        else:
            sys = LLM.DEFAULT_SYSTEM
        for tool in self.tools.values():
            if info := tool.init_system_msg():
                extra.append(info)
        appendum = ("\n\n" + "\n\n".join(extra)) if extra else ""

        msgs.insert(0, self.msg_system(sys + appendum))
        return msgs

    # TODO: split class to test/mock it easier
    # TODO: split callbacks better so all these extra stuff of functions/Response are separate
    def generate(self, messages: list[dict]) -> Response:
        if messages[0]["role"] != "system":
            raise ValueError("Call messages = llm.prepend_system_message(messages)")
        try:
            self.INSTANCES.append(LlmInstace(self, messages))
            return self._generate(messages)
        finally:
            if self.INSTANCES:
                last = self.INSTANCES[-1]
                if last.llm is self and last.messages is messages:
                    self.INSTANCES.pop()
                else:
                    print(f"LLM.INSTANCES desync", file=sys.stderr)
                    print(f" LLM: exp: {id(self)}, got: {id(last.llm)}")
                    print(f" MSG: exp: {id(messages)}, got: {id(last.messages)}")
            else:
                print("LLM.INSTANCES is empty", file=sys.stderr)

    def _generate(self, messages: list[dict]) -> Response:
        """Generate response. If tools to be called, will recurisvely call itself and return last response. If calling tools is not desired, create new LLM instance with empty tools"""

        context_handler().prepare_current_llm(self)

        payload = {
            "messages": messages,
            "stream": True,
        }

        if self.tools:
            payload["tool_choice"] = "auto"
            payload["tools"] = self._tool_listing_for_llm()

        response = requests.post(self.url, json=payload, stream=True)
        response.raise_for_status()

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
                msg["tool_calls"].append(tool.llm_func_call_info())
        messages.append(msg)

        if finish_reason == "tool_calls":
            for call in res.tool_calls:
                tool_callback = self.tools[call.function]
                pfx = f">>>{self.name_tag()} {LLMVerbose.TOOL_CALL}FUNC:{LLMVerbose.RESET}"
                print(f">>>{pfx} {call.function[:32]} ARGS: `{call.arguments[:200]}`")
                self.tool_calls_id.append(call.id)

                # Log generation before calling tool to get history_id
                self.last_tool_call_num = utilsql.log_generation(
                    utilsql.prompt_id(), self.llm_id, messages
                )

                try:
                    args = json.loads(call.arguments)
                    result = tool_callback(**args)  # type: ignore
                except Exception as ex:
                    traceback.print_exc()
                    result = json.dumps({"TOOL CALL ERROR": str(ex)})
                if self.tool_calls_id[-1:] == [call.id]:
                    self.tool_calls_id.pop()
                else:
                    print("Warning: tool call id mismatch", file=sys.stderr)

                # Check if tool returned FinishGeneration to stop the loop
                if isinstance(result, FinishGeneration):
                    # Add the FinishGeneration.value as tool response
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "name": call.function,
                            "content": postprocess_tool_result(
                                result.value, self.tools_indentation
                            ),
                        }
                    )
                    # Return response with the FinishGeneration value as content
                    return Response(
                        content=result.value, reasoning_content="", tool_calls=[]
                    )

                result = postprocess_tool_result(result, self.tools_indentation)
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

            return self._generate(messages)

        # Log normal messages
        self.last_tool_call_num = utilsql.log_generation(
            utilsql.prompt_id(), self.llm_id, messages
        )

        return res

    def msg_assistant(self, txt: str, reasoning: Optional[str] = None) -> dict:
        d = {"role": "assistant", "content": txt}
        if reasoning is not None:
            d["reasoning_content"] = reasoning
        return d

    def msg_system(self, txt: str):
        return {"role": "system", "content": txt}

    def msg_user(self, txt: str) -> dict[str, str]:
        return {"role": "user", "content": txt}

    def _tool_listing_for_llm(self) -> list:
        out = []

        for tool in self.tools.values():
            out.append(tool.llm_func_tool_info())
        return out

    def name_tag(self):
        return "".join([name_tag(id(x.llm)) for x in LLM.INSTANCES])

    def messages(self):
        assert self.INSTANCES
        assert self.INSTANCES[-1].llm is self
        return self.INSTANCES[-1].messages

    def append_tool_call(self, function: str, **kwargs) -> int:
        """Append a tool call to the current messages and return the tool call ID.
        Example:
            llm.append_tool_call("close_file", files=["foo.txt", "bar.txt"], reason="done")
            llm.append_tool_call("close_file", id="id123", files=["foo.txt"], reason="done")
        """

        # Create the tool call message
        self.last_tool_call_num = (self.last_tool_call_num or 0) + 1
        msg = {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": self.last_tool_call_num,
                    "type": "function",
                    "function": {
                        "name": function,
                        "arguments": json.dumps(kwargs),
                    },
                }
            ],
        }

        # Get current messages and append
        messages = self.messages()
        messages.append(msg)
        return self.last_tool_call_num


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
        TOOL_CALL = "tool_call"
        CONTENT = "content"

        if finish_reason:
            print(self.RESET, end="" if self.last.endswith("\n") else "\n")
            return

        if (calls := data.get("tool_calls")) is not None:
            if self.last_type != TOOL_CALL:
                if not self.last.endswith("\n"):
                    print()
            self.last_type = TOOL_CALL
            call = calls[0]

            argument = call.get("function", {}).get("arguments", "")
            if argument:
                print(self.TOOL_CALL + argument, flush=True, end="")
                self.last = argument
            return

        if (text := data.get(REASONING)) is not None:
            if self.last_type != REASONING:
                self.last_type = REASONING
                text = self.REASONING + text
            print(text, flush=True, end="")
            self.last = text
            return

        if (text := data.get(CONTENT)) is not None:
            if self.last_type != CONTENT:
                if not self.last.endswith("\n"):
                    print()
                self.last_type = CONTENT
                text = self.CONTENT + text
            print(text, flush=True, end="")
            self.last = text
            return
