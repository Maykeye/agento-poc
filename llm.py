import copy
import json
import os
import requests
import sys
import traceback
from dataclasses import dataclass, field
from typing import Optional
from tool import Tool, ToolCall
from llm_fix import llm_fix_message, llm_model_id, coerce_arg_types
import utils
import utilsql
from utils import TEMP_DIR
from context import context_handler
import re

RE_WS = re.compile(r"\s", re.MULTILINE)
GLOBAL_STEPS = int(os.getenv("LLAMA_AGENTO_DEBUG", "-1"))


def purge(messages: list[dict], _info=[]) -> list[dict]:
    """Removes reasoning_content"""
    N = int(os.getenv("AGENTO_PURGE_N", "15"))
    if not _info:
        print(f"%% Purging after: {N} messages")
        _info.append(1)
    L = 250
    messages = copy.deepcopy(messages)
    for message in messages[:-N]:
        if not (content := message.get("reasoning_content")) or len(content) <= L:
            continue
        message["reasoning_content"], tail = content[:L], content[L:]
        if found := RE_WS.search(tail):
            message["reasoning_content"] += tail[: found.start()] + "..."
    return messages


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
        url = os.environ.get("LLAMA_CPP_URL", "http://localhost")
        self.url = (
            f"{url}:{os.environ.get('LLAMA_CPP_PORT', 10000)}/v1/chat/completions"
        )
        self.tools: dict[str, Tool] = {}
        self.tools_indentation = 1
        self.callback = LLMVerbose()
        self.tool_calls_id = []
        self.llm_id = utilsql.llm_id()
        self.last_tool_call_num: Optional[int] = None
        self.model_id = llm_model_id(self.url)
        self._logged_tools_info = False

    def log_tools_info(self):
        if self._logged_tools_info:
            return
        # TODO: log tools if they are different from last time?
        self._logged_tools_info = True

    def clone(self):
        # TODO: copy or not callback? Now we assign
        llm = copy.deepcopy(self)
        llm._logged_tools_info = False
        llm.callback = self.callback
        llm.llm_id = utilsql.llm_id()
        return llm

    def add_tool(self, tool: Tool):
        """Add tools, description is taken from docstring, description  and type of arguments is taken from their type info(Annotated[raw, description])"""
        assert tool.name not in self.tools, f"tool {tool.name} must be unique"
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
        self.log_tools_info()
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

    def estimate_pseudo_tokens(self, messages: list[dict]):
        message_str = json.dumps(messages)
        tools_str = "\n".join(
            json.dumps(tool.llm_func_tool_info()) for tool in self.tools.values()
        )
        prompt = message_str + tools_str
        return f"%% PSEUDO-TOKENS: {int(len(prompt)//4)}; MSGS: {len(messages)}; Δ𝑡: {utils.delta_time_str()}"

    def _debug_abort(self):
        global GLOBAL_STEPS
        if GLOBAL_STEPS < 0:
            return
        GLOBAL_STEPS -= 1
        if not GLOBAL_STEPS:
            print("%%% LLAMA_AGENTO_DEBUG reached zero")
            exit(1)

    def _generate(self, messages: list[dict]) -> Response:
        """Generate response. If tools to be called, will recurisvely call itself and return last response. If calling tools is not desired, create new LLM instance with empty tools"""

        self._debug_abort()

        print(self.estimate_pseudo_tokens(messages))
        context_handler().prepare_current_llm(self)
        payload_msgs = purge(messages)

        payload = {
            "messages": payload_msgs,
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
            if os.getenv("LLAMA_AGENTO_VERBOSE"):
                print("%%%", data, file=sys.stderr)
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

        # Qwen can fail tool call and insert it into context/reasoning_content
        res = llm_fix_message(res, self.model_id, self.tools)

        msg = self.msg_assistant(res.content, res.reasoning_content)
        if res.tool_calls:
            msg["tool_calls"] = []
            for tool in res.tool_calls:
                msg["tool_calls"].append(tool.llm_func_call_info())
        messages.append(msg)
        payload_msgs.append(copy.deepcopy(messages[-1]))
        if res.tool_calls:
            # llm_fix_message may have fixed the message
            finish_reason = "tool_calls"

        if finish_reason == "tool_calls":
            for call in res.tool_calls:
                if call.function not in self.tools:
                    utils.error(
                        f"LLM {self.llm_id} to call `{call.function}` (type:{type(call.function)}), tool list: {self.tools.keys()}"
                    )
                    utilsql.log_tool_error(
                        llm_id=self.llm_id,
                        reason="non-existing function",
                        name_tag=self.name_tag(),
                        tool_list=str(list(self.tools.keys())),
                        function_name=call.function,
                        function_args=call.arguments,
                    )
                tool_callback = self.tools[call.function]
                pfx = f">>>{self.name_tag()} {LLMVerbose.TOOL_CALL}FUNC:{LLMVerbose.RESET}"
                print(f">>>{pfx} {call.function[:32]} ARGS: `{call.arguments[:200]}`")
                self.tool_calls_id.append(call.id)

                # Log generation before calling tool to get history_id
                self.last_tool_call_num = utilsql.log_generation(
                    utilsql.prompt_id(), self.llm_id, payload_msgs, self.tools
                )

                try:
                    args = json.loads(call.arguments)
                    # Coerce argument types to match tool's expected types
                    args = coerce_arg_types(args, tool_callback)
                    result = tool_callback(**args)  # type: ignore
                except Exception as ex:
                    tb = traceback.format_exc()
                    traceback.print_exc()
                    utilsql.log_tool_error(
                        llm_id=self.llm_id,
                        reason="execution error",
                        name_tag=self.name_tag(),
                        tool_list=str(list(self.tools.keys())),
                        function_name=call.function,
                        function_args=call.arguments,
                        exception_traceback=tb,
                    )
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
                payload_msgs.append(copy.deepcopy(messages[-1]))

            return self._generate(messages)

        # Log normal messages
        self.last_tool_call_num = utilsql.log_generation(
            utilsql.prompt_id(), self.llm_id, payload_msgs, self.tools
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
        return "".join([f"q{x.llm.llm_id}" for x in LLM.INSTANCES])

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
