import json
import traceback
import requests
import random
from tool import Tool, ToolCall
import typing

import utils

if typing.TYPE_CHECKING:
    import llm


def coerce_arg_types(
    args: dict[str, object], tool: "Tool"
) -> dict[str, object]:
    """Coerce argument types to match tool's expected parameter types.
    
    This handles cases where the LLM passes string values for parameters
    that should be integers, numbers, booleans, or arrays.
    
    Args:
        args: The parsed arguments dictionary
        tool: The Tool instance whose parameters define expected types
        
    Returns:
        A new dictionary with coerced values where possible
    """
    if not (expected_args := tool.parameters.get("properties")):
        return args

    coerced = dict(args)
    for parm, value in coerced.items():
        if not isinstance(value, str):
            continue
        if not (expected_arg := expected_args.get(parm)):
            continue
        if not (expected_type := expected_arg.get("type")):
            continue

        try:
            if expected_type == "integer":
                coerced[parm] = int(value)
            elif expected_type == "number":
                coerced[parm] = float(value)
            elif expected_type == "boolean":
                coerced[parm] = value.strip().lower() in ("true", "1", "yes")
            elif expected_type == "array":
                coerced[parm] = json.loads(value)
        except (ValueError, json.JSONDecodeError):
            pass  # Keep original value if coercion fails

    return coerced
def llm_model_id(chat_url: str):
    try:
        url = chat_url[: chat_url.rindex("/v1")] + "/v1/models"
        j = requests.get(url).json()
        data = j.get("data")
        id: str = data[0].get("id").lower()
        for patient in KNOWN_PATIENTS:
            if patient in id:
                return patient
        return id
    except:
        traceback.print_exc()
        return "<None>"


def rand_id(n=29):
    alpha = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "fix" + "".join([random.choice(alpha) for _ in range(n)])


def llm_fix_qwen(response, model_id: str, tools: dict[str, Tool]):
    import llm

    assert isinstance(response, llm.Response)
    del model_id  # not used

    def impl(txt: str):
        if not txt.strip():
            return txt
        txt = txt.rstrip()
        if not txt.endswith("</tool_call>"):
            return txt  # Noting to fix

        if txt.startswith("<tool_call>"):
            txt = f"\n{txt}"

        old = ""
        tool_calls: list[ToolCall] = []
        # Iterate over tool calls
        while True:
            # safety
            if txt == old:
                break
            old = txt
            txt = txt.rstrip()
            if not txt.endswith("</tool_call>"):
                break
            txt = txt.removesuffix("</tool_call>").rstrip()
            txt1 = txt.removesuffix("</function>").rstrip()
            if txt1 == txt:
                break
            if (cut := txt1.rfind("\n<tool_call>\n")) == -1:
                break
            txt, tool_call = txt1[:cut], txt1[cut:].lstrip()
            # Get function name
            tool_call = tool_call.removeprefix("<tool_call>\n<function=")
            if (i := tool_call.find(">\n")) <= 0:
                if "\n" in tool_call:  # Something went wrong
                    break
                i = tool_call.rfind(">")  # no args expected
            func_name, tool_call = tool_call[:i], tool_call[i + 1 :]
            parms = {}
            raw_parms = tool_call.split("\n<parameter=")
            raw_parms = raw_parms[1:]
            for raw in raw_parms:
                i = raw.find(">\n")
                if i <= 0:
                    break
                parms[raw[:i]] = raw[i + 2 :].removesuffix("\n</parameter>")

            if (tool := tools.get(func_name)) and (
                expected_args := tool.parameters.get("properties")
            ):
                # recast to python if can
                parms = coerce_arg_types(parms, tool)

            tool_calls.append(
                ToolCall(
                    function=func_name,
                    arguments=json.dumps(parms, ensure_ascii=False),
                    id=rand_id(),
                )
            )
        if tool_calls:
            tc = [tool.function for tool in tool_calls]
            utils.error(f"Tool calls was fixed and appended: {tc}")
            old_tool_calls = response.tool_calls or []
            response.tool_calls = old_tool_calls + tool_calls[::-1]

        return txt

    response.content = impl(response.content)
    response.reasoning_content = impl(response.reasoning_content)
    return response


def llm_fix_message(response: "llm.Response", model_id: str, tools) -> "llm.Response":
    fixer = KNOWN_PATIENTS.get(model_id, lambda message, *_: message)
    fixer(response, model_id, tools)
    return response


KNOWN_PATIENTS = {"qwen3.6-27b": llm_fix_qwen}
