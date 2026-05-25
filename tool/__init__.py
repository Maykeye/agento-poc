from typing import Callable, Literal, Optional
import inspect
import typing
from typing import Annotated
from dataclasses import dataclass, field
import subprocess
import os


@dataclass
class Tool:
    """Dataclass to store tool information"""

    name: str
    description: str
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.parameters:
            self.parameters = parse_tool_parms(self.__call__)  # type: ignore

    def llm_func_tool_info(self):
        return {
            "type": "function",
            "function": {
                "type": "function",
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def init_system_msg(self) -> Optional[str]:
        return None


@dataclass
class ToolEcho(Tool):
    def __init__(self):
        super().__init__("echo", "(debug) print string")

    def __call__(self, string: Annotated[str, "String to print to screen"]):
        print("DEBUG STRING ECHOED", string)
        return f"String {string} printed"


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


def parse_tool_parms(tool_fn: Callable):
    def _parse_type(raw_type) -> dict:
        origin = typing.get_origin(raw_type)
        if origin == Literal:
            values = typing.get_args(raw_type)
            res = _parse_type(type(values[0]))
            res["enum"] = list(values)
            return res

        if origin is list or raw_type is list:
            items_type = {"type": "string"}  # Default inner type
            args = typing.get_args(raw_type)
            if args:
                items_type = _parse_type(args[0])
            return {"type": "array", "items": items_type}

        known = {
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            str: "string",
        }
        if name := known.get(raw_type):
            return {"type": name}
        raise ValueError(raw_type)

    sig = list(inspect.signature(tool_fn).parameters.values())
    args = {}
    required = []
    for parm in sig:
        if parm.name == "self":
            continue
        assert (
            typing.get_origin(parm.annotation) is Annotated
        ), f"tool must have Annotated arg, not {parm}"

        type_info = typing.get_args(parm.annotation)
        assert len(type_info) == 2, "Expected: Annotated[type, description]"
        argument = _parse_type(type_info[0])
        argument["description"] = type_info[1]
        args[parm.name] = argument
        required.append(parm.name)
    # print(tool_fn, args)

    return {
        "type": "object",
        "properties": args,
        "required": required,
    }


class ToolRegistry:
    """Collection of tools available to MCP server"""

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def add_tool(self, tool: Tool):
        """
        Registers a new tool with the server.

        Args:
            tool: A Tool instance to register.

        Raises:
            ValueError: If a tool with the same name already exists.
        """

        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' already exists in server registry.")

        self.tools[tool.name] = tool
        print(f"✅ Tool registered: {tool.name}")

    def get_tools_list(self):
        """Returns the list of tools formatted for JSON-RPC response"""
        return [tool.llm_func_tool_info() for tool in self.tools.values()]

    def call_tool(self, name, arguments):
        """Executes a tool by name with given arguments."""
        if tool := self.tools.get(name):
            return tool(**arguments)  # type: ignore

        raise KeyError(f"Tool '{name}' not found")


class TimeTool(Tool):
    """Sample"""

    def __init__(self):
        super().__init__(
            name="get_current_time",
            description="Returns the current local time in ISO format",
        )

    def __call__(self):
        from datetime import datetime

        now = datetime.now()
        return {
            "local_time": now.isoformat(),
            "timestamp": now.timestamp(),
            "timezone": str(now.tzinfo) if now.tzinfo else "Local",
            "formatted": now.strftime("%Y-%m-%d %H:%M:%S"),
        }


def empty_stdin():
    """Helper function to disable stdin in subprocesses"""
    return open("/dev/null")


def run_executable(args: list[str], stdin_text: Optional[str] = None):
    env = None
    if not os.getenv("NODE_PATH"):
        env = os.environ.copy()
        env["NODE_PATH"] = "/usr/lib/node_modules/"
    try:
        # TODO: tee me
        if stdin_text is not None:
            p = subprocess.run(
                args, capture_output=True, text=True, input=stdin_text, env=env
            )
        else:
            with empty_stdin() as stdin:
                print(args)
                p = subprocess.run(
                    args, capture_output=True, text=True, stdin=stdin, env=env
                )
        return {"exitcode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except Exception as ex:
        print(ex)
        return {"error": str(ex)}
