from dataclasses import dataclass
from agento.tool import Tool
from typing import Annotated


@dataclass
class ToolTestExternalTool(Tool):
    def __init__(self):
        super().__init__(name="test_external_tool", description="testing external api")

    def __call__(
        self, number: Annotated[int, "prove an integer to get negated number"]
    ):
        return f"{-number}"


def import_tools(tools: list[Tool]):
    tools.append(ToolTestExternalTool())
    print(f"Appending tool: ToolTestExternalTool")
