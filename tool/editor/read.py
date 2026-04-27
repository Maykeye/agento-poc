from typing import Annotated

from llm import LLM
from tool import Tool
from tool.io import ToolReadFile
from tool.editor.editor import ToolEditor
from tool.editor.tool_list import EDITOR_TOOLS


class EditorToolRead(Tool):
    """Read another file while in editor mode."""

    def __init__(self):
        super().__init__(
            name="read_file_to_view",
            description="""Read another file to view while in editor mode.

This allows you to view other files without leaving the editor.
This does not change the current file being edited.
All edit commands after read_file will apply to the file that was edited before reading""",
        )

    def __call__(self, path: Annotated[str, "Path to the file to read"]):
        read_tool = ToolReadFile()
        result = read_tool(path)
        if isinstance(result, str):
            llm = LLM.INSTANCES[-1].llm
            llm_id = id(llm)
            current_path = ToolEditor._state[llm_id].path
            result += f"\n\nNote: still editing {current_path}"

        return result


EDITOR_TOOLS.append(EditorToolRead)
