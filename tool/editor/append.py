from typing import Annotated

from config import real_path
from llm import LLM
from tool import Tool
from tool.io import ToolAppend
from tool.editor.editor import ToolEditor
from tool.editor.tool_list import EDITOR_TOOLS


class EditorToolAppend(Tool):
    """Write new content to the current file and exit editing mode."""

    def __init__(self):
        super().__init__(
            name="append_to_file",
            description="""Append text to the end of the file.

This tool:
1. Validates that the provided path matches the currently editing file
2. Writes the entire `text` at the end of the file (replacing all existing content)

REQUIREMENTS:
- path: Must match the file currently being edited in editor mode, as only editing it is allowed.
- text: Text to write

Use this when you want to append new content to the end of the file.""",
        )

    def __call__(
        self,
        path: Annotated[
            str,
            "Path to the file to write to (must match the file currently being edited)",
        ],
        text: Annotated[
            str,
            "Complete new content to write to the file (replaces all existing content)",
        ],
    ):
        # Get current LLM instance
        if not LLM.INSTANCES:
            return {"error": "No LLM instance available"}

        llm = LLM.INSTANCES[-1].llm
        llm_id = id(llm)

        # Get current state
        if llm_id not in ToolEditor._state:
            return {"error": "No file being edited"}

        current_editing_path = ToolEditor._state[llm_id].path

        # Validate that the provided path matches the currently editing file
        provided_path = real_path(path)
        editing_path = real_path(current_editing_path)

        # Normalize paths for comparison (resolve to absolute paths)
        if provided_path != editing_path:
            return {
                "error": f"Path mismatch",
                "suggestion": f"You are editing '{current_editing_path}' but tried to append to '{path}'. Use the correct path or use finish_editing() to exit and write_file() for a different file.",
                "editing_file": current_editing_path,
                "requested_path": path,
            }

        append_tool = ToolAppend()
        return append_tool(path, text)


EDITOR_TOOLS.append(ToolAppend)
