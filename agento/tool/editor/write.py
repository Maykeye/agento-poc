import json
from typing import Annotated

from agento.config import real_path
from agento.llm import LLM, FinishGeneration
from agento.tool import Tool
from agento.tool.editor.editor import ToolEditor
from agento.tool.editor.tool_list import EDITOR_TOOLS
from agento.tool.io import ToolWriteFile


class EditorToolWriteNewContent(Tool):
    """Write new content to the current file and exit editing mode."""

    def __init__(self):
        super().__init__(
            name="write_file",
            description="""Write the complete new content to the current file and exit editing mode.

This tool:
1. Validates that the provided path matches the currently editing file
2. Writes the entire `text` to the file (replacing all existing content)
3. Exits editor mode
4. Returns a result indicating success

REQUIREMENTS:
- path: Must match the file currently being edited in editor mode, as only editing it is allowed.
- text: Complete new content to write (replaces all existing content)

Use this when you want to completely replace the file content with new content.
The editor will exit and return control to the main LLM.""",
        )

    def __call__(
        self,
        path: Annotated[
            str,
            "Path to the file to write to (must match the file currently being edited)",
        ],
        new_content: Annotated[
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
                "suggestion": f"You are editing '{current_editing_path}' but tried to write to '{path}'. Use the correct path or use finish_editing() to exit and write_file() for a different file.",
                "editing_file": current_editing_path,
                "requested_path": path,
            }

        write_tool = ToolWriteFile()
        result = write_tool(path, new_content)

        # Check if write was successful
        if isinstance(result, dict) and "error" in result:
            return result

        # Clean up editor state
        ToolEditor.reset(llm_id)

        # Return FinishGeneration to stop the editor LLM and return to main LLM
        result_value = {
            "status": "success",
            "message": f"File {path} written successfully and editor mode exited",
            "file": path,
            "lines_written": len(new_content.splitlines()),
        }
        return FinishGeneration(value=json.dumps(result_value, indent=1))


EDITOR_TOOLS.append(EditorToolWriteNewContent)
