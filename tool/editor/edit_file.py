from typing import Annotated

from config import real_path
from llm import LLM
from tool import Tool
from tool.editor.editor import EditorEntry, ToolEditor
from tool.editor.tool_list import EDITOR_TOOLS


class EditorToolEditFile(Tool):
    """Switch to editing a different file while in editor mode."""

    def __init__(self):
        super().__init__(
            name="edit_file",
            description="""Switch to editing a different file while in editor mode.

This tool allows you to switch to a new file without exiting editor mode.

REQUIREMENTS:
- The new file must exist
- After switching, the editor continues with the new file starting from line 1
- Empty files will be initialized with empty content

Use this when you need to edit other file from the current one quckly, but it's preferrably to finish_editing and in report state what needs to be edited.""",
        )

    def __call__(self, path: Annotated[str, "Path to the new file to edit"]):
        # Get current LLM instance
        if not LLM.INSTANCES:
            return {"error": "No LLM instance available"}

        llm = LLM.INSTANCES[-1].llm
        llm_id = id(llm)

        # Check if we're in editor mode
        if llm_id not in ToolEditor._state:
            return {"error": "Not in editor mode"}

        # Validate the new file
        p = real_path(path)

        if not p.exists():
            return {
                "error": f"File {path} does not exist",
                "suggestion": "Use write_file to create a new file first",
            }

        if not p.is_file():
            return {"error": f"{path} is not a file"}

        # Read the new file content
        try:
            new_text = p.read_text()
        except Exception as e:
            return {"error": f"Failed to read file {path}: {e}"}

        # Check if file is empty
        if not new_text.strip():
            # Write empty string to the file to allow editing
            p.write_text("")
            new_text = ""

        # Prune old buffers if we exceed KEEP_OLD_BUFFERS
        messages = LLM.INSTANCES[-1].messages
        ToolEditor._prune_old_buffers(messages)

        # Update editing state to the new file
        old_file = ToolEditor._state[llm_id]
        ToolEditor._state[llm_id] = EditorEntry(path, 1)

        # Format output
        output_lines = []
        output_lines.append(f"Switching from '{old_file}' to '{path}'")
        output_lines.append("")

        # Print buffer from line 1 of new file
        buffer_output = ToolEditor._format_buffer(path, 1, new_text)
        output_lines.append(buffer_output)

        return "\n".join(output_lines)


EDITOR_TOOLS.append(EditorToolEditFile)
