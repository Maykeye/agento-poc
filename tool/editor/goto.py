from typing import Annotated

from tool import Tool
from llm import LLM
from tool.editor.editor import ToolEditor
from tool.editor.tool_list import EDITOR_TOOLS
from config import real_path


class EditorToolGoto(Tool):
    """Go to a specific line and update current line position."""

    def __init__(self):
        super().__init__(
            name="goto",
            description="""Jump to a specific line number (1-indexed) and set current line to it.

After jumping, prints the buffer starting from the new current line position.
Line numbers must be within file bounds (1 to total_lines).
After goto, old buffer messages are pruned to save context.""",
        )

    def __call__(
        self, line_number: Annotated[int, "Line number to jump to (1-indexed)"]
    ):
        # Get current LLM instance
        if not LLM.INSTANCES:
            return {"error": "No LLM instance available"}

        llm = LLM.INSTANCES[-1].llm
        llm_id = id(llm)

        # Get current state
        if llm_id not in ToolEditor._state:
            return {"error": "Not in editor mode"}

        path = ToolEditor._state[llm_id].path

        p = real_path(path)
        text = p.read_text()
        lines = text.splitlines()
        total_lines = len(lines)

        # Validate line number is within bounds (1-indexed)
        if line_number < 1:
            return {"error": f"Line number must be >= 1, got {line_number}"}

        if line_number > total_lines:
            return {
                "error": f"Line number {line_number} exceeds file length ({total_lines} lines)",
                "suggestion": f"Use a line number between 1 and {total_lines}",
            }

        # Update current line
        ToolEditor._state[llm_id].current_line = line_number

        # Format output with buffer
        output_lines = []
        output_lines.append(f"Goto line {line_number}")
        output_lines.append("")

        # Print buffer from new current line
        buffer_output = ToolEditor._format_buffer(path, line_number, text)
        output_lines.append(buffer_output)

        # Prune old buffers
        messages = LLM.INSTANCES[-1].messages
        ToolEditor._prune_old_buffers(messages)

        return "\n".join(output_lines)


EDITOR_TOOLS.append(EditorToolGoto)

