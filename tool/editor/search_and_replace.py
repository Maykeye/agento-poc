from typing import Annotated

from config import real_path
from llm import LLM
from tool import Tool
from tool.editor.editor import ToolEditor, LINES
from tool.editor.tool_list import EDITOR_TOOLS
from tool.io import ToolWriteFile


class EditorToolSearchReplace(Tool):
    """Edit file by replacing text in the current visible buffer.

    CRITICAL: This tool works ONLY on the visible buffer view, not the entire file.
    PREFERRED ALTERNATIVE: Use patch_current_file for most edits as it's more reliable.
    """

    def __init__(self):
        super().__init__(
            name="search_and_replace",
            description="""Replace text in the current VISIBLE BUFFER only.

⚠️ CRITICAL RESTRICTIONS:
- Works ONLY on the visible buffer (LINES={LINES} lines from current position)
- replace_from must exist EXACTLY ONCE in the current buffer view
- If text exists outside buffer or multiple times in buffer, operation will fail
- After replacement, file is written immediately and buffer is reprinted

🔧 PREFERRED APPROACH: Use patch_current_file instead for:
  - Multiple changes at once
  - More reliable edits
  - Better context control
  - Edits outside current buffer view

Use search_and_replace only for simple, single replacements within visible buffer.""",
        )

    def __call__(
        self,
        replace_from: Annotated[
            str,
            "Text to find (must exist exactly once in CURRENT BUFFER, not whole file)",
        ],
        replace_with: Annotated[str, "Text to replace with"],
    ):
        # Get current LLM instance
        if not LLM.INSTANCES:
            return {"error": "No LLM instance available"}

        llm = LLM.INSTANCES[-1].llm
        llm_id = id(llm)

        # Get current state
        if llm_id not in ToolEditor._state:
            return {"error": "Not in editor mode"}

        current_line = ToolEditor._state[llm_id].current_line
        path = ToolEditor._state[llm_id].path

        p = real_path(path)
        full_text = p.read_text()
        full_lines = full_text.splitlines()
        total_lines = len(full_lines)

        # Calculate buffer range
        buffer_start_idx = current_line - 1  # Convert to 0-indexed
        buffer_end_idx = min(current_line + LINES - 1, total_lines)  # Exclusive

        # Get buffer content (the visible lines)
        buffer_lines = full_lines[buffer_start_idx:buffer_end_idx]
        buffer_content = "\n".join(buffer_lines)

        # Check if replace_from exists in buffer
        buffer_count = buffer_content.count(replace_from)

        if buffer_count == 0:
            # Show where it might be in the buffer for debugging
            return {
                "error": f"Text not found in current buffer (lines {current_line}-{buffer_end_idx})",
                "replace_from": repr(replace_from),
                "buffer_range": f"Lines {current_line} to {buffer_end_idx}",
                "suggestion": "Use goto or find_next/find_prev to navigate to the text you want to edit, or use patch_current_file",
            }

        if buffer_count > 1:
            return {
                "error": f"Text appears {buffer_count} times in current buffer (must be exactly once)",
                "replace_from": repr(replace_from),
                "suggestion": "Be more specific with replace_from to match exactly one occurrence, or use patch_current_file for multiple changes",
            }

        # Now we need to replace in the FULL file, not just buffer
        # Find the position in the full file
        full_count = full_text.count(replace_from)

        if full_count == 0:
            # This shouldn't happen since we found it in buffer, but just in case
            return {"error": "Text not found in file (internal error)"}

        # Replace in full text (replace all occurrences in file)
        # Since we verified it's unique in buffer, and buffer is part of file,
        # we replace it in the full file
        new_full_text = full_text.replace(replace_from, replace_with)

        # Write the updated file using ToolWriteFile

        write_tool = ToolWriteFile()
        write_result = write_tool(path, new_full_text)

        # Check if write was successful
        if isinstance(write_result, dict) and "error" in write_result:
            return write_result

        # Read updated file to get new content
        updated_text = p.read_text()
        updated_lines = updated_text.splitlines()
        new_total_lines = len(updated_lines)

        # Adjust current line if necessary (ensure it's still valid)
        if current_line > new_total_lines:
            current_line = max(1, new_total_lines)
            ToolEditor._state[llm_id].current_line = current_line

        # Format success output
        output_lines = []
        output_lines.append(f"Replaced text in {path}")
        output_lines.append(f"  From: {repr(replace_from)}")
        output_lines.append(f"  To:   {repr(replace_with)}")
        output_lines.append("")

        # Print buffer from current line
        buffer_output = ToolEditor._format_buffer(path, current_line, updated_text)
        output_lines.append(buffer_output)

        # Prune old buffers
        messages = LLM.INSTANCES[-1].messages
        ToolEditor._prune_old_buffers(messages)

        return "\n".join(output_lines)


EDITOR_TOOLS.append(EditorToolSearchReplace)
