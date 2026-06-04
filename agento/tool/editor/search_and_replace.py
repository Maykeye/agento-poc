from typing import Annotated

from agento.config import real_path
from agento.llm import LLM
from agento.tool import Tool
from agento.tool.editor.editor import ToolEditor
from agento.tool.editor.tool_list import EDITOR_TOOLS
from agento.tool.io import ToolWriteFile
from agento import utils


class EditorToolSearchReplace(Tool):
    """Edit file by replacing text in the current file."""

    def __init__(self):
        super().__init__(
            name="search_and_replace",
            description="""Replace text in the current file.
Use search_and_replace for simple, one-or-all replacements. Use `sed` for more precise replaces and regexs.

⚠️ CRITICAL RESTRICTIONS:
- if replace_all is false, replace_from must exist EXACTLY ONCE in the current buffer view
- After replacement, file is written immediately

`diff` will be returned
""",
        )

    def __call__(
        self,
        replace_from: Annotated[
            str,
            "Text to find (must exist exactly once in CURRENT BUFFER, not whole file)",
        ],
        replace_with: Annotated[str, "Text to replace with"],
        replace_all: Annotated[bool, "Should replace all instances?"],
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
        full_text = p.read_text()
        full_lines = full_text.splitlines()
        total_lines = len(full_lines)

        # Calculate buffer range
        buffer_start_idx = 0
        buffer_end_idx = total_lines

        # Get buffer content (the visible lines)
        buffer_lines = full_lines[buffer_start_idx:buffer_end_idx]
        buffer_content = "\n".join(buffer_lines)

        # Check if replace_from exists in buffer
        full_count = buffer_content.count(replace_from)

        if full_count == 0:
            # Show where it might be in the buffer for debugging
            return {
                "error": f"Text not found in current buffer",
                "replace_from": repr(replace_from),
                "suggestion": "Use goto or find_next/find_prev to navigate to the text you want to edit, or use patch_current_file",
            }

        if full_count > 1 and not replace_all:
            return {
                "error": f"Text appears {full_count} times in current buffer (must be exactly once)",
                "replace_from": repr(replace_from),
                "suggestion": "Be more specific with replace_from to match exactly one occurrence, or use patch_current_file for multiple changes",
            }

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

        diff = utils.diff_gen(full_text, new_full_text, path)

        # Format success output
        output_lines = []
        output_lines.append(f"Replaced text in {path}")
        output_lines.append(f"  From: {repr(replace_from)}")
        output_lines.append(f"  To:   {repr(replace_with)}")
        output_lines.append("```diff")
        output_lines.extend(diff)
        output_lines.append("```")

        # Prune old buffers
        messages = LLM.INSTANCES[-1].messages
        ToolEditor._prune_old_buffers(messages)

        result = "\n".join(output_lines)
        return result


EDITOR_TOOLS.append(EditorToolSearchReplace)
