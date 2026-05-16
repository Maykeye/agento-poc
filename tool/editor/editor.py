"""
Tool Editor - A specialized editing mode for LLM with buffer-based operations.

This module provides an interactive editing mode where LLM works with a buffered view
of a file, with specialized tools for navigation, pattern searching, and editing.
"""

import copy
from dataclasses import dataclass
from typing import Annotated

from config import real_path
from llm import LLM
from tool import Tool
from tool.editor.tool_list import EDITOR_TOOLS

# Constants
LINES = 250  # Size of buffer (number of lines to show)
KEEP_OLD_BUFFERS = 5  # Number of buffer prints to keep in messages


@dataclass
class EditorEntry:
    path: str
    current_line: int  # 1 indexed


class ToolEditor(Tool):
    _state: dict[int, EditorEntry] = {}  # {id(LLM) -> state }

    @staticmethod
    def reset(id=None):
        if id is None:
            ToolEditor._state.clear()
        else:
            if id in ToolEditor._state:
                del ToolEditor._state[id]

    SKIP_PRINTING: bool = False

    def __init__(self):
        super().__init__(
            name="edit_file",
            description="""Start editing a file in special editor mode.

This tool enters an interactive editing session where you work with a buffered view of the file.
Empty or non-existing files can be edited - they will be initialized with empty content.""",
        )

    @staticmethod
    def init_editor_tools(llm: LLM):
        for tool in EDITOR_TOOLS:
            llm.add_tool(tool())

    def __call__(
        self,
        path: Annotated[str, "Path to the file to edit (must exist)"],
    ):
        p = real_path(path)

        if not p.exists():
            p.write_text("")
        elif not p.is_file():
            return {"error": f"{path} is not a file"}

        # Read file content
        text = p.read_text()

        # Check if file is empty
        if not text.strip():
            # Write empty string to the file to allow editing
            p.write_text("")
            text = ""

        # Clone the current LLM
        original_llm = LLM.INSTANCES[-1].llm
        original_messages = LLM.INSTANCES[-1].messages

        # Create new LLM for editing
        editor_llm = original_llm.clone()

        # Remove all tools from the cloned LLM
        editor_llm.tools.clear()

        # Add editor-specific tools
        ToolEditor.init_editor_tools(editor_llm)

        # Store current line and file path for this LLM instance
        llm_id = id(editor_llm)
        ToolEditor._state[llm_id] = EditorEntry(path, 1)

        # Prepare messages for the editor LLM
        editor_messages = copy.deepcopy(original_messages)

        # Remove the parent's tool call from the clone's memory
        if (
            editor_messages[-1]["role"] == "assistant"
            and "tool_calls" in editor_messages[-1]
        ):
            editor_messages.pop()

        # Add system message about editor mode
        editor_messages.append(
            {
                "role": "user",
                "content": f"""[SYSTEM OVERRIDE: EDITOR MODE ACTIVATED]

You are now in EDITOR MODE for file: {path}

EDITOR MODE RULES:
1. You are working with a buffered view of the file (showing {LINES} lines at a time)
2. Line numbers are 1-indexed and displayed as 5-digit numbers (e.g., 00001, 00123)
3. The buffer shows lines starting from your current line position
4. You can navigate using: goto <line>, find_prev/find_next <regex>
5. You can switch files using: edit_file <path>
6. When done, use finish_editing(report) to quit editing and continue other tasks(e.g. compilation)
7. All tools except read work only on the current buffer view
8. The current file path is: {path}

Current buffer (starting from line 1):"""
                + "\n"
                + ToolEditor._format_buffer(path, 1, text),
            }
        )

        # Generate response from editor LLM
        result = editor_llm.generate(editor_messages)

        # After editing is done, restore current line tracking for original LLM
        # (editor LLM instance will be cleaned up)

        return {"status": "editing_complete", "file": path, "result": result.content}

    @staticmethod
    def _format_buffer(path: str, start_line: int, full_text: str) -> str:
        """Format a buffer view of the file starting from start_line (1-indexed).

        Returns formatted text with line numbers.
        Line numbers are shown only for:
        - First line of buffer
        - Last line of buffer
        - Lines divisible by 10
        """
        lines = full_text.splitlines()
        total_lines = len(lines)

        # Calculate the range to display
        end_line = min(start_line + LINES - 1, total_lines)

        # Format lines with line numbers
        buffer_lines = [lines[i] for i in range(start_line - 1, end_line)]

        # Add file info at the top
        info = f"[FILE: {path} | LINES: {start_line}..{end_line}/{total_lines}]"
        info += "\n```"

        result = "\n".join([info] + buffer_lines)
        result += "\n```"
        if not ToolEditor.SKIP_PRINTING:
            print(result)
        return result

    @staticmethod
    def _prune_old_buffers(messages: list[dict], keep_count: int = KEEP_OLD_BUFFERS):
        """Prune old buffer prints and tool calls from messages.

        Iterates messages from last backwards and prunes tool calls and outputs
        that are beyond keep_count from the end.

        This is similar to context/suffix.py pruning but for editor buffers.
        """

        # Track buffer prints from the end
        buffer_count = 0
        editor_tools = {
            "print_buffer",
            "find_prev",
            "find_next",
            "goto",
            "patch_current_file",
            "patch_suffix",
            "edit_file",
            "read_file",
            "write_file",
            "finish_editing",
            "insert_before",
            "insert_after",
        }

        # Iterate backwards
        for msg_idx in range(len(messages) - 1, -1, -1):
            msg = messages[msg_idx]

            # Check tool output messages
            if msg.get("role") == "tool":
                tool_name = msg.get("name", "")
                if tool_name in editor_tools:
                    buffer_count += 1
                    if buffer_count > keep_count:
                        # Prune this tool output
                        msg["content"] = f"[PRUNED: Old {tool_name} output]"

            # Check assistant messages with tool calls
            elif msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    func_name = tc.get("function", {}).get("name", "")
                    if func_name in editor_tools:
                        # This will be pruned when its output is pruned
                        pass


# ============================================================================
# Editor-Specific Tools
# ============================================================================
