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
KEEP_OLD_BUFFERS = 5  # Number of tool calls contexts to keep


@dataclass
class EditorEntry:
    path: str


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
        ToolEditor._state[llm_id] = EditorEntry(path)

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
1. You are editing {path}
2. Line numbers are 1-indexed
3. You can switch files using: edit_file <path>
4. When done, use finish_editing(report) to quit editing and continue other tasks(e.g. compilation)
5. All tools except read work only on the current buffer view
6. The current file path is: {path}

Current buffer (starting from line 1):""",
            }
        )

        # Generate response from editor LLM
        result = editor_llm.generate(editor_messages)

        # After editing is done, restore current line tracking for original LLM
        # (editor LLM instance will be cleaned up)

        return {"status": "editing_complete", "file": path, "result": result.content}

    @staticmethod
    def _prune_old_buffers(messages: list[dict], keep_count: int = KEEP_OLD_BUFFERS):
        """Prune old buffer prints and tool calls from messages.

        Iterates messages from last backwards and prunes tool calls and outputs
        that are beyond keep_count from the end.

        This is similar to context/suffix.py pruning but for editor buffers.
        """

        # Track buffer prints from the end
        buffer_count = 0

        # Iterate backwards
        for msg_idx in range(len(messages) - 1, -1, -1):
            msg = messages[msg_idx]

            # Check tool output messages
            if msg.get("role") == "tool":
                tool_name = msg.get("name", "")
                buffer_count += 1
                if buffer_count > keep_count:
                    # Prune this tool output
                    msg["content"] = f"[PRUNED: Old {tool_name} output]"
