from config import real_path
from llm import LLM
from tool import Tool
from tool.editor.editor import ToolEditor
from tool.editor.tool_list import EDITOR_TOOLS


class EditorToolPrint(Tool):
    """Print current buffer view (called implicitly after navigation/edit operations)"""

    def __init__(self):
        super().__init__(
            name="print_buffer",
            description="Print current buffer view (called implicitly after navigation/edit operations)",
        )

    def __call__(self):
        """Print current buffer view.

        This is called implicitly after navigation/edit operations to show the current state.
        """

        # Get current LLM instance
        if not LLM.INSTANCES:
            return {"error": "No LLM instance available"}

        llm = LLM.INSTANCES[-1].llm
        llm_id = id(llm)

        # Get current state
        if llm_id not in ToolEditor._state:
            return {"error": "Not in editor mode"}

        state = ToolEditor._state[id(llm)]
        p = real_path(state.path)
        text = p.read_text()

        # Print buffer from current line
        buffer_output = ToolEditor._format_buffer(state.path, state.current_line, text)

        return buffer_output


EDITOR_TOOLS.append(EditorToolPrint)
