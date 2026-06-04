import json
from typing import Annotated

from agento.llm import LLM, FinishGeneration
from agento.tool import Tool
from agento.tool.editor.editor import ToolEditor
from agento.tool.editor.tool_list import EDITOR_TOOLS


class EditorToolFinishEditing(Tool):
    """Exit editing mode with an optional report without making changes."""

    def __init__(self):
        super().__init__(
            name="finish_editing",
            description="""Exit editing mode with a report that will be reported to the main agent.

Use this when you've finished reviewing/editing and want to return to the main LLM.
The report can describe what changes were made or what was observed.""",
        )

    def __call__(
        self,
        report: Annotated[
            str, "Optional report describing what was done during editing session"
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

        path = ToolEditor._state[llm_id].path

        # Clean up editor state
        ToolEditor.reset(llm_id)

        # Return FinishGeneration to stop the editor LLM and return to main LLM
        result_value = {
            "status": "editing_finished",
            "file": path,
            "report": report if report else "No report provided",
            "message": f"Editor mode exited for {path}",
        }
        return FinishGeneration(value=json.dumps(result_value, indent=1))


EDITOR_TOOLS.append(EditorToolFinishEditing)
