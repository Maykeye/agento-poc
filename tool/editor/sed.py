# File with sed tool that runs sed on a file
import copy
from typing import Annotated
from tool import Tool, run_executable
from config import real_path
from llm import LLM
from tool.editor.editor import ToolEditor
from utils import extract_tag, diff_gen
from tool.editor.tool_list import EDITOR_TOOLS


class EditorToolSed(Tool):
    def __init__(self):
        super().__init__(
            name="sed",
            description="Run sed on a file with given script. Then queries subagent if result is appropriate and sed outputs should be written to a file. This way you can edit file(by accepting changes) or just query it(by rejecting). Highly recommended to use this tool over any other",
        )
        self.debug_assumed_decision = ""

    def __call__(
        self,
        path: Annotated[
            str,
            "Project path to the file to process. Must exactly match the current editing file to confirm intention is to edit it",
        ],
        script: Annotated[str, "sed script/expression to apply"],
    ):
        p = real_path(path)

        if not p.exists():
            return {path: "error", "error": f"File {path} doesn't exist"}
        if not p.is_file():
            return {path: "error", "error": f"Path {path} is not a file"}

        # Get current LLM instance
        if not LLM.INSTANCES:
            return {"error": "No LLM instance available"}

        llm = LLM.INSTANCES[-1].llm
        llm_id = id(llm)

        # Get current state
        if llm_id not in ToolEditor._state:
            return {"error": "No file being edited"}

        current_editing_path = ToolEditor._state[llm_id].path
        provided_path = real_path(path)
        editing_path = real_path(current_editing_path)

        # Normalize paths for comparison (resolve to absolute paths)
        if provided_path != editing_path:
            return {
                "error": f"Path mismatch",
                "suggestion": f"You are editing '{current_editing_path}' but tried to edit to '{path}'. Use the correct path or use finish_editing() to exit and write_file() for a different file.",
                "editing_file": current_editing_path,
                "requested_path": path,
            }

        original_text = p.read_text()

        # Run sed with the file content as stdin
        result = run_executable(["sed", script], stdin_text=original_text)

        if result.get("exitcode", -1) != 0:
            return {
                path: "error",
                "error": f"sed failed with exitcode {result.get('exitcode')}",
                "stderr": result.get("stderr", ""),
            }

        sed_output = result.get("stdout", "")

        # Generate diff -u between original and sed output
        diff_text = "".join(diff_gen(original_text, sed_output, path))

        accept, decision = self.get_decision(diff_text)
        if accept:
            p.write_text(sed_output)
            return {
                "path": path,
                "result": "applied",
                "decision": decision,
                "diff": diff_text if diff_text else "(no changes)",
            }
        else:
            return {
                "path": path,
                "result": "discarded",
                "decision": decision,
                "diff": diff_text if diff_text else "(no changes)",
            }

    def get_decision(self, diff: str):
        decision = self.query_decision(diff).strip()
        lines = decision.splitlines()
        accept = False
        if lines and lines[0]:
            accept = lines[0].upper().startswith("APPLY")
        return (accept, decision)

    def query_decision(self, diff: str):
        if not self.debug_assumed_decision:
            original_messages = LLM.INSTANCES[-1].messages
            llm = LLM.INSTANCES[-1].llm.clone()
            llm.tools.clear()
            fork_messages = copy.deepcopy(original_messages)

            intro = llm.msg_user(f"""[SYSTEM OVERRIDE: SED CONFIRMATION]
    You will be given a result of applying sed in form of diff in a message below.
    Decide if result should be saved. If diffs looks like appropriate changes that need to be saved to the file,
    your response should include

    <CONFIRMATION>
    APPLY
    ... (your brief reasoning) ...
    </CONFIRMATION>

    If script changed too something wrong or there were not intention to save, respond
    <CONFIRMATION>
    DISCARD
    ... (your brief reasoning) ...
    </CONFIRMATION>

    Your response should contain exactly one CONFIRMATION tag. If there is no CONFIRMATION tag with ACCEPT as its first line, changes will be discarded.

    Now, here is `diff -u` between original file and result of sed
    <DIFF>
    {diff}
    </DIFF>
        """.strip())
            fork_messages.append(intro)
            r = llm.generate(fork_messages)
            message = r.content
        else:
            message = self.debug_assumed_decision
        confirmation = extract_tag(message, "CONIFRMATION")
        confirmation = confirmation.removeprefix("<CONFIRMATION>")
        confirmation = confirmation.removesuffix("</CONFIRMATION>")
        return confirmation


EDITOR_TOOLS.append(EditorToolSed)
