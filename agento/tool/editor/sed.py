# File with sed tool that runs sed on a file
import copy
from pathlib import Path
from typing import Annotated, Union

from agento.tool import Tool, run_executable
from agento.config import real_path
from agento.llm import LLM
from agento.tool.editor.editor import ToolEditor
from agento.utils import extract_tag, diff_gen
from agento.tool.editor.tool_list import EDITOR_TOOLS


def _initial_check(path: str) -> Union[tuple[None, dict], tuple[Path, None]]:
    # Get current LLM instance
    if not LLM.INSTANCES:
        return None, {"error": "No LLM instance available"}

    p = real_path(path)
    if not p.exists():
        return None, {path: "error", "error": f"File {path} doesn't exist"}
    if not p.is_file():
        return None, {path: "error", "error": f"Path {path} is not a file"}

    llm = LLM.INSTANCES[-1].llm
    llm_id = id(llm)

    # Get current state
    if llm_id not in ToolEditor._state:
        return None, {"error": "No file being edited"}
    return p, None


class EditorToolSedEdit(Tool):
    def __init__(self):
        super().__init__(
            name="sed_edit",
            description="Run `sed` on a file with given script to edit file. Then queries subagent if result is appropriate and sed outputs should be written to a file. This way you can edit file(by accepting changes) or just query it(by rejecting). Highly recommended to use this tool for editing text over any other.",
        )
        self.debug_assumed_decision = ""

    def __call__(
        self,
        path: Annotated[
            str,
            "Project path to the file to process. Must exactly match the current editing file to confirm intention is to edit it",
        ],
        script: Annotated[
            str,
            "sed script/expression to apply.",
        ],
    ):
        p, err = _initial_check(path)
        if err is not None:
            return err
        assert p is not None

        llm_id = id(LLM.INSTANCES[-1].llm)

        current_editing_path = ToolEditor._state[llm_id].path
        editing_path = real_path(current_editing_path)

        # Normalize paths for comparison (resolve to absolute paths)
        if p != editing_path:
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
        diff_text = "\n".join(diff_gen(original_text, sed_output, path))

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
You are a subagent given a result of applying sed in form of diff in a message below.
Decide if result should be saved. If diffs looks like appropriate changes that need to be saved to the file, your reply should include

<CONFIRMATION>
APPLY
... (your brief reasoning) ...
</CONFIRMATION>

If script changed too something wrong or there were not intention to save, respond

<CONFIRMATION>
DISCARD
... (your brief reasoning) ...
</CONFIRMATION>

Your response should contain exactly one CONFIRMATION tag.
Your response should contain exactly zero tool calls (you have no tools at this context).
If there is no CONFIRMATION tag with ACCEPT as its first line, changes will be discarded.

Now, here is `diff -u` between original file and result of sed
<DIFF>
{diff}
</DIFF>""")
            fork_messages.append(intro)
            r = llm.generate(fork_messages)
            message = r.content
        else:
            message = self.debug_assumed_decision
        confirmation = extract_tag(message, "CONIFRMATION")
        confirmation = confirmation.removeprefix("<CONFIRMATION>")
        confirmation = confirmation.removesuffix("</CONFIRMATION>")
        return confirmation


class EditorToolSedQuery(Tool):
    def __init__(self):
        super().__init__(
            name="sed_query",
            description="Run `sed -n` on a file with given script to query state of the file. No changes will be saved, but sed output will be returned as-is. Use for commands like `p` command, no changes will be saved",
        )
        self.debug_assumed_decision = ""

    def __call__(
        self,
        path: Annotated[
            str,
            "Project path to the file to process. Must exactly match the current editing file to confirm intention is to edit it",
        ],
        script: Annotated[
            str,
            "sed script/expression to apply.",
        ],
    ):
        p, err = _initial_check(path)
        if err is not None:
            return err
        assert p is not None

        llm_id = id(LLM.INSTANCES[-1].llm)
        current_editing_path = ToolEditor._state[llm_id].path
        editing_path = real_path(current_editing_path)

        # Normalize paths for comparison (resolve to absolute paths)
        if p != editing_path:
            return {
                "error": f"Path mismatch",
                "suggestion": f"You are editing '{current_editing_path}' but tried to edit to '{path}'. Use the correct path or use finish_editing() to exit and write_file() for a different file.",
                "editing_file": current_editing_path,
                "requested_path": path,
            }

        original_text = p.read_text()

        # Run sed with the file content as stdin
        result = run_executable(["sed", "-n", script], stdin_text=original_text)
        sed_output = result.get("stdout", "")
        return sed_output


EDITOR_TOOLS.append(EditorToolSedEdit)
EDITOR_TOOLS.append(EditorToolSedQuery)
