from copy import Error
from typing import Annotated
from tool import Tool
from llm import LLM
from utils import extract_tag


class ToolFork(Tool):
    def __init__(self):
        super().__init__(
            "fork",
            """
A tool to fork task into separate context.
Purpose of this tool is to save context size. It must be used this way:
Let say you need to implement or edit foo(), bar(), then check the result. 
To do so call do the following sequence:
1) call fork tool with instruction to implement foo()
2) call fork tool with instruction to implement bar()
3) call fork tool with the instruction to check the result.

After fork ends the task, it will report success or failure of its task so you can proceed without seeing the actions performed by fork.

Each time fork starts, it will receives the context so far, plus instruction of the subtask you passed. After subtask is complete, it will report the result and quit. You will see brief report only.
""".strip(),
        )

    def __call__(self, instruction: Annotated[str, "Instruction for the forked llm"]):
        assert LLM.INSTANCES, "Must see indexes"

        llm = LLM.INSTANCES[-1].llm.clone()
        messages = LLM.INSTANCES[-1].messages
        assert messages, "Messages must have tool call"
        assert llm.tool_calls_id, "tool calls must exist"
        messages.append(
            {
                "role": "tool",
                "tool_call_id": llm.tool_calls_id[-1],
                "name": self.name,
                "content": f"""
You are a fork provided the following instruction:
<SUBTASK>
{instruction}
</SUBTASK>
Perform the subtask, taking the context above into account. Do NOT perform anything beyond steps required for the subtask.
After performing the substack, report the result in
<SUBTASK_REPORT>
(brief report which will be reported to caller)
</SUBTASK_REPORT>
Do not provide anything in the reply beside the SUBTASK_REPORT, as such information will be discarded.
""".strip(),
            }
        )

        print("FORK START: Started generating")
        res = llm.generate(messages)
        report = extract_tag(res.content, "SUBTASK_REPORT")
        ans = {"subtask": instruction, "report": report}
        print("FORK DONE:", ans)
        return ans


if __name__ == "__main__":
    llm = LLM()
    llm.add_tool(ToolFork())
    llm.generate(
        [
            llm.msg_user(
                """We are testing tool fork. Please calculate 3 factorial using it, e.g. call fork to calc factorial(3), from there fork to cal factorial(2), fork all fork way to calculate factorial(1), from that fork return 1."""
            )
        ]
    )
