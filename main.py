#!/usr/bin/env python
from typing import Optional

from llm import LLM
from utils import extract_tag, read_text, data_tag
import rusto_vfs


class AgencyNode:
    def __init__(self, initial: str) -> None:
        self._llm: Optional[LLM] = None
        self.initial = initial

    def _llm_initializers(self):
        return [self._llm_reading, self._llm_editing]

    @property
    def llm(self) -> LLM:
        if not self._llm:
            self._llm = self.__init_llm()
        return self._llm

    def __init_llm(self):
        llm = LLM()
        for initializer in self._llm_initializers():
            initializer(llm)
        return llm

    def _llm_editing(self, llm: LLM):
        llm.add_tool(rusto_vfs.write_file)
        llm.add_tool(rusto_vfs.edit_file)
        llm.add_tool(rusto_vfs.delete_file)
        llm.add_tool(rusto_vfs.mkdir)
        llm.add_tool(rusto_vfs.rmdir)
        llm.add_tool(rusto_vfs.cargo_add_crate)

    def _llm_reading(self, llm: LLM):
        llm.add_tool(rusto_vfs.read_file)
        llm.add_tool(rusto_vfs.ls)
        llm.add_tool(rusto_vfs.cargo_check)
        llm.add_tool(rusto_vfs.cargo_test)
        llm.add_tool(rusto_vfs.git_diff)
        llm.add_tool(rusto_vfs.git_status)

    def _simple_system(self):
        return "You are a helpful AI assistant, expert rust programmer."

    def _query_part_initial_task(self):
        return (
            "Glboal initial task to take note of:\n"
            + data_tag("GLOBAL_TASK", self.initial)
            + "\n"
        )

    def _query_current_task(self, text):
        text = self._query_part_initial_task() + "\n"
        text += "The following task is the step that needs to be done towards the global task:\n"
        text += data_tag("TASK_TO_IMPLEMENT_IMMEDIATELY", text)
        return text

    def query(self, text: str):
        return self.simple(self._query_current_task(text))

    def simple(self, user_prompt: str):
        llm = self.llm
        messages = [
            llm.msg_system("You are a helpful AI assistant, expert rust programmer."),
            llm.msg_user(user_prompt),
        ]
        return self.llm.generate(messages)

    def plan_do_review(self, user_request: str):
        """Do a task in the following step:
        * Reader: Prepare plan
        * Writer: Execute plan
        * Reader: Teview the plan
        """
        planner = AiReadNode(self.initial)
        executioner = AiWriteNode(self.initial)
        reviewer = AiReadNode(self.initial)

        the_plan_template = f"""Your role is an AI planner.
Given a user request and access to the source,
prepare high-level design plan of what needs to be done in the current step of implementation. Omit writing low-level source code.
Your design plan will be passed to an agent that will execute it.
Your goal is to "enhance" user request, make decisions that necessary so writer will avoid doing it.

Expected output: write plan in

<PLAN>
The actual plan that you'll make
</PLAN>

Text outside of <PLAN></PLAN> is resereved for your considerations of what to do. <PLAN> will be extracted and passed.
Initial user requesst:
<INITIAL_REQUEST>
{self.initial}
</INITIAL_REQUEST>

Current step we are working on that you must plan:
<CURRENT_STEP>
{user_request}
</CURRENT_STEP>
"""

        executioner_template = f"""Your role is an AI software engineer. You will edit the code given the following:
* Global task. A description of the software we are writing
* Current step. A description of the step that needs to be done
* Plan of implementation. A detailed plan how to execute the the current plan.
    * You are free to sidestep the plan if you believe it is necessary, but try to stick to it

After implementing the current step in accordance to the Current Step, please run `cargo check` and `cargo test` and fix the errors.


Initial global task:
<INITIAL_REQUEST>
{self.initial}
</INITIAL_REQUEST>

Current step we are working on that you must plan:
<CURRENT_STEP>
{user_request}
</CURRENT_STEP>
"""

        reviewer_template = f"""Your role is an AI software engineer. You will review the code given the following:
* Global task. A description of the software we are writing
* Current step. A description of the step that needs to be done
* Plan of implementation. A detailed plan how to execute the the current plan.
    * You are free to sidestep the plan if you believe it is necessary, but try to stick to it
* Results of git_diff (to see what was changed) and git_status(to see what files were added so you'll know what to read)
After implementing the current step in accordance to the Current Step, please run `cargo check` and `cargo test` and fix the errors.
* Feel free to read not mentioned files too but pay less attention to them if they and changes don't touch each other. 

Initial global task:
<INITIAL_REQUEST>
{self.initial}
</INITIAL_REQUEST>

Current step we are working on that you must plan:
<CURRENT_STEP>
{user_request}
</CURRENT_STEP>

The plan
<PLAN_OF_CURRENT_STEP>
%%%PLAN
</PLAN_OF_CURRENT_STEP>

Expected output: write review in

<REVIEW>
[DECISION]
Your review commentary
</REVIEW>

where [DECISION] is either `[ACCEPT]` (not quoted) or `[REJECT]`, e.g. (not quoted)

<REVIEW
[ACCEPT]
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum
</REVIEW>

Your review will be given to the code writer."""

        the_plan = planner.simple(the_plan_template).content
        the_plan = extract_tag(the_plan_template, "PLAN")
        review = ""

        while True:
            for_executioner = executioner_template + f"""The plan:
<PLAN_OF_CURRENT_STEP>
{the_plan}
</PLAN_OF_CURRENT_STEP>
 """
            if review:
                for_executioner += f"""The old implementation was rejected for the following reasons that ough to be fixed:
<REVIEW>
{review}
</REVIEW>
            """
            executioner.simple(for_executioner)
            review = reviewer.simple(
                reviewer_template.replace("%%%PLAN", the_plan)
            ).content.strip()
            if not review.startswith("[REJECT]"):
                break


class AiReadNode(AgencyNode):
    def _llm_initializers(self):
        return [self._llm_reading]


class AiWriteNode(AgencyNode):
    def _llm_initializers(self):
        return [self._llm_reading, self._llm_editing]


def main():
    node = AgencyNode(read_text("./prompts/initial-game-idea.txt"))
    node.plan_do_review("""

Focus on: tests/test_utils.rs: fn test_random_weighted_zero_weight()

Right now it checks only one combination of weights: vec![0, 1, 1_000_000].
Change it to three arrays + three chgecks:
* [1_000_000, 0, 1]
* [0, 1_000_000, 1]
* [0, 1, 1_000_000]

with corresponsing expected result

""")


if __name__ == "__main__":
    main()
