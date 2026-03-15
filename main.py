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
        doer = AiWriteNode(self.initial)
        reviewer = AiReadNode(self.initial)

        the_plan_template = read_text("./prompts/pdr_planner.txt").format(
            initial=self.initial, user_request=user_request
        )

        doer_template = read_text("./prompts/pdr_doer.txt").format(
            initial=self.initial, user_request=user_request
        )

        review_template = read_text("./prompts/pdr_review.txt").format(
            initial=self.initial, user_request=user_request
        )

        the_plan = planner.simple(the_plan_template).content
        the_plan = extract_tag(the_plan_template, "PLAN")

        review = ""
        while True:
            for_doer = doer_template + f"""The plan:
<PLAN_OF_CURRENT_STEP>
{the_plan}
</PLAN_OF_CURRENT_STEP>
 """
            if review:
                for_doer += f"""The old implementation was rejected for the following reasons that ough to be fixed:
<REVIEW>
{review}
</REVIEW>
            """
            doer.simple(for_doer)
            review = reviewer.simple(
                review_template.replace("%%%PLAN", the_plan)
            ).content.strip()
            review = extract_tag(review, "REVIEW")
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

    rusto_vfs.make_file_readonly("DESIGN.md")
    node.plan_do_review("""
Let's change ore density definition
Replace tuple with range
""")


if __name__ == "__main__":
    main()
