#!/usr/bin/env python
import os
from typing import Callable, Optional

from llm import LLM
from utils import read_text, data_tag, commit_files, extract_tag
import markdown_edit


class AgencyNode:
    def __init__(self, initial: str) -> None:
        self.initial = initial
        self._llm: Optional[LLM] = None

    def _llm_initializers(self):
        return [self._llm_reading]

    def reset_plan(self, path_load_from: Optional[str] = None):
        if path_load_from:
            markdown_edit.from_json_file(path_load_from)
        else:
            markdown_edit.reset()

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
        llm.add_tool(markdown_edit.create_document)
        llm.add_tool(markdown_edit.change_document_parent)
        llm.add_tool(markdown_edit.change_document_content)
        llm.add_tool(markdown_edit.change_document_title)
        llm.add_tool(markdown_edit.delete_document)

    def _llm_reading(self, llm: LLM):
        llm.add_tool(markdown_edit.list_documents)
        llm.add_tool(markdown_edit.read_document)


class DesignPlanNode(AgencyNode):
    def __init__(self, initial: str) -> None:
        super().__init__(initial)
        self.system = read_text("./prompts/sys_planner.txt")

    def _llm_initializers(self):
        return super()._llm_initializers() + [self._llm_editing]

    def user(self, initial: str, old_plan_review: str | None):
        plan = data_tag("INITIAL_REQUIREMENTS", initial)
        if old_plan_review:
            plan += f'\n{data_tag("REVIEW_DECISION", old_plan_review)}'
            plan += f"\n\nPlease edit a plan to address the reviewer's feedback."
        return plan

    def __call__(self, review_raw_text=None):
        self.reset_plan("PLAN.json" if review_raw_text else None)

        messages = [
            self.llm.msg_system(self.system),
            self.llm.msg_user(self.user(self.initial, review_raw_text)),
        ]

        self.llm.generate(messages)
        text_md = markdown_edit.to_markdown()
        text_json = markdown_edit.to_json_string()
        commit_files(
            "New/updated plan",
            {
                "PLAN.json": text_json,
                "PLAN.md": text_md,
            },
        )


class ReviewNode(AgencyNode):
    def __init__(self, initial: str) -> None:
        super().__init__(initial)
        self.system = read_text("./prompts/sys_review.txt")

    def user(self, initial: str):
        return data_tag("INITIAL_REQUIREMENTS", initial)

    def __call__(self):
        markdown_edit.from_json_file("PLAN.json")
        messages = [
            self.llm.msg_system(self.system),
            self.llm.msg_user(self.user(self.initial)),
        ]
        review = self.llm.generate(messages).content
        review = extract_tag(review, "REVIEW")
        commit_files("Review of the plan", {"REVIEW.md": review})
        return review


class IterateWhileNode:
    def __init__(
        self, condition: Callable, then: Callable, max_steps: Optional[int] = None
    ) -> None:
        self.condition = condition
        self.then = then
        self.max_steps = max_steps

    def __call__(self):
        step = 0
        while self.max_steps is None or step < self.max_steps:
            step += 1
            if not self.condition():
                break
            self.then()


class IterativelyReviewNode(AgencyNode):
    def __init__(self, initial: str, max_steps=None) -> None:
        super().__init__(initial)
        self.reviewer = ReviewNode(initial)
        self.planner = DesignPlanNode(initial)
        self.loop = IterateWhileNode(self.__condition, self.__body, max_steps)
        self.last_review = None

    def __condition(self):
        self.last_review = self.reviewer()
        return self.last_review.startswith("[REJECT]")

    def __body(self):
        return self.planner(self.last_review)

    def __call__(self):
        return self.loop()


def main():
    initial = "Let's write mining game in rust using ratatui: the deeper we move, the harder ore there is. But more valuable resources there are. Ideas to play around: we can rewind time to next day/planet to reset terrain, buy dynamites, get new pickaxes, buy new weapons. Genre is mining, nothing more. Dig, dig, dig. Game ideas is up to you, define them in the PLAN as well and refine it until Agency accepts it."
    reviewer = IterativelyReviewNode(initial)
    os.chdir("./.planning")

    reviewer.planner("""[REJECT]
Changes to be done:

Overall:

    * Remove combat. Weapons are used for mining only. (They have different AoE, speed, damage to terrain, etc)

In # 2. Directory structure:

    * Do not include lib.rs -- this is excetuable only
    * Rendering error: game.rs/world.rs/etc are placed into unnamed subdirectory of `src/`
    * `ui/` is rendered as 

    ```
    │   ├──
    │   │   ├── ui/
        ... etc
    ```

    fix it 

    ```
    │   ├── ui/
    │   │   ├── menu.rs      # Main menu, settings
        ... etc
    ```

    * Remove `tests/integration.rs`. This implementation detail is low-level, it will be considered later.

""")

    # reviewer()


if __name__ == "__main__":
    main()
