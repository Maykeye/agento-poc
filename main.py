#!/usr/bin/env python
from typing import Optional

from llm import LLM
from utils import read_text, data_tag
import rusto_vfs


class AgencyNode:
    def __init__(self) -> None:
        self._llm: Optional[LLM] = None

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
        llm.add_tool(rusto_vfs.cargo_add)

    def _llm_reading(self, llm: LLM):
        llm.add_tool(rusto_vfs.read_file)
        llm.add_tool(rusto_vfs.ls)
        llm.add_tool(rusto_vfs.cargo_check)

    def simple(self, user_prompt: str):
        llm = self.llm
        messages = [
            llm.msg_system("You are a helpful AI assistant, expert rust programmer."),
            llm.msg_user(user_prompt),
        ]
        self.llm.generate(messages)


def main():
    initial = read_text("./prompts/initial-game-idea.txt")
    node = AgencyNode()
    node.simple(
        data_tag("PAST_ORIGINAL_TASK", initial)
        + "\n"
        + data_tag(
            "IMPLEMENT_TASK",
            """
Preparation:
* Read the current DESIGN.md to know what project about and common style for e.g. testsing.

* New task:
* utils.rs: Create weighted sampler
    * Typedef "Xoshiro128PlusPlus" as Rng.
    * Create a new function `random_weighted(rng: &mut Rng, weights: &[usize]) -> usize`
        that returns index of random value depending on weights
            E.g. if there are three weights, weights=vec!(0,1,10), then value=0 will not be returned, value=2 will be returned 10 times more often than 1.
""",
        )
    )


if __name__ == "__main__":
    main()
