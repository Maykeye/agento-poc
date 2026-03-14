#!/usr/bin/env python
import os
from typing import Callable, Optional

from llm import LLM
from utils import read_text, data_tag, commit_files, extract_tag
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
        llm.add_tool(rusto_vfs.rmdir)

    def _llm_reading(self, llm: LLM):
        llm.add_tool(rusto_vfs.read_file)
        llm.add_tool(rusto_vfs.ls)

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
    node.simple(initial)


if __name__ == "__main__":
    main()
