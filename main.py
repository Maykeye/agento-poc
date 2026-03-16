#!/usr/bin/env python
from typing import Optional

from llm import LLM
from utils import read_text
import rusto
from utils import log_prompt


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
        llm.add_tool(rusto.ToolWriteFile())
        llm.add_tool(rusto.ToolEditFile())
        llm.add_tool(rusto.ToolDeleteFile())
        llm.add_tool(rusto.ToolMkDir())
        llm.add_tool(rusto.ToolRmDir())
        llm.add_tool(rusto.ToolCargoAdd())

    def _llm_reading(self, llm: LLM):
        llm.add_tool(rusto.ToolReadFile())
        llm.add_tool(rusto.ToolLs())
        llm.add_tool(rusto.ToolCargoCheck())
        llm.add_tool(rusto.ToolCargoTest())
        llm.add_tool(rusto.ToolGitDiff())
        llm.add_tool(rusto.ToolGitStatus())
        llm.add_tool(rusto.ToolRustApiInfo())

    def simple(self, user_prompt: str):
        llm = self.llm
        messages = [
            llm.msg_system("You are a helpful AI assistant, expert rust programmer."),
            llm.msg_user(user_prompt),
        ]
        return self.llm.generate(messages)


def main():
    # init IO
    rusto.read_config("~/.config/agento.json")
    rusto.make_file_readonly("DESIGN.md")

    # read prompt
    prompt = read_text("prompt.txt")
    log_prompt(str(rusto.PROJECT_DIRECTORY), prompt)

    # run
    node = AgencyNode()
    node.simple(prompt)
    rusto.rustfmt()


if __name__ == "__main__":
    main()
