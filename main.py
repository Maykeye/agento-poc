#!/usr/bin/env python
from typing import Optional

from llm import LLM
from utils import log_prompt, read_text
import config
import tool_io
import tool_sh


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
        llm.add_tool(tool_io.ToolWriteFile())
        llm.add_tool(tool_io.ToolEditFile())
        llm.add_tool(tool_io.ToolDeleteFile())
        llm.add_tool(tool_io.ToolMkDir())
        llm.add_tool(tool_io.ToolRmDir())
        llm.add_tool(tool_sh.ToolCargoAdd())

    def _llm_reading(self, llm: LLM):
        llm.add_tool(tool_io.ToolReadFile())
        llm.add_tool(tool_io.ToolLs())
        llm.add_tool(tool_sh.ToolCargoCheck())
        llm.add_tool(tool_sh.ToolCargoTest())
        llm.add_tool(tool_sh.ToolGitDiff())
        llm.add_tool(tool_sh.ToolGitStatus())
        llm.add_tool(tool_sh.ToolGitAdd())
        llm.add_tool(tool_sh.ToolRustApiInfo())

    def simple(self, user_prompt: str):
        llm = self.llm
        messages = [
            llm.msg_system("You are a helpful AI assistant, expert rust programmer."),
            llm.msg_user(user_prompt),
        ]
        return self.llm.generate(messages)


def main():
    # init IO
    config.read_config("~/.config/agento.json")
    #  rusto.make_file_readonly("DESIGN.md")

    # read prompt
    prompt = read_text("prompt.md")
    log_prompt(str(config.PROJECT_DIRECTORY), prompt)

    # run
    node = AgencyNode()
    node.simple(prompt)
    tool_sh.rustfmt()


if __name__ == "__main__":
    main()
