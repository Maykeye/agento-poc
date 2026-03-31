#!/usr/bin/env python
from typing import Optional

from llm import LLM
import tool_fork
from utils import log_prompt, read_text
import config
import tool_io
import tool_rpg
import tool_sh
import sys
from context import set_context_mode, ContextMode


class AgencyNode:
    def __init__(self, read_only=False, lang: str = "rust") -> None:
        self._llm: Optional[LLM] = None
        self.readonly = read_only
        self.lang = lang
        assert self.lang in ["rust", "py", "js", "nul", "null", "rpg"]

    def _llm_initializers(self):
        if self.lang == "rpg":
            return [self._llm_rpg]

        if self.readonly:
            return [self._llm_reading]
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

    def _llm_rpg(self, llm: LLM):
        llm.add_tool(tool_io.ToolReadFile())
        llm.add_tool(tool_io.ToolEditFile())
        llm.add_tool(tool_io.ToolWriteFile())
        llm.add_tool(tool_io.ToolAppend())
        llm.add_tool(tool_rpg.ToolRollDice())
        llm.add_tool(tool_rpg.ToolRollCheck())
        llm.add_tool(tool_rpg.ToolRollVersus())

    def _llm_editing(self, llm: LLM):
        llm.add_tool(tool_io.ToolWriteFile())
        llm.add_tool(tool_io.ToolEditFile())
        llm.add_tool(tool_io.ToolDeleteFile())
        llm.add_tool(tool_io.ToolMkDir())
        llm.add_tool(tool_io.ToolRmDir())
        if self.lang == "rust":
            llm.add_tool(tool_sh.ToolCargoAdd())

    def _llm_reading(self, llm: LLM):
        llm.add_tool(tool_io.ToolReadFile())
        llm.add_tool(tool_io.ToolLs())
        if self.lang == "rust":
            llm.add_tool(tool_sh.ToolCargoCheck())
            llm.add_tool(tool_sh.ToolCargoClippy())
            llm.add_tool(tool_sh.ToolCargoTest())
            llm.add_tool(tool_sh.ToolRustApiInfo())
        if self.lang == "py":
            llm.add_tool(tool_sh.ToolPythonUnittest())
        if self.lang == "js":
            llm.add_tool(tool_sh.ToolPupeeter())
        llm.add_tool(tool_sh.ToolGitDiff())
        llm.add_tool(tool_sh.ToolGitStatus())
        llm.add_tool(tool_sh.ToolGitAdd())
        llm.add_tool(tool_fork.ToolFork())

    def simple(self, user_prompt: str):
        llm = self.llm
        messages = [
            llm.msg_system("You are a helpful AI assistant, expert rust programmer."),
            llm.msg_user(user_prompt),
        ]
        return self.llm.generate(messages)


def main():
    set_context_mode(ContextMode.SUFFIX)

    # init IO
    config.read_config("~/.config/agento.json")

    # Example of preventing editing DESIGN.md
    # rusto.make_file_readonly("DESIGN.md")

    # read prompt
    intro = read_text("intro.md", "")
    prompt = read_text("prompt.md")
    prompt = f"{intro}\n\n{prompt}".strip()
    log_prompt(str(config.project_directory()), prompt)

    # TODO: normal opt parse
    read_only = sys.argv[1:] == ["--read-only"]
    if not read_only and len(sys.argv) > 1:
        raise ValueError("invalid opts. Use --read-only for R/O")

    # run
    print("Read only:", read_only)
    node = AgencyNode(read_only=read_only, lang="rpg")
    node.simple(prompt)
    if node.lang == "rust":
        tool_sh.rustfmt()


if __name__ == "__main__":
    main()
