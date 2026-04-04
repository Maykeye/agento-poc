#!/usr/bin/env python
from pathlib import Path
from typing import Optional

from context.context import context_handler
from llm import LLM
import tool_fork
from utils import log_prompt, expand_file
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
        sys = []
        if self.lang == "rpg":
            sys = [self._llm_rpg]

        elif self.readonly:
            sys = [self._llm_reading]
        else:
            sys = [self._llm_reading, self._llm_editing]
        sys += [self._llm_fold]
        return sys

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
        llm.add_tool(tool_sh.ToolGitAdd())
        if self.lang == "rust":
            llm.add_tool(tool_sh.ToolCargoAdd())

    def _llm_fold(self, llm: LLM):
        if context_handler().mode() == ContextMode.SUFFIX:
            llm.add_tool(tool_io.ToolFoldAdd())
            llm.add_tool(tool_io.ToolUnfold())
            llm.add_tool(tool_io.ToolUnfoldAll())

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
        llm.add_tool(tool_fork.ToolFork())

    def simple(self, user_prompt: str):
        llm = self.llm
        messages = [
            llm.msg_system("You are a helpful AI assistant, expert rust programmer."),
            llm.msg_user(user_prompt),
        ]
        return self.llm.generate(messages)


def main():
    set_context_mode(ContextMode.SUFFIX)  # TODO: add to @config?

    # Parse initial prompt file
    assert len(sys.argv) == 2
    filename = sys.argv[1]
    assert Path(filename).resolve().is_file()

    # Example of preventing editing DESIGN.md
    # config.make_file_readonly("DESIGN.md")
    # TODO: use it in prompt.md as @readonly file?

    # read prompt
    prompt = expand_file(filename)
    assert config.project_directory().is_dir(), "set @project_dir in prompt"

    log_prompt(str(config.project_directory()), prompt)

    # run
    lang = config.guess_project_language()
    print(f"{lang=}")
    node = AgencyNode(lang=lang)
    node.simple(prompt)
    if node.lang == "rust":
        tool_sh.rustfmt()


if __name__ == "__main__":
    main()
