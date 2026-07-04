#!/usr/bin/env python
import os
import sys
import time
from typing import Optional

from agento.config import CONFIG
from agento.llm import LLM
from agento.utils import expand_file, format_duration
from agento.utilsql import log_prompt
from agento.tool import Tool, io as tool_io
import agento.tool as tool
import agento.tool.rpg as tool_rpg
import agento.tool.fork as tool_fork
import agento.tool.sh as tool_sh
from agento.context import set_context_mode, ContextMode


class AgencyNode:
    def __init__(self, read_only=False, lang: str = "rust") -> None:
        self._llm: Optional[LLM] = None
        self.readonly = read_only
        self.lang = lang
        CONFIG.language = self.lang
        assert self.lang in ["rust", "py", "js", "none", "rpg"]

    def _llm_initializers(self):
        if self.lang == "none":
            print("(Skipped all tools)")
            return []
        sys = []
        if self.readonly:
            sys = [self._llm_reading]
        else:
            sys = [self._llm_reading, self._llm_editing]
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
        if os.getenv("LLAMA_AGENTO_VERBOSE"):
            llm.add_tool(tool.ToolDebugPing())
            llm.add_tool(tool.ToolDebugEcho())
            llm.add_tool(tool.ToolDebugAdd())
        for external_tool in CONFIG.external_tools:
            assert isinstance(external_tool, Tool)
            llm.add_tool(external_tool)
        return llm

    def _llm_editing(self, llm: LLM):
        llm.add_tool(tool_io.ToolDeleteFile())
        llm.add_tool(tool_io.ToolRename())
        llm.add_tool(tool_io.ToolSearchReplaceOnce())
        llm.add_tool(tool_io.ToolMkDir())
        llm.add_tool(tool_io.ToolRmDir())
        llm.add_tool(tool_sh.ToolGitAdd())
        llm.add_tool(tool_io.ToolAppend())
        llm.add_tool(tool_sh.ToolBash())
        llm.add_tool(tool_io.ToolWriteFile())
        if self.lang == "rust":
            llm.add_tool(tool_sh.ToolCargoAdd())

    def _llm_reading(self, llm: LLM):
        llm.add_tool(tool_io.ToolReadFile())
        llm.add_tool(tool_io.ToolLs())
        if self.lang == "rust":
            llm.add_tool(tool_sh.ToolCargoCheck())
            llm.add_tool(tool_sh.ToolCargoClippy())
            llm.add_tool(tool_sh.ToolCargoTest())
            # llm.add_tool(tool.sh.ToolRustApiInfo()) # disabled as I no longer use it in pdoman
        if self.lang == "py":
            llm.add_tool(tool_sh.ToolPythonUnittest())
        if self.lang == "js":
            llm.add_tool(tool_sh.ToolPupeeter())
        llm.add_tool(tool_sh.ToolGitDiff())
        llm.add_tool(tool_sh.ToolGitStatus())
        llm.add_tool(tool_sh.ToolAck())
        llm.add_tool(tool_fork.ToolFork())
        llm.add_tool(tool_fork.ToolForkTemplate())

    def simple(self, user_prompt: str):
        llm = self.llm
        messages = [
            llm.msg_system("You are a helpful AI assistant, expert rust programmer."),
            llm.msg_user(user_prompt),
        ]
        return self.llm.generate(messages)


def main():
    start = time.monotonic()
    set_context_mode(ContextMode.RAW)  # TODO: add to @config?

    # Parse initial prompt file
    assert len(sys.argv) == 2
    filename = sys.argv[1]

    # Example of preventing editing DESIGN.md
    # CONFIG.make_file_readonly("DESIGN.md")
    # TODO: use it in prompt.md as @readonly file?

    # read prompt
    CONFIG.language = CONFIG.guess_project_language()
    prompt = expand_file(filename)
    assert CONFIG.project_directory.is_dir(), "set @project_dir in prompt"
    # sync tools
    tools = CONFIG.external_tools
    assert all(isinstance(tool, Tool) for tool in tools)
    assert len(tools) == len(set([t.name for t in tools])), "non-unique names"
    print(
        f"Extern tools: {len(tools)}",
        [t.name for t in tools],
    )

    log_prompt(str(CONFIG.project_directory), prompt)

    # run
    node = AgencyNode(lang=CONFIG.language)
    node.simple(prompt)
    if node.lang == "rust":
        tool_sh.rustfmt()
    end = time.monotonic()

    # Report
    print(f"Elapsed time: {format_duration(end - start)}")


if __name__ == "__main__":
    main()
