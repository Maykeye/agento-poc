#!/usr/bin/env python

import context
from context import context_handler
from context.context_handler import ContextMode
from context.suffix import SUFFIX_CONTEXTS
import tool_io
from tool_edit_patch import ToolEditDiffPatch
from pathlib import Path
import os
import config
from dataclasses import dataclass
from llm import LLM, LlmInstace


@dataclass
class PatchCombo:
    orig_path: str
    patch: str


def fix_headers():
    res = []
    for p in Path(".").glob("*.patch.*"):
        orig = str(p).replace(".patch.", ".orig.")
        assert Path(orig).exists(), orig
        lines = Path(p).read_text().splitlines()
        lines[0] = f"--- a/{orig}"
        lines[1] = f"+++ b/{orig}"
        text = "\n".join(lines) + "\n"
        res.append(PatchCombo(orig, text))
    return res


def init_llm():
    llm = LLM()
    LLM.INSTANCES.clear()
    LLM.INSTANCES.append(LlmInstace(llm=llm, messages=[llm.msg_system("ai agent")]))
    llm.add_tool(tool_io.ToolReadFile())
    return LLM.INSTANCES[0].llm


def apply(patches: list[PatchCombo]):
    llm = init_llm()
    for p in patches:
        tool_io.ToolReadFile()(p.orig_path)
        context_handler().prepare_current_llm(llm)
        result = ToolEditDiffPatch()(p.orig_path, p.patch)
        print(result)


def run_all():
    patches = fix_headers()
    apply(patches)


def main():
    p = PatchCombo("focus.orig", Path("focus.patch").read_text())
    apply([p])


if __name__ == "__main__":
    context.set_context_mode(ContextMode.SUFFIX)
    os.chdir("./tests/dat")
    config.set_project_directory(".")
    main()
