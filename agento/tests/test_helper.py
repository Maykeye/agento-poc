import unittest
import os
from pathlib import Path
from typing import Any, Callable, Optional
import shutil

from agento.config import CONFIG
from agento.tool import Tool
from agento.tool import io as tool_io
from agento import utilsql
from agento.context import context_handler, ContextMode, set_context_mode
from agento.llm import LLM, LlmInstace
from agento.utils import TEMP_DIR


def tmpfilename(name: str) -> Path:
    return Path(f"{TEMP_DIR}/{name}")


class TestBase(unittest.TestCase):
    FILE_FOO = tmpfilename(".agento.demo.foo")
    FILE_BAR = tmpfilename(".agento.demo.bar")
    FILE_TEST = tmpfilename(".agento.editor.test")

    def init_test_llm(self):
        """Initialize LLM for the test."""
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Test operations"))
        return dummy_llm, msgs

    def setUp(self):
        if Path(tmpfilename("")).exists():
            assert Path(tmpfilename("")).is_dir()
            shutil.rmtree(tmpfilename(""))
        Path(tmpfilename("")).mkdir(parents=True, exist_ok=True)
        set_context_mode(ContextMode.RAW)
        os.chdir(tmpfilename(""))
        CONFIG.project_directory = tmpfilename("")
        CONFIG.logging_sqlite_path = Path(":memory:")
        utilsql.reset_all_caches()
        LLM.INSTANCES.clear()
        self.FILE_FOO.write_text("foo\ntext")
        self.FILE_BAR.write_text("bar\nvalue")

    def tearDown(self):
        self.FILE_FOO.unlink(True)
        self.FILE_BAR.unlink(True)

    def tool_call(self, tool: Tool, **kwargs):
        context_handler().prepare_current_llm(LLM.INSTANCES[-1].llm)
        args = {k: v.name if isinstance(v, Path) else v for (k, v) in kwargs.items()}
        msgs = LLM.INSTANCES[-1].messages
        LLM.INSTANCES[-1].llm.append_tool_call(tool.name, **args)
        res = tool(**args)  # type: ignore
        return self.append_tool_call_result(tool.name, msgs, res)

    def tool_call_read(self, path: Path) -> Any:
        return self.tool_call(tool_io.ToolReadFile(), path=path)

    def tool_call_rename(self, src: Path, dst: Path) -> Any:
        return self.tool_call(tool_io.ToolRename(), path_src=src, path_dst=dst)

    def tool_call_write(self, path: Path, text: str) -> Any:
        return self.tool_call(tool_io.ToolWriteFile(), path=path, text=text)

    def tool_call_delete_foo(self) -> Any:
        return self.tool_call(tool_io.ToolDeleteFile(), path=self.FILE_FOO)

    def tool_call_editor_append(self, path: Path, text: str) -> Any:
        return self.tool_call(tool_io.ToolAppend(), path=path, text=text)

    def tool_call_with_check(
        self,
        tool_func: Callable,
        tool_args: tuple = (),
        expected_ctx_id: Optional[int] = None,
        check_items: Optional[list[str]] = None,
    ) -> Any:
        """Helper to call a tool with assistant message and validate response.

        Uses LLM.INSTANCES[-1].llm and LLM.INSTANCES[-1].messages directly.

        Args:
            tool_func: The tool call function to invoke
            tool_args: Arguments to pass to tool_func (after msgs)
            expected_ctx_id: Expected context ID to check for (e.g., CTX(0))
            check_items: List of strings to assert are present in response

        Returns:
            The tool call result
        """
        # TODO: simplify
        llm = LLM.INSTANCES[-1].llm
        msgs = LLM.INSTANCES[-1].messages
        msgs.append(llm.msg_assistant(f"{tool_func}"))
        res = tool_func(*tool_args)
        if expected_ctx_id is not None:
            self.assertIn(f"CTX({expected_ctx_id})", res)
        if check_items:
            for item in check_items:
                self.assertIn(item, res)
        return res

    def tool_call_write_with_check(self, path: Path, expected_ctx_id: int, text: str):
        check_items = ["\n>>> === CONTENT START ===", text]
        return self.tool_call_with_check(
            self.tool_call_write,
            tool_args=(path, text),
            expected_ctx_id=expected_ctx_id,
            check_items=check_items,
        )

    def tool_call_read_with_check(
        self, path: Path, expected_ctx_id: int, content_to_check: str = ""
    ):
        check_items = ["\n>>> === CONTENT START ==="]
        if content_to_check:
            check_items.append(content_to_check)
        return self.tool_call_with_check(
            self.tool_call_read,
            tool_args=(path,),
            expected_ctx_id=expected_ctx_id,
            check_items=check_items,
        )

    def append_tool_call_result(self, func: str, msgs: list[dict], result: str | dict):
        llm = LLM.INSTANCES[-1].llm
        msgs = llm.messages()
        if msgs and (tool_calls := msgs[-1].get("tool_calls")):
            id = tool_calls[0]["id"]
        else:
            id = -1

        msgs.append(
            {
                "role": "tool",
                "tool_call_id": id,
                "name": func,
                "content": result,
            }
        )
        return msgs[-1]["content"]

    def epilogue(self):
        """After tool call is finished, llm is always called again"""
        context_handler().prepare_current_llm(LLM.INSTANCES[-1].llm)
        LLM.INSTANCES[-1].messages.append(LLM.INSTANCES[-1].llm.msg_assistant("Done"))

    def init_llm_msgs(self) -> tuple[LLM, list[dict]]:
        dummy_llm = LLM()
        dummy_llm.INSTANCES.append(LlmInstace(dummy_llm, []))
        dummy_llm.add_tool(tool_io.ToolReadFile())
        msgs = dummy_llm.INSTANCES[-1].messages
        return (dummy_llm, msgs)
