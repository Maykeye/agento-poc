import unittest
import os
from pathlib import Path
import json

import config
import tool_edit_patch
from tool_editor import ToolEditor
import tool_io
import utilsql
from context import context, context_handler, ContextMode
from llm import LLM, LlmInstace, ToolCall
from typing import Any, Callable, Optional
import shutil

TMP_PREFIX = "/run/user"


def tmpfilename(name: str) -> Path:
    return Path(f"{TMP_PREFIX}/{os.getuid()}/.agento/{name}")


class TestBase(unittest.TestCase):
    FILE_FOO = tmpfilename(".agento.demo.foo")
    FILE_BAR = tmpfilename(".agento.demo.bar")
    ID = 1000

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
        context.set_context_mode(ContextMode.RAW)
        os.chdir(tmpfilename(""))
        config.set_project_directory(tmpfilename(""), silent=True)
        config.set_logging_sqlite_path(":memory:")
        utilsql.reset_all_caches()
        ToolEditor.reset()
        LLM.INSTANCES.clear()
        self.FILE_FOO.write_text("foo\ntext")
        self.FILE_BAR.write_text("bar\nvalue")
        tool_edit_patch.ToolEditDiffPatch.SKIP_SAVING_INVALID_PATCHES = True

    def tearDown(self):
        self.FILE_FOO.unlink(True)
        self.FILE_BAR.unlink(True)

    def tool_call_read(self, path: Path) -> Any:
        context_handler().prepare_current_llm(LLM.INSTANCES[-1].llm)
        self.ID += 1
        msgs = LLM.INSTANCES[-1].messages
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    ToolCall(
                        function="read_file",
                        arguments=json.dumps({"path": path.name}),
                        id=f"id{self.ID}",
                    ).llm_func_call_info()
                ],
            }
        )
        res = tool_io.ToolReadFile()(path.name)
        return self.append_tool_call_result("read_file", msgs, res)

    def tool_call_write(self, path: Path, text: str) -> Any:
        context_handler().prepare_current_llm(LLM.INSTANCES[-1].llm)
        self.ID += 1
        msgs = LLM.INSTANCES[-1].messages
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    ToolCall(
                        function="write_file",
                        arguments=json.dumps({"path": path.name, "text": text}),
                        id=f"id{self.ID}",
                    ).llm_func_call_info()
                ],
            }
        )
        res = tool_io.ToolWriteFile()(path.name, text)
        return self.append_tool_call_result("read_file", msgs, res)

    def tool_call_delete_foo(self) -> Any:
        context_handler().prepare_current_llm(LLM.INSTANCES[-1].llm)
        self.ID += 1
        msgs = LLM.INSTANCES[-1].messages
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    ToolCall(
                        function="delete_file",
                        arguments=json.dumps({"path": self.FILE_FOO.name}),
                        id=f"id{self.ID}",
                    ).llm_func_call_info()
                ],
            }
        )
        res = tool_io.ToolDeleteFile()(self.FILE_FOO.name)
        return self.append_tool_call_result("delete_file", msgs, res)

    def tool_call_edit_foo(self, replace_from: str, replace_with: str) -> Any:
        context_handler().prepare_current_llm(LLM.INSTANCES[-1].llm)
        self.ID += 1
        msgs = LLM.INSTANCES[-1].messages
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    ToolCall(
                        function="edit_file",
                        arguments=json.dumps(
                            {
                                "path": self.FILE_FOO.name,
                                "replace_from": replace_from,
                                "replace_with": replace_with,
                            }
                        ),
                        id=f"id{self.ID}",
                    ).llm_func_call_info()
                ],
            }
        )
        res = tool_io.ToolEditFile()(self.FILE_FOO.name, replace_from, replace_with)
        return self.append_tool_call_result("edit_file", msgs, res)

    def tool_call_add_fold(
        self,
        path: Path,
        fold_from_line_num: int,
        fold_from_line: str,
        fold_to_line_num: int,
        fold_to_line: str,
        name: str,
    ) -> Any:
        """Call the file_add_fold tool using the implementation class (line numbers)."""
        context_handler().prepare_current_llm(LLM.INSTANCES[-1].llm)
        self.ID += 1
        msgs = LLM.INSTANCES[-1].messages
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    ToolCall(
                        function="file_add_fold",
                        arguments=json.dumps(
                            {
                                "path": path.name,
                                "fold_from_line_num": fold_from_line_num,
                                "fold_from_line": fold_from_line,
                                "fold_to_line_num": fold_to_line_num,
                                "fold_to_line": fold_to_line,
                                "name": name,
                            }
                        ),
                        id=f"id{self.ID}",
                    ).llm_func_call_info()
                ],
            }
        )
        res = tool_io.ToolFoldAddImpl()(
            path.name,
            fold_from_line_num,
            fold_from_line,
            fold_to_line_num,
            fold_to_line,
            name,
        )
        return self.append_tool_call_result("file_add_fold", msgs, res)

    def tool_call_add_fold_regex(
        self,
        path: Path,
        start_pattern: str,
        end_pattern: str,
        name: str,
    ) -> Any:
        """Call the file_add_fold tool with regex patterns (new API)."""
        context_handler().prepare_current_llm(LLM.INSTANCES[-1].llm)
        self.ID += 1
        msgs = LLM.INSTANCES[-1].messages
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    ToolCall(
                        function="file_add_fold",
                        arguments=json.dumps(
                            {
                                "path": path.name,
                                "start_pattern": start_pattern,
                                "end_pattern": end_pattern,
                                "name": name,
                            }
                        ),
                        id=f"id{self.ID}",
                    ).llm_func_call_info()
                ],
            }
        )
        res = tool_io.ToolFoldAdd()(
            path.name,
            start_pattern,
            end_pattern,
            name,
        )
        return self.append_tool_call_result("file_add_fold", msgs, res)

    def tool_call_unfold(self, path, name: str):
        """Call the file_unfold tool."""

        context_handler().prepare_current_llm(LLM.INSTANCES[-1].llm)
        self.ID += 1
        msgs = LLM.INSTANCES[-1].messages
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    ToolCall(
                        function="file_unfold",
                        arguments=json.dumps({"path": path.name, "name": name}),
                        id=f"id{self.ID}",
                    ).llm_func_call_info()
                ],
            }
        )
        res = tool_io.ToolUnfold()(path.name, name)
        return self.append_tool_call_result("file_unfold", msgs, res)

    def tool_call_unfold_all(self, path):
        """Call the file_unfold_all tool."""

        context_handler().prepare_current_llm(LLM.INSTANCES[-1].llm)
        self.ID += 1
        msgs = LLM.INSTANCES[-1].messages
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    ToolCall(
                        function="file_unfold_all",
                        arguments=json.dumps({"path": path.name}),
                        id=f"id{self.ID}",
                    ).llm_func_call_info()
                ],
            }
        )
        res = tool_io.ToolUnfoldAll()(path.name)
        return self.append_tool_call_result("file_unfold_all", msgs, res)

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
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": self.ID,
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
