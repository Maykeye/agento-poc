"""Tests for ToolCloseFile across all context modes."""

import unittest
import json

from tests.test_helper import TestBase
from context import context, ContextMode
from llm import LLM, ToolCall
import tool_io
from context.prefix import CONTEXTS


class TestCloseFile(TestBase):
    """Test ToolCloseFile functionality."""

    def test_close_file_raw_mode(self):
        """Test close_file in RAW context mode (should be NOP)."""
        context.set_context_mode(ContextMode.RAW)

        _, msgs = self.init_test_llm()

        # Read the file first
        self.tool_call_read(self.FILE_FOO)

        # Close the file
        self.ID += 1
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    ToolCall(
                        function="close_file",
                        arguments=json.dumps(
                            {"file": self.FILE_FOO.name, "reason": "done with file"}
                        ),
                        id=f"id{self.ID}",
                    ).llm_func_call_info()
                ],
            }
        )

        res = tool_io.ToolCloseFile()(self.FILE_FOO.name, "done with file")

        # In RAW mode, close_file should return success message
        self.assertIn("OK", res)
        self.assertIn("close_file", res)
        self.assertIn(self.FILE_FOO.name, res)

    def test_close_file_prefix_mode(self):
        """Test close_file in PREFIX context mode (removes from CONTEXTS)."""
        context.set_context_mode(ContextMode.PREFIX)

        _, msgs = self.init_test_llm()

        # Read the file first
        self.tool_call_read(self.FILE_FOO)

        # Verify file is in CONTEXTS

        assert self.FILE_FOO.name in CONTEXTS, "File should be in CONTEXTS after read"

        # Close the file
        self.ID += 1
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    ToolCall(
                        function="close_file",
                        arguments=json.dumps(
                            {"file": self.FILE_FOO.name, "reason": "no longer needed"}
                        ),
                        id=f"id{self.ID}",
                    ).llm_func_call_info()
                ],
            }
        )

        res = tool_io.ToolCloseFile()(self.FILE_FOO.name, "no longer needed")

        # Verify success
        self.assertIn("OK", res)

        # Verify file was removed from CONTEXTS
        assert (
            self.FILE_FOO.name not in CONTEXTS
        ), "File should be removed from CONTEXTS after close"

    def test_close_file_suffix_mode_basic(self):
        """Test close_file in SUFFIX context mode (basic functionality)."""
        context.set_context_mode(ContextMode.SUFFIX)

        _, msgs = self.init_test_llm()

        # Read the file first
        self.tool_call_read(self.FILE_FOO)

        # Close the file
        self.ID += 1
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    ToolCall(
                        function="close_file",
                        arguments=json.dumps(
                            {"file": self.FILE_FOO.name, "reason": "editing complete"}
                        ),
                        id=f"id{self.ID}",
                    ).llm_func_call_info()
                ],
            }
        )

        res = tool_io.ToolCloseFile()(self.FILE_FOO.name, "editing complete")

        # Verify success
        self.assertIn("OK", res)
        self.assertIn("close_file", res)

    def test_close_file_suffix_mode_prunes_tool_calls(self):
        """Test that close_file prunes old tool calls in SUFFIX mode."""
        context.set_context_mode(ContextMode.SUFFIX)

        _, msgs = self.init_test_llm()

        # Read the file
        self.tool_call_read(self.FILE_FOO)

        # Edit the file multiple times
        self.tool_call_edit_foo("foo", "foo1")
        self.tool_call_edit_foo("foo1", "foo2")
        self.tool_call_edit_foo("foo2", "foo3")
        self.tool_call_write(self.FILE_FOO, "foo1.1")
        self.tool_call_write(self.FILE_FOO, "foo1.2")

        # Close the file
        self.ID += 1
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    ToolCall(
                        function="close_file",
                        arguments=json.dumps(
                            {"file": self.FILE_FOO.name, "reason": "done"}
                        ),
                        id=f"id{self.ID}",
                    ).llm_func_call_info()
                ],
            }
        )

        res = tool_io.ToolCloseFile()(self.FILE_FOO.name, "done")

        # Verify success
        self.assertIn("OK", res)

        # Verify tool calls were pruned (check messages)
        # All edit_file and read_file calls should have been pruned
        messages = LLM.INSTANCES[-1].messages
        for msg in messages:
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    func_info = tc.get("function", {})
                    func_name = func_info.get("name", "")
                    if func_name in ["edit_file", "read_file"]:
                        args = json.loads(func_info.get("arguments", "{}"))
                        # Pruned calls should have cleanup flag
                        if args.get("path") == self.FILE_FOO.name:
                            # Verify it was pruned
                            assert args.get("cleanup") == "the call is removed"

    def test_close_file_suffix_mode_replaces_content(self):
        """Test that close_file replaces content blocks with closed message."""
        context.set_context_mode(ContextMode.SUFFIX)

        _, msgs = self.init_test_llm()

        # Read the file
        self.tool_call_read(self.FILE_FOO)

        # Close the file
        self.ID += 1
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    ToolCall(
                        function="close_file",
                        arguments=json.dumps(
                            {"file": self.FILE_FOO.name, "reason": "finished editing"}
                        ),
                        id=f"id{self.ID}",
                    ).llm_func_call_info()
                ],
            }
        )

        res = tool_io.ToolCloseFile()(self.FILE_FOO.name, "finished editing")

        # Verify success
        self.assertIn("OK", res)

        # Verify content blocks were replaced with closed message
        messages = LLM.INSTANCES[-1].messages
        found_closed_message = False
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                if "File closed" in content and "reason: finished editing" in content:
                    found_closed_message = True
                    break

        assert found_closed_message, "Should find 'File closed' message in content"

    def test_close_file_suffix_mode_with_read_file_prune(self):
        """Test that close_file prunes read_file calls in SUFFIX mode."""
        context.set_context_mode(ContextMode.SUFFIX)

        _, msgs = self.init_test_llm()

        # Read the file multiple times
        self.tool_call_read(self.FILE_FOO)
        self.tool_call_read(self.FILE_FOO)

        # Close the file
        self.ID += 1
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    ToolCall(
                        function="close_file",
                        arguments=json.dumps(
                            {"file": self.FILE_FOO.name, "reason": "done reading"}
                        ),
                        id=f"id{self.ID}",
                    ).llm_func_call_info()
                ],
            }
        )

        res = tool_io.ToolCloseFile()(self.FILE_FOO.name, "done reading")

        # Verify success
        self.assertIn("OK", res)

        # Verify read_file calls were pruned
        messages = LLM.INSTANCES[-1].messages
        for msg in messages:
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    func_info = tc.get("function", {})
                    func_name = func_info.get("name", "")
                    if func_name == "read_file":
                        args = json.loads(func_info.get("arguments", "{}"))
                        # Pruned read_file calls should have cleanup flag
                        if args.get("path") == self.FILE_FOO.name:
                            assert args.get("cleanup") == "the call is removed"

    def test_close_file_nonexistent_file(self):
        """Test close_file on a file that was never opened."""
        context.set_context_mode(ContextMode.SUFFIX)

        _, msgs = self.init_test_llm()

        # Try to close a file that was never read
        self.ID += 1
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    ToolCall(
                        function="close_file",
                        arguments=json.dumps(
                            {"file": "nonexistent.txt", "reason": "test"}
                        ),
                        id=f"id{self.ID}",
                    ).llm_func_call_info()
                ],
            }
        )

        res = tool_io.ToolCloseFile()("nonexistent.txt", "test")

        # Should still return success (close_file doesn't check if file was opened)
        self.assertIn("OK", res)

    def test_close_file_after_write(self):
        """Test close_file after write_file operation."""
        context.set_context_mode(ContextMode.SUFFIX)

        _, msgs = self.init_test_llm()

        # Write to a file
        self.tool_call_write(self.FILE_BAR, "written content")

        # Close the file
        self.ID += 1
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    ToolCall(
                        function="close_file",
                        arguments=json.dumps(
                            {"file": self.FILE_BAR.name, "reason": "writing complete"}
                        ),
                        id=f"id{self.ID}",
                    ).llm_func_call_info()
                ],
            }
        )

        res = tool_io.ToolCloseFile()(self.FILE_BAR.name, "writing complete")

        # Verify success
        self.assertIn("OK", res)

        # Verify tool calls were pruned
        messages = LLM.INSTANCES[-1].messages
        for msg in messages:
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    func_info = tc.get("function", {})
                    func_name = func_info.get("name", "")
                    if func_name == "write_file":
                        args = json.loads(func_info.get("arguments", "{}"))
                        if args.get("path") == self.FILE_BAR.name:
                            assert args.get("cleanup") == "the call is removed"


if __name__ == "__main__":
    unittest.main()
