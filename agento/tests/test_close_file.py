"""Tests for ToolCloseFile across all context modes."""

import json
import unittest
from agento.context import ContextMode, set_context_mode
from agento.llm import LLM
from agento.tests.test_helper import TestBase
from agento.tests.test_helper import tmpfilename
from agento.tool import io as tool_io


class TestCloseFile(TestBase):
    """Test ToolCloseFile functionality."""

    def setUp(self):
        self.close = tool_io.ToolCloseFile()
        return super().setUp()

    def test_close_file_raw_mode(self):
        """Test close_file in RAW context mode (should be NOP)."""
        set_context_mode(ContextMode.RAW)

        self.init_test_llm()

        # Read the file first
        self.tool_call_read(self.FILE_FOO)

        # Close the file
        res = self.tool_call(
            self.close, files=[self.FILE_FOO.name], reason="done with file"
        )

        # In RAW mode, close_file should return success message
        self.assertIn("OK", res)
        self.assertIn("close_file", res)
        self.assertIn(self.FILE_FOO.name, res)

    def test_close_file_suffix_mode_basic(self):
        """Test close_file in SUFFIX context mode (basic functionality)."""
        set_context_mode(ContextMode.SUFFIX)

        self.init_test_llm()

        # Read the file first
        self.tool_call_read(self.FILE_FOO)

        # Close the file
        res = self.tool_call(
            self.close, files=[self.FILE_FOO.name], reason="editing complete"
        )

        # Verify success
        self.assertIn("OK", res)
        self.assertIn("close_file", res)

    def test_close_file_suffix_mode_prunes_tool_calls(self):
        """Test that close_file prunes old tool calls in SUFFIX mode."""
        set_context_mode(ContextMode.SUFFIX)

        self.init_test_llm()

        # Read the file
        self.tool_call_read(self.FILE_FOO)

        # Edit the file multiple times
        self.tool_call_edit_foo("foo", "foo1")
        self.tool_call_edit_foo("foo1", "foo2")
        self.tool_call_edit_foo("foo2", "foo3")
        self.tool_call_write(self.FILE_FOO, "foo1.1")
        self.tool_call_write(self.FILE_FOO, "foo1.2")

        # Close the file
        res = self.tool_call(self.close, files=[self.FILE_FOO.name], reason="done")

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
        set_context_mode(ContextMode.SUFFIX)

        self.init_test_llm()

        # Read the file
        self.tool_call_read(self.FILE_FOO)

        # Close the file
        res = self.tool_call(
            self.close, files=[self.FILE_FOO.name], reason="finished editing"
        )

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
        set_context_mode(ContextMode.SUFFIX)

        self.init_test_llm()

        # Read the file multiple times
        self.tool_call_read(self.FILE_FOO)
        self.tool_call_read(self.FILE_FOO)

        # Close the file
        res = self.tool_call(
            self.close, files=[self.FILE_FOO.name], reason="done reading"
        )

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
        set_context_mode(ContextMode.SUFFIX)

        self.init_test_llm()

        # Try to close a file that was never read
        res = self.tool_call(self.close, files=["nonexistent.txt"], reason="test")

        # Should still return success (close_file doesn't check if file was opened)
        self.assertIn("OK", res)

    def test_close_file_after_write(self):
        """Test close_file after write_file operation."""
        set_context_mode(ContextMode.SUFFIX)

        self.init_test_llm()

        # Write to a file
        self.tool_call_write(self.FILE_BAR, "written content")

        # Close the file
        res = self.tool_call(
            self.close, files=[self.FILE_BAR.name], reason="writing complete"
        )

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

    def test_close_multiple_files_raw_mode(self):
        """Test closing multiple files at once in RAW context mode."""
        set_context_mode(ContextMode.RAW)

        self.init_test_llm()

        # Read both files first
        self.tool_call_read(self.FILE_FOO)
        self.tool_call_read(self.FILE_BAR)

        # Close both files at once
        res = self.tool_call(
            self.close,
            files=[self.FILE_FOO.name, self.FILE_BAR.name],
            reason="done with both files",
        )

        # In RAW mode, close_file should return success message
        self.assertIn("OK", res)
        self.assertIn("close_file", res)
        # Both files should be mentioned in the response
        self.assertIn(self.FILE_FOO.name, res)
        self.assertIn(self.FILE_BAR.name, res)

    def test_close_multiple_files_suffix_mode(self):
        """Test closing multiple files at once in SUFFIX context mode."""
        set_context_mode(ContextMode.SUFFIX)

        self.init_test_llm()

        # Read both files first
        self.tool_call_read(self.FILE_FOO)
        self.tool_call_read(self.FILE_BAR)

        # Close both files at once
        res = self.tool_call(
            self.close,
            files=[self.FILE_FOO.name, self.FILE_BAR.name],
            reason="editing complete",
        )

        # Verify success
        self.assertIn("OK", res)
        self.assertIn("close_file", res)

        # Verify both files are mentioned
        self.assertIn(self.FILE_FOO.name, res)
        self.assertIn(self.FILE_BAR.name, res)

    def test_close_multiple_files_suffix_mode_prunes_all(self):
        """Test that closing multiple files prunes all tool calls for each file."""
        set_context_mode(ContextMode.SUFFIX)

        self.init_test_llm()

        # Read and edit both files multiple times
        self.tool_call_read(self.FILE_FOO)
        self.tool_call_edit_foo("foo", "foo1")
        self.tool_call_edit_foo("foo1", "foo2")

        self.tool_call_read(self.FILE_BAR)
        self.tool_call_write(self.FILE_BAR, "bar1")
        self.tool_call_write(self.FILE_BAR, "bar2")

        # Close both files at once
        res = self.tool_call(
            self.close,
            files=[self.FILE_FOO.name, self.FILE_BAR.name],
            reason="done with both",
        )

        # Verify success
        self.assertIn("OK", res)

        # Verify tool calls were pruned for both files
        messages = LLM.INSTANCES[-1].messages
        for msg in messages:
            if msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    func_info = tc.get("function", {})
                    func_name = func_info.get("name", "")
                    if func_name in ["edit_file", "read_file", "write_file"]:
                        args = json.loads(func_info.get("arguments", "{}"))
                        # Pruned calls for either file should have cleanup flag
                        if args.get("path") in [self.FILE_FOO.name, self.FILE_BAR.name]:
                            assert (
                                args.get("cleanup") == "the call is removed"
                            ), f"Tool call for {args.get('path')} should be pruned"

    def test_reopen_file_after_close_raw_mode(self):
        """Test that a file can be reopened after being closed in RAW mode."""
        set_context_mode(ContextMode.RAW)

        self.init_test_llm()

        # Read the file
        self.tool_call_read(self.FILE_FOO)

        # Close the file
        self.tool_call(
            self.close, files=[self.FILE_FOO.name], reason="temporarily done"
        )

        # Reopen the file
        self.tool_call_read(self.FILE_FOO)

        # Verify file content is readable again
        messages = LLM.INSTANCES[-1].messages
        found_reopened_content = False
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                if "foo" in content and "text" in content:
                    found_reopened_content = True
                    break

        assert found_reopened_content, "File content should be readable after reopening"

    def test_reopen_file_after_close_suffix_mode(self):
        """Test that a file can be reopened after being closed in SUFFIX mode."""
        set_context_mode(ContextMode.SUFFIX)

        self.init_test_llm()

        # Read the file
        self.tool_call_read(self.FILE_FOO)

        # Close the file
        self.tool_call(
            self.close, files=[self.FILE_FOO.name], reason="temporarily done"
        )

        # Verify "File closed" message exists
        messages = LLM.INSTANCES[-1].messages
        found_closed_message = False
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                if "File closed" in content and "reason: temporarily done" in content:
                    found_closed_message = True
                    break

        assert found_closed_message, "Should find 'File closed' message"

        # Reopen the file
        self.tool_call_read(self.FILE_FOO)

        # Verify file content is readable again
        found_reopened_content = False
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                if "foo" in content and "text" in content:
                    found_reopened_content = True
                    break

        assert found_reopened_content, "File content should be readable after reopening"

    def test_reopen_multiple_files_after_close(self):
        """Test that multiple files can be reopened after being closed together."""
        set_context_mode(ContextMode.SUFFIX)

        self.init_test_llm()

        # Read both files
        self.tool_call_read(self.FILE_FOO)
        self.tool_call_read(self.FILE_BAR)

        # Close both files at once
        self.tool_call(
            self.close,
            files=[self.FILE_FOO.name, self.FILE_BAR.name],
            reason="temporarily done",
        )

        # Reopen both files
        self.tool_call_read(self.FILE_FOO)
        self.tool_call_read(self.FILE_BAR)

        # Verify both files are readable
        messages = LLM.INSTANCES[-1].messages
        found_foo_content = False
        found_bar_content = False
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                if "foo" in content and "text" in content:
                    found_foo_content = True
                if "bar" in content and "value" in content:
                    found_bar_content = True

        assert found_foo_content, "FILE_FOO content should be readable after reopening"
        assert found_bar_content, "FILE_BAR content should be readable after reopening"

    def test_reopen_after_edit_and_close(self):
        """Test reopening a file after editing and closing it."""
        set_context_mode(ContextMode.SUFFIX)

        self.init_test_llm()

        # Read the file
        self.tool_call_read(self.FILE_FOO)

        # Edit the file
        self.tool_call_edit_foo("foo", "edited_foo")

        # Close the file
        self.tool_call(self.close, files=[self.FILE_FOO.name], reason="editing done")

        # Reopen the file
        self.tool_call_read(self.FILE_FOO)

        # Verify file content reflects the edit
        messages = LLM.INSTANCES[-1].messages
        found_edited_content = False
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                if "edited_foo" in content:
                    found_edited_content = True
                    break

        assert (
            found_edited_content
        ), "File should contain edited content after reopening"

    def test_reopen_and_edit_again(self):
        """Test reopening a closed file and making additional edits."""
        set_context_mode(ContextMode.SUFFIX)

        self.init_test_llm()

        # Read the file
        self.tool_call_read(self.FILE_FOO)

        # Edit the file first time
        self.tool_call_edit_foo("foo", "first_edit")

        # Close the file
        self.tool_call(self.close, files=[self.FILE_FOO.name], reason="first edit done")

        # Reopen the file
        self.tool_call_read(self.FILE_FOO)

        # Edit the file again
        self.tool_call_edit_foo("first_edit", "second_edit")

        # Verify file has second edit
        messages = LLM.INSTANCES[-1].messages
        found_second_edit = False
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                if "second_edit" in content:
                    found_second_edit = True
                    break

        assert (
            found_second_edit
        ), "File should contain second edit after reopening and editing"

    def test_multiple_close_reopen_cycles(self):
        """Test multiple close and reopen cycles on the same file."""
        set_context_mode(ContextMode.SUFFIX)

        self.init_test_llm()

        # First cycle: read, close, reopen
        self.tool_call_read(self.FILE_FOO)
        self.tool_call(self.close, files=[self.FILE_FOO.name], reason="cycle 1")
        self.tool_call_read(self.FILE_FOO)

        # Second cycle: close, reopen
        self.tool_call(self.close, files=[self.FILE_FOO.name], reason="cycle 2")
        self.tool_call_read(self.FILE_FOO)

        # Third cycle: close, reopen
        self.tool_call(self.close, files=[self.FILE_FOO.name], reason="cycle 3")
        self.tool_call_read(self.FILE_FOO)

        # Verify file content is still readable after all cycles
        messages = LLM.INSTANCES[-1].messages
        found_content = False
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                if "foo" in content and "text" in content:
                    found_content = True
                    break

        assert (
            found_content
        ), "File should be readable after multiple close/reopen cycles"

    def test_partial_reopen_after_closing_multiple(self):
        """Test reopening only one file after closing multiple files."""
        set_context_mode(ContextMode.SUFFIX)

        self.init_test_llm()

        # Read both files
        self.tool_call_read(self.FILE_FOO)
        self.tool_call_read(self.FILE_BAR)

        # Close both files
        self.tool_call(
            self.close,
            files=[self.FILE_FOO.name, self.FILE_BAR.name],
            reason="done with both",
        )

        # Reopen only FILE_FOO
        self.tool_call_read(self.FILE_FOO)

        # Verify only FILE_FOO is reopened
        messages = LLM.INSTANCES[-1].messages
        found_foo = False
        found_bar = False
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                if "foo" in content and "text" in content:
                    found_foo = True
                # Check for bar content in a new read_file response (not the closed message)
                if (
                    "bar" in content
                    and "value" in content
                    and "File closed" not in content
                ):
                    found_bar = True

        assert found_foo, "FILE_FOO should be reopened"
        assert not found_bar, "FILE_BAR should still be closed"

    def test_reopen_in_different_order_than_close(self):
        """Test reopening files in a different order than they were closed."""
        set_context_mode(ContextMode.SUFFIX)

        self.init_test_llm()

        # Read files in order: FOO, BAR
        self.tool_call_read(self.FILE_FOO)
        self.tool_call_read(self.FILE_BAR)

        # Close both files
        self.tool_call(
            self.close,
            files=[self.FILE_FOO.name, self.FILE_BAR.name],
            reason="closing both",
        )

        # Reopen in reverse order: BAR first, then FOO
        self.tool_call_read(self.FILE_BAR)
        self.tool_call_read(self.FILE_FOO)

        # Verify both are reopened
        messages = LLM.INSTANCES[-1].messages
        found_foo = False
        found_bar = False
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                if "foo" in content and "text" in content:
                    found_foo = True
                if "bar" in content and "value" in content:
                    found_bar = True

        assert found_foo, "FILE_FOO should be reopened"
        assert found_bar, "FILE_BAR should be reopened"

    def test_close_empty_file_list(self):
        """Test close_file with an empty file list."""
        set_context_mode(ContextMode.SUFFIX)

        self.init_test_llm()

        # Try to close with empty list
        res = self.tool_call(self.close, files=[], reason="test")

        # Should return empty string (nothing to close)
        self.assertEqual(res, "")

    def test_close_duplicate_files_in_list(self):
        """Test close_file with duplicate file names in the list."""
        set_context_mode(ContextMode.SUFFIX)

        self.init_test_llm()

        # Read the file
        self.tool_call_read(self.FILE_FOO)

        # Close with duplicate in the list
        res = self.tool_call(
            self.close,
            files=[self.FILE_FOO.name, self.FILE_FOO.name],
            reason="test duplicate",
        )

        # Should still return success
        self.assertIn("OK", res)

    def test_close_three_files_at_once(self):
        """Test closing three files at once."""

        set_context_mode(ContextMode.SUFFIX)

        self.init_test_llm()

        # Create and read a third file
        third_file = tmpfilename("third.txt")
        third_file.write_text("third file content")
        self.tool_call_read(third_file)

        # Read other files
        self.tool_call_read(self.FILE_FOO)
        self.tool_call_read(self.FILE_BAR)

        # Close all three files at once
        res = self.tool_call(
            self.close,
            files=[self.FILE_FOO.name, self.FILE_BAR.name, third_file.name],
            reason="done with all three",
        )

        # Verify success
        self.assertIn("OK", res)

        # Verify all three files are mentioned
        self.assertIn(self.FILE_FOO.name, res)
        self.assertIn(self.FILE_BAR.name, res)
        self.assertIn(third_file.name, res)

        # Clean up
        third_file.unlink()


if __name__ == "__main__":
    unittest.main()
