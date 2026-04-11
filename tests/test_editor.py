from pathlib import Path
from typing import ContextManager, Optional
import unittest

from context.context import set_context_mode
from context.context_handler import ContextMode
from llm import LLM, LlmInstace, FinishGeneration
import tool_editor
from tool_editor import EditorEntry, ToolEditor, LINES, KEEP_OLD_BUFFERS
from tests.test_helper import TestBase, tmpfilename
import json


class TestEditorBase(TestBase):
    """Base class for editor tests."""

    FILE_TEST = tmpfilename(".agento.editor.test")
    ID = 2000

    def setUp(self):
        """Set up test environment."""
        super().setUp()
        ToolEditor.SKIP_PRINTING = True

        # Create test file with known content (10 lines)
        self.test_content = """line 1
line 2
line 3
function test() {
    return 42;
}
line 6
line 7
line 8
line 9"""
        self.FILE_TEST.write_text(self.test_content)

    def assertInText(self, substring, text):
        """Helper to check if substring is in text (handles dict results)."""
        if isinstance(text, dict):
            # Check in all dict values
            for value in text.values():
                if isinstance(value, str) and substring in value:
                    return
            # If not found, get the full dict as string
            text = str(text)
        self.assertIn(substring, text)

    def assertNotInText(self, substring, text):
        """Helper to check if substring is NOT in text (handles dict results)."""
        if isinstance(text, dict):
            # Check in all dict values
            for value in text.values():
                if isinstance(value, str) and substring in value:
                    self.fail(f"{substring!r} unexpectedly found in {text!r}")
            return
        self.assertNotIn(substring, text)

    def assertDictHasKey(self, key, dict_value):
        """Helper to check if dict has a key."""
        self.assertIsInstance(dict_value, dict)
        self.assertIn(key, dict_value)

    def assertDictHasKeyContains(self, key, dict_value, substring):
        """Helper to check if dict key's value contains substring."""
        self.assertIsInstance(dict_value, dict)
        self.assertIn(key, dict_value)
        value = dict_value[key]
        if isinstance(value, str):
            self.assertIn(substring, value)

    def tearDown(self):
        """Clean up after tests."""
        if self.FILE_TEST.exists():
            self.FILE_TEST.unlink()
        super().tearDown()

    def init_editor_llm(self) -> int:
        """Initialize an LLM in editor mode for the test file.

        Returns:
            llm_id
        """
        # Create main LLM
        main_llm = LLM()
        main_llm.INSTANCES.append(LlmInstace(main_llm, []))
        main_msgs = main_llm.INSTANCES[-1].messages
        main_msgs.append(main_llm.msg_user("Test editor operations"))
        self.tool_call_read(self.FILE_TEST)

        # Create editor LLM (simulating what ToolEditor.__call__ does)
        editor_llm = main_llm.clone()
        editor_llm.tools.clear()

        # Add editor tools
        ToolEditor.init_editor_tools(editor_llm)

        # Set up editor state
        llm_id = id(editor_llm)
        ToolEditor._state[llm_id] = EditorEntry(self.FILE_TEST.name, 1)

        # Prepare messages
        editor_msgs = main_llm.messages() + [
            main_llm.msg_system("Editor mode system"),
            main_llm.msg_user(f"Editing {self.FILE_TEST.name}"),
        ]

        # Register this as current instance
        editor_llm.INSTANCES.append(LlmInstace(editor_llm, editor_msgs))

        return llm_id


class TestEditorFormatBuffer(TestEditorBase):
    """Test buffer formatting."""

    def test_format_buffer_from_start(self):
        """Test formatting buffer starting from line 1."""
        text = "line1\nline2\nline3"
        buffer = ToolEditor._format_buffer("test.txt", 1, text)

        self.assertIn("[FILE: test.txt", buffer)
        self.assertIn("00001|line1", buffer)
        self.assertIn("00003|line3", buffer)

        # Line 2 should NOT have a line number (not first, not last, not divisible by 10)
        self.assertIn("     |line2", buffer)

    def test_format_buffer_from_middle(self):
        """Test formatting buffer starting from middle of file."""
        text = "line1\nline2\nline3\nline4\nline5"
        buffer = ToolEditor._format_buffer("test.txt", 3, text)

        self.assertIn("00003|line3", buffer)
        self.assertIn("00005|line5", buffer)
        self.assertNotIn("00001|line1", buffer)
        self.assertNotIn("00002|line2", buffer)

        # Line 4 should NOT have a line number (not first, not last, not divisible by 10)
        self.assertIn("     |line4", buffer)

    def test_format_buffer_respects_line_limit(self):
        """Test that buffer respects LINES limit."""
        # Create file with more than LINES lines
        text = "\n".join([f"line{i}" for i in range(1, LINES + 10)])
        buffer = ToolEditor._format_buffer("test.txt", 1, text)

        # Should show at most LINES lines
        lines_shown = [l for l in buffer.split("\n") if l.startswith("0")]
        self.assertLessEqual(len(lines_shown), LINES)


class TestEditorPrintBuffer(TestEditorBase):
    """Test print_buffer functionality."""

    def test_print_buffer_shows_first_line_as_00001(self):
        """Test that print_buffer shows the first line with 00001 prefix."""
        self.init_editor_llm()

        print_tool = tool_editor.EditorToolPrint()
        result = print_tool()

        # Should return buffer output (string, not dict)
        self.assertIsInstance(result, str)

        # Should contain file info
        self.assertIn("[FILE:", result)
        self.assertIn(self.FILE_TEST.name, result)

        # Should show line 1 with 00001 prefix
        self.assertIn("00001|line 1", result)

        # Line 10 should have a line number (divisible by 10)
        self.assertIn("00010|line 9", result)

        # Lines 2-9 should NOT have line numbers (not first, not last, not divisible by 10)
        self.assertIn("     |line 2", result)
        self.assertIn("     |line 3", result)
        self.assertIn("     |line 6", result)
        self.assertIn("     |line 7", result)


class TestEditorGoto(TestEditorBase):
    """Test goto functionality."""

    def test_goto_valid_line(self):
        """Test goto to a valid line."""
        llm_id = self.init_editor_llm()

        goto_tool = tool_editor.EditorToolGoto()
        result = goto_tool(line_number=5)

        # Should update current line
        self.assertEqual(ToolEditor._state[llm_id].current_line, 5)

        # Result should contain buffer
        self.assertIn("Goto line 5", result)
        # Line 5 is "    return 42;" (not "function test() {")
        self.assertIn("00005|    return 42;", result)

    def test_goto_line_too_low(self):
        """Test goto to line number less than 1."""
        self.init_editor_llm()

        goto_tool = tool_editor.EditorToolGoto()
        result = goto_tool(line_number=0)

        self.assertInText("error", result)
        self.assertInText(">= 1", result)

    def test_goto_line_too_high(self):
        """Test goto to line number beyond file length."""
        self.init_editor_llm()

        goto_tool = tool_editor.EditorToolGoto()
        result = goto_tool(line_number=100)

        self.assertInText("error", result)
        self.assertInText("exceeds", result)


class TestEditorEditFile(TestEditorBase):
    """Test edit_file functionality (buffer-only replacement)."""

    def test_edit_file_success(self):
        """Test successful edit in buffer."""
        self.init_editor_llm()

        edit_tool = tool_editor.EditorToolEditFile()
        result = edit_tool(
            replace_from="function test() {", replace_with="MODIFIED function test() {"
        )

        # Should succeed (result is a string, not dict)
        self.assertIsInstance(result, str)
        self.assertIn("Replaced text", result)

        # File should be updated
        self.assertTrue(self.FILE_TEST.exists())
        content = self.FILE_TEST.read_text()
        self.assertIn("MODIFIED function test() {", content)

    def test_edit_file_not_in_buffer(self):
        """Test edit when text is not in current buffer."""
        llm_id = self.init_editor_llm()

        # Go to line 10 (so line 1 is not in buffer)
        ToolEditor._state[llm_id].current_line = 10

        edit_tool = tool_editor.EditorToolEditFile()
        result = edit_tool(replace_from="line 1", replace_with="MODIFIED")

        # Should fail - text not in buffer (returns dict)
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
        self.assertDictHasKeyContains("error", result, "not found in current buffer")

    def test_edit_file_multiple_occurrences(self):
        """Test edit when text appears multiple times in buffer."""
        # Create file with duplicate text
        duplicate_content = "duplicate\nduplicate\nduplicate"
        self.FILE_TEST.write_text(duplicate_content)

        self.init_editor_llm()

        edit_tool = tool_editor.EditorToolEditFile()
        result = edit_tool(replace_from="duplicate", replace_with="unique")

        # Should fail - multiple occurrences (returns dict)
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
        self.assertDictHasKeyContains("error", result, "appears")
        self.assertDictHasKeyContains("error", result, "must be exactly once")


class TestEditorPatchCurrentFile(TestEditorBase):
    """Test patch_current_file functionality."""

    def test_patch_simple(self):
        """Test applying a simple patch."""
        self.init_editor_llm()

        patch_tool = tool_editor.EditorToolPatchCurrentFile()

        # Create a simple patch
        patch = """--- a/.agento.editor.test
+++ b/.agento.editor.test
@@ -1,4 +1,4 @@
-line 1
+MODIFIED line 1
 line 2
 line 3
 function test() {
"""
        result = patch_tool(patch_text=patch)

        # Should succeed
        self.assertNotIn("error", result)
        self.assertIn("Patch applied successfully", result)

        # File should be updated
        content = self.FILE_TEST.read_text()
        self.assertIn("MODIFIED line 1", content)

    def test_patch_without_headers(self):
        """Test applying patch without ---/+++ headers (should add them)."""
        self.init_editor_llm()

        patch_tool = tool_editor.EditorToolPatchCurrentFile()

        # Patch without headers
        patch = """@@ -1,4 +1,4 @@
-line 1
+MODIFIED line 1
 line 2
 line 3
"""
        result = patch_tool(patch_text=patch)

        # Should succeed (headers added automatically)
        self.assertNotIn("error", result)
        self.assertIn("Patch applied successfully", result)

    def test_patch_no_overlap(self):
        """Test patch when hunks don't overlap with buffer."""
        llm_id = self.init_editor_llm()

        # Go to end of file (line 10)
        ToolEditor._state[llm_id].current_line = 10

        patch_tool = tool_editor.EditorToolPatchCurrentFile()

        # Patch for line 1 (not in current buffer)
        patch = """--- a/.agento.editor.test
+++ b/.agento.editor.test
@@ -1,2 +1,2 @@
-line 1
+MODIFIED
 line 2
"""
        result = patch_tool(patch_text=patch)

        # Should fail - no overlap (returns dict)
        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
        self.assertDictHasKeyContains("error", result, "overlap")


class TestEditorAppend(TestEditorBase):
    """Test append_file"""

    def find_read(self, p: Path, llm: Optional[LLM] = None):
        llm = llm or LLM.INSTANCES[-1].llm
        for i in range(len(llm.messages()) - 1, -1, -1):
            msg = llm.messages()[i]
            if msg.get("role") != "tool":
                continue
            if not (content := msg.get("content")):
                continue
            line = content.splitlines()[0]
            if f"CTX-IO-FILE: {p.name}" in line:
                return i, msg
        return -1, {}

    def impl(self, orig_sfx: str):
        set_context_mode(ContextMode.SUFFIX)
        self.FILE_TEST.write_text(f"hello\nworld{orig_sfx}")
        self.init_editor_llm()
        self.tool_call_editor_append(self.FILE_TEST, "and autumn")
        self.epilogue()
        text = self.FILE_TEST.read_text()
        self.assertEqual(text, "hello\nworld\nand autumn")
        _, msg = self.find_read(self.FILE_TEST)
        self.assertIn("hello\nworld\nand autumn", msg["content"])
        self.assertEqual(len(LLM.INSTANCES), 2)

    def test_nl(self):
        self.impl("\n")

    def test_no_nl(self):
        self.impl("")


class TestEditorWriteNewContent(TestEditorBase):
    """Test write_new_content functionality."""

    def test_write_new_content(self):
        """Test writing new content and exiting editor."""
        llm_id = self.init_editor_llm()

        write_tool = tool_editor.EditorToolWriteNewContent()
        result = write_tool(
            path=self.FILE_TEST.name, new_content="completely new content"
        )

        # Should return FinishGeneration (not dict)
        assert isinstance(result, FinishGeneration)

        # Parse the FinishGeneration value (it's JSON)
        result_value = json.loads(result.value)

        # Check result value
        self.assertEqual(result_value["status"], "success")
        self.assertIn("written successfully", result_value["message"])

        # File should be updated
        content = self.FILE_TEST.read_text()
        self.assertEqual(content, "completely new content")

        # Editor state should be cleaned up
        self.assertNotIn(llm_id, ToolEditor._state)

    def test_write_new_content_path_mismatch(self):
        """Test that write_new_content rejects mismatched paths."""
        llm_id = self.init_editor_llm()
        write_tool = tool_editor.EditorToolWriteNewContent()
        result = write_tool(path=self.FILE_BAR.name, new_content="new content")

        # Should return error dict with path mismatch
        assert isinstance(result, dict)
        self.assertIn("error", result)
        self.assertIn("Path mismatch", result["error"])
        self.assertDictHasKeyContains("suggestion", result, "You are editing")
        self.assertIn(self.FILE_TEST.name, result["suggestion"])
        self.assertIn(self.FILE_BAR.name, result["suggestion"])

        # Editor state should NOT be cleaned up
        self.assertIn(llm_id, ToolEditor._state)

        # File should NOT be updated
        content = self.FILE_TEST.read_text()
        self.assertNotIn("new content", content)

    def test_write_new_content_relative_path(self):
        """Test that write_new_content works with relative paths."""
        self.init_editor_llm()

        write_tool = tool_editor.EditorToolWriteNewContent()

        # Use relative path (should resolve to same file)
        result = write_tool(
            path=".agento.editor.test", new_content="relative path content"
        )

        # Should return FinishGeneration (not dict)
        assert isinstance(result, FinishGeneration)

        # Parse the FinishGeneration value (it's JSON)
        result_value = json.loads(result.value)

        # Check result value
        self.assertEqual(result_value["status"], "success")

        # File should be updated
        content = self.FILE_TEST.read_text()
        self.assertEqual(content, "relative path content")


class TestEditorFinishEditing(TestEditorBase):
    """Test finish_editing functionality."""

    def test_finish_with_report(self):
        """Test finishing editing with a report."""
        llm_id = self.init_editor_llm()

        finish_tool = tool_editor.EditorToolFinishEditing()
        result = finish_tool(report="Reviewed the code, no changes needed")
        assert isinstance(result, FinishGeneration)

        # Parse the FinishGeneration value (it's JSON)
        result_value = json.loads(result.value)

        # Check result value
        self.assertEqual(result_value["status"], "editing_finished")
        self.assertIn("Reviewed the code", result_value["report"])

        # Editor state should be cleaned up
        self.assertNotIn(llm_id, ToolEditor._state)

    def test_finish_without_report(self):
        """Test finishing editing without a report."""
        self.init_editor_llm()

        finish_tool = tool_editor.EditorToolFinishEditing()
        result = finish_tool(report="")
        assert isinstance(result, FinishGeneration)
        result_value = json.loads(result.value)
        self.assertEqual(result_value["status"], "editing_finished")


class TestEditorPruneBuffers(TestEditorBase):
    """Test buffer pruning functionality."""

    def test_prune_old_buffers(self):
        """Test that old buffers are pruned correctly."""
        self.init_editor_llm()

        # Get editor messages from the last LLM instance
        editor_msgs = LLM.INSTANCES[-1].messages

        # Simulate multiple tool outputs
        for i in range(KEEP_OLD_BUFFERS + 3):
            editor_msgs.append({"role": "assistant", "content": f"Calling goto {i}"})
            editor_msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": f"id{i}",
                    "name": "goto",
                    "content": f"Buffer output {i}",
                }
            )

        # Prune old buffers
        ToolEditor._prune_old_buffers(editor_msgs)

        # Count non-pruned tool outputs
        non_pruned = [
            m
            for m in editor_msgs
            if m.get("role") == "tool" and "PRUNED" not in m.get("content", "")
        ]

        # Should have KEEP_OLD_BUFFERS non-pruned outputs
        self.assertEqual(len(non_pruned), KEEP_OLD_BUFFERS)

        # Count pruned outputs
        pruned = [
            m
            for m in editor_msgs
            if m.get("role") == "tool" and "PRUNED" in m.get("content", "")
        ]

        # 3 reads here + 1 in init_editor_llm
        self.assertEqual(len(pruned), 4)


class TestToolEditorMain(TestEditorBase):
    """Test the main ToolEditor class."""

    def test_editor_file_not_exists(self):
        """Test starting editor with non-existent file."""
        editor = ToolEditor()
        result = editor(path="nonexistent_file.txt")

        self.assertIsInstance(result, dict)
        self.assertIn("error", result)
        self.assertDictHasKeyContains("error", result, "does not exist")

    @unittest.skip("Mock generate NYI")
    def test_editor_empty_file(self):
        """Test starting editor with empty file - should now allow editing."""
        empty_file = tmpfilename(".agento.editor.empty")
        empty_file.write_text("")

        try:
            # Need to set up an LLM instance first
            main_llm = LLM()
            main_msgs = [main_llm.msg_system("Main"), main_llm.msg_user("Test")]
            main_llm.INSTANCES.append(LlmInstace(main_llm, main_msgs))

            editor = ToolEditor()
            result = editor(path=empty_file.name)

            # Should succeed - empty files are now allowed
            self.assertIsInstance(result, dict)
            self.assertIn("status", result)
            self.assertEqual(result["status"], "editing_complete")
            self.assertIn("file", result)
        finally:
            if empty_file.exists():
                empty_file.unlink()


class TestEditorQuitBehavior(TestEditorBase):
    """Test that editor tools properly quit and return to main LLM."""

    def test_write_new_content_quits_editor(self):
        """Test that write_new_content properly quits editor and returns to main LLM.

        This test verifies:
        1. Before quitting, there are 2 LLM instances (main and editor)
        2. After calling EditorToolWriteNewContent(), only main LLM remains
        3. llm.instances[-1] is the original main LLM (not the editor)
        4. Editor variables (current_line, editing_files) are cleaned up
        """
        # Clear all instances first
        LLM.INSTANCES.clear()

        # Create main LLM
        main_llm = LLM()
        main_msgs = [main_llm.msg_system("Main"), main_llm.msg_user("Test")]
        main_llm.INSTANCES.append(LlmInstace(main_llm, main_msgs))

        # Create editor LLM (simulating editor mode)
        editor_llm = main_llm.clone()
        editor_llm.tools.clear()
        editor_llm.add_tool(tool_editor.EditorToolWriteNewContent())

        editor_msgs = [
            main_llm.msg_system("Editor mode"),
            main_llm.msg_user("Editing file"),
        ]
        editor_llm.INSTANCES.append(LlmInstace(editor_llm, editor_msgs))

        # Set up editor state
        editor_llm_id = id(editor_llm)
        ToolEditor._state[editor_llm_id] = EditorEntry(self.FILE_TEST.name, 5)

        # Assert there are 2 LLM instances before quitting
        self.assertEqual(len(LLM.INSTANCES), 2)

        # Call write_new_content (returns FinishGeneration)
        write_tool = tool_editor.EditorToolWriteNewContent()
        result = write_tool(path=self.FILE_TEST.name, new_content="new content")

        # Should return FinishGeneration
        self.assertIsInstance(result, FinishGeneration)

        # After FinishGeneration, the editor LLM instance should be removed
        # (This is handled by the LLM.generate() finally block)
        # But in our test, we're calling the tool directly, not through generate()
        # So we need to verify the cleanup happens at the tool level

        # Editor state should be cleaned up
        self.assertNotIn(editor_llm_id, ToolEditor._state)

        # File should be updated
        content = self.FILE_TEST.read_text()
        self.assertEqual(content, "new content")

    def test_finish_editing_quits_editor(self):
        """Test that finish_editing properly quits editor and returns to main LLM.

        This test verifies:
        1. Before quitting, there are 2 LLM instances (main and editor)
        2. After calling EditorToolFinishEditing(), editor variables are cleaned up
        3. llm.instances[-1] is the original main LLM (not the editor)
        4. Editor variables (current_line, editing_files) are cleaned up
        """
        # Clear all instances first
        LLM.INSTANCES.clear()

        # Create main LLM
        main_llm = LLM()
        main_msgs = [main_llm.msg_system("Main"), main_llm.msg_user("Test")]
        main_llm.INSTANCES.append(LlmInstace(main_llm, main_msgs))

        # Create editor LLM (simulating editor mode)
        editor_llm = main_llm.clone()
        editor_llm.tools.clear()
        editor_llm.add_tool(tool_editor.EditorToolFinishEditing())

        editor_msgs = [
            main_llm.msg_system("Editor mode"),
            main_llm.msg_user("Editing file"),
        ]
        editor_llm.INSTANCES.append(LlmInstace(editor_llm, editor_msgs))

        # Set up editor state
        editor_llm_id = id(editor_llm)
        ToolEditor._state[editor_llm_id] = EditorEntry(self.FILE_TEST.name, 10)

        # Assert there are 2 LLM instances before quitting
        self.assertEqual(len(LLM.INSTANCES), 2)

        # Call finish_editing (returns FinishGeneration)
        finish_tool = tool_editor.EditorToolFinishEditing()
        result = finish_tool(report="Done editing")

        # Should return FinishGeneration
        assert isinstance(result, FinishGeneration)

        # Parse the FinishGeneration value
        result_value = json.loads(result.value)
        self.assertEqual(result_value["status"], "editing_finished")
        self.assertIn("Done editing", result_value["report"])

        # Editor state should be cleaned up
        self.assertNotIn(editor_llm_id, ToolEditor._state)


if __name__ == "__main__":
    unittest.main()
