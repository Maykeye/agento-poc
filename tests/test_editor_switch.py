import unittest
import json
import os
from pathlib import Path
import shutil

import config
import tool_editor
import utilsql
from context import context, context_handler, ContextMode
from llm import LLM, LlmInstace, ToolCall

TMP_PREFIX = "/run/user"


def tmpfilename(name: str) -> Path:
    return Path(f"{TMP_PREFIX}/{os.getuid()}/.agento/{name}")


class TestEditorSwitch(unittest.TestCase):
    """Test switching files in editor mode."""

    FILE_FOO = tmpfilename(".test.editor.switch.foo")
    FILE_BAR = tmpfilename(".test.editor.switch.bar")
    FILE_BAZ = tmpfilename(".test.editor.switch.baz")
    ID = 1000

    def setUp(self):
        """Set up test fixtures."""
        if Path(tmpfilename("")).exists():
            assert Path(tmpfilename("")).is_dir()
            shutil.rmtree(tmpfilename(""))
        Path(tmpfilename("")).mkdir(parents=True, exist_ok=True)

        context.set_context_mode(ContextMode.RAW)
        os.chdir(tmpfilename(""))
        config.set_project_directory(tmpfilename(""), silent=True)
        config.set_logging_sqlite_path(":memory:")
        utilsql.reset_all_caches()
        LLM.INSTANCES.clear()

        # Create test files
        self.FILE_FOO.write_text("foo\nline2\nline3\n")
        self.FILE_BAR.write_text("bar\nline2\nline3\n")
        self.FILE_BAZ.write_text("baz\ncontent\nhere\n")

    def tearDown(self):
        """Clean up test files."""
        for f in [self.FILE_FOO, self.FILE_BAR, self.FILE_BAZ]:
            if f.exists():
                f.unlink(True)

    def init_test_llm(self):
        """Initialize LLM for the test."""
        dummy_llm = LLM()
        dummy_llm.INSTANCES.append(LlmInstace(dummy_llm, []))
        dummy_llm.add_tool(tool_editor.ToolEditor())
        msgs = dummy_llm.INSTANCES[-1].messages
        msgs.append(dummy_llm.msg_user("Test editor switch"))
        return dummy_llm, msgs

    def test_switch_file_basic(self):
        """Test basic file switching functionality."""
        llm, msgs = self.init_test_llm()

        # Enter editor mode with FILE_FOO
        context_handler().prepare_current_llm(llm)
        self.ID += 1
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    ToolCall(
                        function="edit_file",
                        arguments=json.dumps({"path": self.FILE_FOO.name}),
                        id=f"id{self.ID}",
                    ).llm_func_call_info()
                ],
            }
        )

        # Simulate being in editor mode - set up the state manually
        editor_llm = llm.clone()
        editor_llm.tools.clear()
        tool_editor.ToolEditor.init_editor_tools(editor_llm)

        llm_id = id(editor_llm)
        tool_editor.ToolEditor._current_lines[llm_id] = 1
        tool_editor.ToolEditor._editing_files[llm_id] = self.FILE_FOO.name

        # Add editor mode message
        msgs.append(
            {
                "role": "user",
                "content": f"[SYSTEM OVERRIDE: EDITOR MODE ACTIVATED]\n\nEditing: {self.FILE_FOO.name}",
            }
        )

        # Now simulate switching to FILE_BAR
        switch_tool = tool_editor.EditorToolSwitchFile()

        # Prepare the message for the switch call
        self.ID += 1
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    ToolCall(
                        function="edit_file_to_switch",
                        arguments=json.dumps({"path": self.FILE_BAR.name}),
                        id=f"id{self.ID}",
                    ).llm_func_call_info()
                ],
            }
        )

        # Make the editor_llm the current instance
        LLM.INSTANCES.append(LlmInstace(editor_llm, msgs))

        result = switch_tool(self.FILE_BAR.name)

        # Check that we successfully switched
        self.assertIn("Switching from", result)
        self.assertIn(self.FILE_FOO.name, result)
        self.assertIn(self.FILE_BAR.name, result)

        # Verify the editor state was updated
        self.assertEqual(
            tool_editor.ToolEditor._editing_files[llm_id], self.FILE_BAR.name
        )
        self.assertEqual(tool_editor.ToolEditor._current_lines[llm_id], 1)

        # Clean up
        del tool_editor.ToolEditor._current_lines[llm_id]
        del tool_editor.ToolEditor._editing_files[llm_id]

    def test_switch_file_nonexistent(self):
        """Test switching to a non-existent file fails gracefully."""
        llm, msgs = self.init_test_llm()

        # Set up editor mode state
        editor_llm = llm.clone()
        editor_llm.tools.clear()
        tool_editor.ToolEditor.init_editor_tools(editor_llm)

        llm_id = id(editor_llm)
        tool_editor.ToolEditor._current_lines[llm_id] = 1
        tool_editor.ToolEditor._editing_files[llm_id] = self.FILE_FOO.name

        LLM.INSTANCES.append(LlmInstace(editor_llm, msgs))

        # Try to switch to non-existent file
        switch_tool = tool_editor.EditorToolSwitchFile()
        result = switch_tool("nonexistent.txt")

        # Should get an error
        assert isinstance(result, dict)
        self.assertIn("error", result)
        self.assertIn("does not exist", result["error"])

    def test_switch_file_empty(self):
        """Test switching to an empty file fails gracefully."""
        llm, msgs = self.init_test_llm()
        self.FILE_BAR.write_text("")

        # Set up editor mode state
        editor_llm = llm.clone()
        editor_llm.tools.clear()
        tool_editor.ToolEditor.init_editor_tools(editor_llm)

        llm_id = id(editor_llm)
        tool_editor.ToolEditor._current_lines[llm_id] = 1
        tool_editor.ToolEditor._editing_files[llm_id] = self.FILE_FOO.name

        LLM.INSTANCES.append(LlmInstace(editor_llm, msgs))

        # Try to switch to empty file
        switch_tool = tool_editor.EditorToolSwitchFile()
        result = switch_tool(self.FILE_BAR.name)

        # Should get an error
        assert isinstance(result, dict), result
        self.assertIn("error", result)
        self.assertIn("empty", result["error"])

    def test_switch_file_not_in_editor_mode(self):
        """Test switching file when not in editor mode fails."""
        llm, msgs = self.init_test_llm()

        # Don't set up editor mode state
        LLM.INSTANCES.append(LlmInstace(llm, msgs))

        switch_tool = tool_editor.EditorToolSwitchFile()
        result = switch_tool(self.FILE_BAR.name)

        # Should get an error
        assert isinstance(result, dict)
        self.assertIn("error", result)
        self.assertIn("Not in editor mode", result["error"])

    def test_switch_file_prunes_old_buffers(self):
        """Test that switching files prunes old buffers when limit exceeded."""
        llm, msgs = self.init_test_llm()

        # Set up editor mode state
        editor_llm = llm.clone()
        editor_llm.tools.clear()
        tool_editor.ToolEditor.init_editor_tools(editor_llm)

        llm_id = id(editor_llm)
        tool_editor.ToolEditor._current_lines[llm_id] = 1
        tool_editor.ToolEditor._editing_files[llm_id] = self.FILE_FOO.name

        # Add many buffer messages to simulate old buffers
        for i in range(10):
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": self.ID + i,
                    "name": "print_buffer",
                    "content": f"Buffer output {i}",
                }
            )

        LLM.INSTANCES.append(LlmInstace(editor_llm, msgs))

        # Switch to another file
        switch_tool = tool_editor.EditorToolSwitchFile()
        switch_tool(self.FILE_BAR.name)

        # Check that old buffers were pruned
        # KEEP_OLD_BUFFERS is 5, so we should have pruned some
        pruned_count = sum(
            1
            for msg in msgs
            if msg.get("role") == "tool"
            and msg.get("name") == "print_buffer"
            and "PRUNED" in msg.get("content", "")
        )

        self.assertGreater(pruned_count, 0, "Old buffers should have been pruned")


if __name__ == "__main__":
    unittest.main()
