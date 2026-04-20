import json
import unittest

import tool_editor
from tool_editor import EditorEntry, ToolEditor
from context import context_handler
from llm import LLM, LlmInstace, ToolCall

from tests.test_helper import TestBase


class TestEditorSwitch(TestBase):
    """Test switching files in editor mode."""

    def init_test_llm(self):
        """Initialize LLM for the test."""
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Test editor switch"))
        return dummy_llm, msgs

    def test_switch_file_basic(self):
        """Test basic file switching functionality."""
        llm, msgs = self.init_test_llm()
        switch_tool = tool_editor.EditorToolSwitchFile()

        # Enter editor mode with FILE_FOO
        context_handler().prepare_current_llm(llm)
        llm.append_tool_call(switch_tool.name, path=self.FILE_FOO.name)

        # Simulate being in editor mode - set up the state manually
        editor_llm = llm.clone()
        editor_llm.tools.clear()
        ToolEditor.init_editor_tools(editor_llm)

        llm_id = id(editor_llm)
        ToolEditor._state[llm_id] = EditorEntry(self.FILE_FOO.name, 1)

        # Add editor mode message
        msgs.append(
            {
                "role": "user",
                "content": f"[SYSTEM OVERRIDE: EDITOR MODE ACTIVATED]\n\nEditing: {self.FILE_FOO.name}",
            }
        )

        # Prepare the message for the switch call
        llm.append_tool_call(switch_tool.name, path=self.FILE_FOO.name)

        # Make the editor_llm the current instance
        LLM.INSTANCES.append(LlmInstace(editor_llm, msgs))

        result = switch_tool(self.FILE_BAR.name)

        # Check that we successfully switched
        self.assertIn("Switching from", result)
        self.assertIn(self.FILE_FOO.name, result)
        self.assertIn(self.FILE_BAR.name, result)

        # Verify the editor state was updated
        self.assertEqual(ToolEditor._state[llm_id].path, self.FILE_BAR.name)
        self.assertEqual(ToolEditor._state[llm_id].current_line, 1)

    def test_switch_file_nonexistent(self):
        """Test switching to a non-existent file fails gracefully."""
        llm, msgs = self.init_test_llm()

        # Set up editor mode state
        editor_llm = llm.clone()
        editor_llm.tools.clear()
        ToolEditor.init_editor_tools(editor_llm)

        llm_id = id(editor_llm)
        ToolEditor._state[llm_id] = EditorEntry(self.FILE_FOO.name, 1)

        LLM.INSTANCES.append(LlmInstace(editor_llm, msgs))

        # Try to switch to non-existent file
        switch_tool = tool_editor.EditorToolSwitchFile()
        result = switch_tool("nonexistent.txt")

        # Should get an error
        assert isinstance(result, dict)
        self.assertIn("error", result)
        self.assertIn("does not exist", result["error"])

    def test_switch_file_empty(self):
        """Test switching to an empty file - should now succeed."""
        llm, msgs = self.init_test_llm()
        self.FILE_BAR.write_text("")

        # Set up editor mode state
        editor_llm = llm.clone()
        editor_llm.tools.clear()
        ToolEditor.init_editor_tools(editor_llm)

        llm_id = id(editor_llm)
        ToolEditor._state[llm_id] = EditorEntry(self.FILE_FOO.name, 1)

        LLM.INSTANCES.append(LlmInstace(editor_llm, msgs))

        # Switch to empty file
        switch_tool = tool_editor.EditorToolSwitchFile()
        result = switch_tool(self.FILE_BAR.name)

        # Should succeed - empty files are now allowed
        assert isinstance(result, str), result
        self.assertIn("Switching from", result)
        self.assertIn(self.FILE_FOO.name, result)
        self.assertIn(self.FILE_BAR.name, result)

        # Verify the editor state was updated
        self.assertEqual(ToolEditor._state[llm_id].path, self.FILE_BAR.name)
        self.assertEqual(ToolEditor._state[llm_id].current_line, 1)

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
        ToolEditor.init_editor_tools(editor_llm)

        llm_id = id(editor_llm)
        ToolEditor._state[llm_id] = tool_editor.EditorEntry(self.FILE_FOO.name, 1)

        # Add many buffer messages to simulate old buffers
        for i in range(10):
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": f"print#{i}",
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
