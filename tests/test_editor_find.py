import unittest
import tool_editor
from tool_editor import ToolEditor
from tests.test_editor import TestEditorBase


class TestEditorFindNext(TestEditorBase):
    """Test find_next functionality."""

    def test_find_next_found(self):
        """Test finding next occurrence of pattern."""
        llm_id = self.init_editor_llm()

        # Start at line 1, find "line" which appears multiple times
        find_tool = tool_editor.EditorToolFindNext()
        result = find_tool(pattern=r"l.ne 9")

        # Should find the pattern
        self.assertNotIn("error", result)
        self.assertIn("Pattern:", result)

        self.assertEqual(ToolEditor._state[llm_id].current_line, 5)

    def test_find_next_not_found(self):
        """Test finding pattern that doesn't exist."""
        self.init_editor_llm()

        find_tool = tool_editor.EditorToolFindNext()
        result = find_tool(pattern=r"nonexistent_pattern_xyz")

        self.assertDictHasKey("status", result)
        assert isinstance(result, dict)
        self.assertEqual(result["status"], "not_found")
        self.assertDictHasKeyContains("message", result, "not found")

    def test_find_next_invalid_regex(self):
        """Test find with invalid regex pattern."""
        self.init_editor_llm()

        find_tool = tool_editor.EditorToolFindNext()
        result = find_tool(pattern=r"[invalid(regex")

        self.assertDictHasKey("error", result)
        self.assertDictHasKeyContains("error", result, "Invalid regex")


class TestEditorFindPrev(TestEditorBase):
    """Test find_prev functionality."""

    def test_find_prev_found(self):
        """Test finding previous occurrence of pattern."""
        self.init_editor_llm()

        # Go to end first
        goto_tool = tool_editor.EditorToolGoto()
        goto_tool(line_number=10)

        # Now find previous "line"
        find_tool = tool_editor.EditorToolFindPrev()
        result = find_tool(pattern=r"li.e \d")

        # Should find the pattern
        self.assertNotIn("error", result)
        self.assertIn("Pattern:", result, result)

    def test_find_prev_not_found(self):
        """Test finding previous occurrence when none exists."""
        self.init_editor_llm()

        find_tool = tool_editor.EditorToolFindPrev()
        result = find_tool(pattern=r"line \d")

        # Should report not found (nothing before line 1) - returns dict
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)
        assert isinstance(result, dict)
        self.assertEqual(result["status"], "not_found")


if __name__ == "__main__":
    unittest.main()
