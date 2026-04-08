"""Tests for insert_before and insert_after editor commands."""

import unittest
from llm import LLM, LlmInstace
import tool_editor
from tool_editor import EditorEntry, ToolEditor
from tests.test_helper import TestBase, tmpfilename


class TestEditorInsert(TestBase):
    """Test insert_before and insert_after commands."""

    FILE_TEST = tmpfilename(".agento.editor.insert.test")
    ID = 3000

    def setUp(self):
        super().setUp()
        ToolEditor.SKIP_PRINTING = True

        # Create test file with known content (5 lines)
        self.test_content = """LINE100
LINE101
LINE102
LINE103
LINE104
"""
        self.FILE_TEST.write_text(self.test_content)

    def tearDown(self):
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

        # Create editor LLM (simulating what ToolEditor.__call__ does)
        editor_llm = main_llm.clone()
        editor_llm.tools.clear()

        # Add editor tools
        ToolEditor.init_editor_tools(editor_llm)

        # Set up editor state
        llm_id = id(editor_llm)
        ToolEditor._state[llm_id] = EditorEntry(self.FILE_TEST.name, 1)

        # Prepare messages
        editor_msgs = [
            main_llm.msg_system("Editor mode system"),
            main_llm.msg_user(f"Editing {self.FILE_TEST.name}"),
        ]

        # Register this as current instance
        editor_llm.INSTANCES.append(LlmInstace(editor_llm, editor_msgs))

        return llm_id

    def test_insert_before_single_line(self):
        """Test inserting text before a single line pattern."""
        self.init_editor_llm()

        # Insert before LINE102
        insert_tool = tool_editor.EditorToolInsertBefore()
        result = insert_tool(text_to_find="LINE102", text_to_insert="Line101.5")

        # Should succeed (result is a string, not dict)
        assert isinstance(result, str)
        assert "Inserted 1 line(s)" in result

        # Read file to verify
        content = self.FILE_TEST.read_text()
        lines = content.splitlines()

        # Check that Line101.5 was inserted before LINE102
        assert "Line101.5" in lines
        line101_5_idx = lines.index("Line101.5")
        line102_idx = lines.index("LINE102")
        assert line101_5_idx == line102_idx - 1

    def test_insert_after_single_line(self):
        """Test inserting text after a single line pattern."""
        self.init_editor_llm()

        # Insert after LINE102
        insert_tool = tool_editor.EditorToolInsertAfter()
        result = insert_tool(text_to_find="LINE102", text_to_insert="Line102.5")

        assert isinstance(result, str)
        assert "Inserted 1 line(s)" in result

        # Read file to verify
        content = self.FILE_TEST.read_text()
        lines = content.splitlines()

        assert "Line102.5" in lines
        line102_idx = lines.index("LINE102")
        line102_5_idx = lines.index("Line102.5")
        assert line102_5_idx == line102_idx + 1

    def test_insert_before_multiline_pattern(self):
        """Test inserting text before a multiline pattern."""
        # Create file with multiline block
        content = """LINE100
LINE101
BLOCK_START
LINE102
BLOCK_END
LINE103
LINE104
"""
        self.FILE_TEST.write_text(content)
        self.init_editor_llm()

        # Insert before multiline pattern
        multiline_pattern = "BLOCK_START\nLINE102\nBLOCK_END"
        insert_tool = tool_editor.EditorToolInsertBefore()
        result = insert_tool(
            text_to_find=multiline_pattern, text_to_insert="BeforeBlock"
        )

        assert isinstance(result, str)
        assert "Inserted 1 line(s)" in result

        content = self.FILE_TEST.read_text()
        lines = content.splitlines()

        assert "BeforeBlock" in lines
        before_idx = lines.index("BeforeBlock")
        block_start_idx = lines.index("BLOCK_START")
        assert before_idx == block_start_idx - 1

    def test_insert_after_multiline_pattern(self):
        """Test inserting text after a multiline pattern."""
        # Create file with multiline block
        content = """LINE100
LINE101
BLOCK_START
LINE102
BLOCK_END
LINE103
LINE104
"""
        self.FILE_TEST.write_text(content)
        self.init_editor_llm()

        # Insert after multiline pattern
        multiline_pattern = "BLOCK_START\nLINE102\nBLOCK_END"
        insert_tool = tool_editor.EditorToolInsertAfter()
        result = insert_tool(
            text_to_find=multiline_pattern, text_to_insert="AfterBlock"
        )

        assert isinstance(result, str)
        assert "Inserted 1 line(s)" in result

        content = self.FILE_TEST.read_text()
        lines = content.splitlines()

        assert "AfterBlock" in lines
        block_end_idx = lines.index("BLOCK_END")
        after_idx = lines.index("AfterBlock")
        assert after_idx == block_end_idx + 1

    def test_insert_before_multiline_insert(self):
        """Test inserting multiple lines before a pattern."""
        self.init_editor_llm()

        # Insert multiple lines before LINE102
        multiline_insert = "Insert1\nInsert2\nInsert3"
        insert_tool = tool_editor.EditorToolInsertBefore()
        result = insert_tool(text_to_find="LINE102", text_to_insert=multiline_insert)

        assert isinstance(result, str)
        assert "Inserted 3 line(s)" in result

        content = self.FILE_TEST.read_text()
        lines = content.splitlines()

        assert "Insert1" in lines
        assert "Insert2" in lines
        assert "Insert3" in lines

        # Verify order
        insert1_idx = lines.index("Insert1")
        insert2_idx = lines.index("Insert2")
        insert3_idx = lines.index("Insert3")
        line102_idx = lines.index("LINE102")

        assert insert1_idx == insert2_idx - 1
        assert insert2_idx == insert3_idx - 1
        assert insert3_idx == line102_idx - 1

    def test_insert_after_multiline_insert(self):
        """Test inserting multiple lines after a pattern."""
        self.init_editor_llm()

        # Insert multiple lines after LINE102
        multiline_insert = "Insert1\nInsert2\nInsert3"
        insert_tool = tool_editor.EditorToolInsertAfter()
        result = insert_tool(text_to_find="LINE102", text_to_insert=multiline_insert)

        assert isinstance(result, str)
        assert "Inserted 3 line(s)" in result

        content = self.FILE_TEST.read_text()
        lines = content.splitlines()

        assert "Insert1" in lines
        assert "Insert2" in lines
        assert "Insert3" in lines

        # Verify order
        line102_idx = lines.index("LINE102")
        insert1_idx = lines.index("Insert1")
        insert2_idx = lines.index("Insert2")
        insert3_idx = lines.index("Insert3")

        assert insert1_idx == line102_idx + 1
        assert insert2_idx == insert1_idx + 1
        assert insert3_idx == insert2_idx + 1

    def test_insert_before_pattern_not_found(self):
        """Test error when pattern is not found."""
        self.init_editor_llm()

        # Try to insert before non-existent pattern
        insert_tool = tool_editor.EditorToolInsertBefore()
        result = insert_tool(
            text_to_find="NONEXISTENT", text_to_insert="ShouldNotInsert"
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert "not found" in result["error"].lower()
        assert "0" in result["error"]

    def test_insert_after_pattern_not_found(self):
        """Test error when pattern is not found."""
        self.init_editor_llm()

        # Try to insert after non-existent pattern
        insert_tool = tool_editor.EditorToolInsertAfter()
        result = insert_tool(
            text_to_find="NONEXISTENT", text_to_insert="ShouldNotInsert"
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert "not found" in result["error"].lower()
        assert "0" in result["error"]

    def test_insert_before_pattern_multiple_occurrences(self):
        """Test error when pattern appears multiple times."""
        # Create file with duplicate pattern
        content = """LINE100
LINE101
LINE102
LINE102
LINE104
"""
        self.FILE_TEST.write_text(content)
        self.init_editor_llm()

        # Try to insert before pattern that appears twice
        insert_tool = tool_editor.EditorToolInsertBefore()
        result = insert_tool(text_to_find="LINE102", text_to_insert="ShouldNotInsert")

        assert isinstance(result, dict)
        assert "error" in result
        assert "2" in result["error"]
        assert "exactly once" in result["error"].lower()

    def test_insert_after_pattern_multiple_occurrences(self):
        """Test error when pattern appears multiple times."""
        # Create file with duplicate pattern
        content = """LINE100
LINE101
LINE102
LINE102
LINE104
"""
        self.FILE_TEST.write_text(content)
        self.init_editor_llm()

        # Try to insert after pattern that appears twice
        insert_tool = tool_editor.EditorToolInsertAfter()
        result = insert_tool(text_to_find="LINE102", text_to_insert="ShouldNotInsert")

        assert isinstance(result, dict)
        assert "error" in result
        assert "2" in result["error"]
        assert "exactly once" in result["error"].lower()

    def test_insert_before_at_start_of_file(self):
        """Test inserting at the very beginning of file."""
        self.init_editor_llm()

        # Insert before first line
        insert_tool = tool_editor.EditorToolInsertBefore()
        result = insert_tool(text_to_find="LINE100", text_to_insert="FirstLine")

        assert isinstance(result, str)
        assert "Inserted 1 line(s)" in result

        content = self.FILE_TEST.read_text()
        lines = content.splitlines()

        assert lines[0] == "FirstLine"
        assert lines[1] == "LINE100"

    def test_insert_after_at_end_of_file(self):
        """Test inserting at the very end of file."""
        self.init_editor_llm()

        # Insert after last line
        insert_tool = tool_editor.EditorToolInsertAfter()
        result = insert_tool(text_to_find="LINE104", text_to_insert="LastLine")

        assert isinstance(result, str)
        assert "Inserted 1 line(s)" in result

        content = self.FILE_TEST.read_text()
        lines = content.splitlines()

        assert lines[-1] == "LastLine"
        assert lines[-2] == "LINE104"


if __name__ == "__main__":
    unittest.main()
