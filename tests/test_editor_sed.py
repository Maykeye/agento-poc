import unittest
from tests.test_helper import TestBase
import tool.editor.sed as tool_sed


class TestSed(TestBase):
    def setUp(self):
        super().setUp()
        self.tool = tool_sed.EditorToolSedEdit()
        self.tool.debug_assumed_decision = "<CONFIRMATION>\nAPPLY\n</CONFIRMATION>"
        self.init_editor_llm()

    def test_sed_basic_substitution(self):
        """Test basic sed substitution on FILE_FOO."""
        self.FILE_FOO.write_text("hello world\nfoo bar\n")
        result = self.tool(path=self.FILE_FOO.name, script="s/hello/HELLO/")

        self.assertEqual(result["result"], "applied")
        content = self.FILE_FOO.read_text()
        self.assertIn("HELLO world", content)
        self.assertIn("foo bar", content)

    def test_sed_no_match(self):
        """Test sed when pattern doesn't match - file unchanged."""
        self.FILE_FOO.write_text("hello world\n")
        original = self.FILE_FOO.read_text()
        result = self.tool(path=self.FILE_FOO.name, script="s/nonexistent/REPLACEMENT/")
        self.assertEqual(result["result"], "applied")
        content = self.FILE_FOO.read_text()
        self.assertEqual(content, original)

    def test_sed_file_not_found(self):
        """Test sed on non-existent file."""
        result = self.tool(path="nonexistent.txt", script="s/a/b/")

        assert isinstance(result, dict)
        self.assertIn("error", result)

    def test_sed_returns_diff(self):
        """Test that sed returns diff output."""
        self.FILE_FOO.write_text("line1\nline2\n")
        result = self.tool(path=self.FILE_FOO.name, script="s/line1/LINE1/")

        self.assertIn("diff", result)
        self.assertIn("LINE1", result["diff"])

    def test_sed_global_substitution(self):
        """Test sed with global flag."""
        self.FILE_FOO.write_text("foo foo foo\nbar foo bar\n")
        result = self.tool(path=self.FILE_FOO.name, script="s/foo/FOO/g")

        self.assertEqual(result["result"], "applied")
        content = self.FILE_FOO.read_text()
        self.assertIn("FOO FOO FOO", content)
        self.assertIn("bar FOO bar", content)

    def test_sed_delete_line(self):
        """Test sed deleting a line."""
        self.FILE_FOO.write_text("keep\nremove\nkeep\n")
        result = self.tool(path=self.FILE_FOO.name, script="/remove/d")

        self.assertEqual(result["result"], "applied")
        content = self.FILE_FOO.read_text()
        self.assertNotIn("remove", content)
        self.assertIn("keep", content)


if __name__ == "__main__":
    unittest.main()
