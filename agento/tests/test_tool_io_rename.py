import unittest
import shutil

from agento.tool import io as tool_io
from agento.tests.test_helper import TestBase, tmpfilename


class TestToolRenameConstraints(TestBase):
    """Test ToolRename constraint validation."""

    def setUp(self):
        super().setUp()

    def test_rename_source_not_exists(self):
        """Test that rename fails when source file doesn't exist."""
        result = tool_io.ToolRename()("nonexistent.txt", "dest.txt")
        assert isinstance(result, dict)
        self.assertEqual(result["nonexistent.txt"], "error")
        self.assertIn("doesn't exist", result["error"])

    def test_rename_source_is_directory(self):
        """Test that rename fails when source is a directory."""
        # Create a directory
        test_dir = tmpfilename("test_dir")
        try:
            test_dir.mkdir(parents=True, exist_ok=True)
            result = tool_io.ToolRename()("test_dir", "dest.txt")
            assert isinstance(result, dict)
            self.assertEqual(result["test_dir"], "error")
            self.assertIn("not a file", result["error"])
        finally:
            shutil.rmtree(test_dir)

    def test_rename_dest_exists(self):
        """Test that rename fails when destination already exists."""
        # Create destination file
        dest_file = tmpfilename("dest.txt")
        try:
            dest_file.write_text("existing content")
            result = tool_io.ToolRename()(self.FILE_FOO.name, dest_file.name)
            assert isinstance(result, dict)
            self.assertEqual(result["dest.txt"], "error")
            self.assertIn("already exists", result["error"])
        finally:
            dest_file.unlink()

    def test_rename_success(self):
        """Test successful rename operation."""
        # Create source file
        dest_name = "rename_dest.txt"

        src_file = tmpfilename("rename_src.txt")
        dest_file = tmpfilename(dest_name)
        try:
            src_file.write_text("content to rename")
            result = tool_io.ToolRename()(src_file.name, dest_name)
            self.assertIn(" OK:", result)
            self.assertFalse(src_file.exists())
            self.assertTrue(dest_file.exists())
            self.assertEqual(dest_file.read_text(), "content to rename")
        finally:
            src_file.unlink(True)
            dest_file.unlink(True)


if __name__ == "__main__":
    unittest.main()
