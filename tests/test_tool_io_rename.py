import unittest
import os
from pathlib import Path
import shutil

from config import CONFIG
from context.suffix import SuffixHandler
from tool import io as tool_io
from context import context, context_handler, ContextMode
from context.prefix import CONTEXTS as PREFIX_CONTEXTS
from tests.test_helper import TestBase, tmpfilename


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


class TestToolRenamePrefix(TestBase):
    """Test ToolRename with prefix context handler."""

    def setUp(self):
        super().setUp()
        PREFIX_CONTEXTS.clear()
        context.set_context_mode(ContextMode.PREFIX)

    def test_rename_file_not_read(self):
        """Test rename when file was not read in prefix context."""
        try:
            result = tool_io.ToolRename()(self.FILE_FOO.name, "new_foo.txt")
            self.assertIsInstance(result, str)
            self.assertIn(" OK:", result)
            self.assertNotIn(self.FILE_FOO.name, PREFIX_CONTEXTS)
            self.assertNotIn("new_foo.txt", PREFIX_CONTEXTS)
            self.assertFalse(self.FILE_FOO.exists())
            self.assertTrue(Path("new_foo.txt").exists())
        finally:
            Path("new_foo.txt").unlink()

    def test_rename_file_was_read(self):
        """Test rename when file was read in prefix context."""
        # Read the file first

        context_handler().update(self.FILE_FOO.name, "foo\ntext", "read_file")

        try:
            result = tool_io.ToolRename()(self.FILE_FOO.name, "new_foo.txt")

            self.assertIsInstance(result, str)
            self.assertIn(" OK:", result)
            self.assertNotIn(self.FILE_FOO.name, PREFIX_CONTEXTS)
            self.assertIn("new_foo.txt", PREFIX_CONTEXTS)
            self.assertEqual(PREFIX_CONTEXTS["new_foo.txt"].text, "foo\ntext")
            self.assertEqual(PREFIX_CONTEXTS["new_foo.txt"].operation, "rename_file")
            self.assertFalse(self.FILE_FOO.exists())
            self.assertTrue(Path("new_foo.txt").exists())
        finally:
            Path("new_foo.txt").unlink(missing_ok=True)

    def test_rename_file_was_read_twice(self):
        """Test rename when file was read, then renamed twice."""

        # Read the file first
        context_handler().update(self.FILE_FOO.name, "foo\ntext", "read_file")
        self.assertIn(self.FILE_FOO.name, PREFIX_CONTEXTS)
        try:
            # First rename
            result1 = tool_io.ToolRename()(self.FILE_FOO.name, "new_foo.txt")
            self.assertIsInstance(result1, str)
            self.assertIn("OK", result1)
            self.assertNotIn(self.FILE_FOO.name, PREFIX_CONTEXTS)
            self.assertIn("new_foo.txt", PREFIX_CONTEXTS)
            self.assertEqual(PREFIX_CONTEXTS["new_foo.txt"].operation, "rename_file")

            # Second rename
            result2 = tool_io.ToolRename()("new_foo.txt", "final_foo.txt")
            self.assertIsInstance(result2, str)
            self.assertIn("OK", result2)
            self.assertNotIn("new_foo.txt", PREFIX_CONTEXTS)
            self.assertIn("final_foo.txt", PREFIX_CONTEXTS)
            self.assertEqual(PREFIX_CONTEXTS["final_foo.txt"].text, "foo\ntext")
            self.assertEqual(PREFIX_CONTEXTS["final_foo.txt"].operation, "rename_file")

            # Check file was actually renamed
            self.assertFalse(self.FILE_FOO.exists())
            self.assertFalse(Path("new_foo.txt").exists())
            self.assertTrue(Path("final_foo.txt").exists())
        finally:
            Path("final_foo.txt").unlink(missing_ok=True)
            Path("new_foo.txt").unlink(missing_ok=True)


class TestToolRenameSuffix(TestBase):
    """Test ToolRename with suffix context handler."""

    def setUp(self):
        super().setUp()

        context.set_context_mode(ContextMode.SUFFIX)
        self.suffix_context: SuffixHandler = context.context_handler()  # type: ignore
        self.suffix_context.llm_to_file_entries.clear()
        os.chdir(tmpfilename(""))
        CONFIG.project_directory = tmpfilename("")

    def test_rename_file_not_read(self):
        """Test rename when file was not read in suffix context."""
        # Don't read the file, just rename
        try:
            result = tool_io.ToolRename()(self.FILE_FOO.name, "new_foo.txt")

            # Check result
            self.assertIsInstance(result, str)
            self.assertIn("OK", result)

            self.assertNotIn(self.FILE_FOO.name, self.suffix_context.file_entries())
            self.assertNotIn("new_foo.txt", self.suffix_context.file_entries())

            # Check file was actually renamed
            self.assertFalse(self.FILE_FOO.exists())
            self.assertTrue(Path("new_foo.txt").exists())
        finally:
            Path("new_foo.txt").unlink(missing_ok=True)

    def test_rename_file_was_read(self):
        """Test rename when file was read in suffix context."""

        # Read the file first
        context_handler().update(self.FILE_FOO.name, "foo\ntext", "read_file")

        self.assertIn(self.FILE_FOO.name, self.suffix_context.file_entries())
        try:
            # Rename the file
            result = tool_io.ToolRename()(self.FILE_FOO.name, "new_foo.txt")

            # Check result
            self.assertIsInstance(result, str)
            self.assertIn("OK", result)

            # Check SUFFIX_CONTEXTS was updated
            entries = self.suffix_context.file_entries()
            self.assertNotIn(self.FILE_FOO.name, entries)
            self.assertIn("new_foo.txt", entries)
            self.assertEqual(entries["new_foo.txt"].text, "foo\ntext")
            self.assertEqual(entries["new_foo.txt"].operation, "rename_file")

            # Check file was actually renamed
            self.assertFalse(self.FILE_FOO.exists())
            self.assertTrue(Path("new_foo.txt").exists())
        finally:
            Path("new_foo.txt").unlink(missing_ok=True)

    def test_rename_twice_file_was_read(self):
        """Test rename when file was read, then renamed twice."""
        # Read the file first
        FILE_DATA = "TEST-FILE-CONTENT"
        self.FILE_FOO.write_text(FILE_DATA)
        _, msgs = self.init_llm_msgs()
        self.tool_call_read(self.FILE_FOO)
        entries = self.suffix_context.file_entries()

        self.assertIn(self.FILE_FOO.name, entries)
        try:
            # First rename
            result = self.tool_call_rename(self.FILE_FOO, Path("new_foo.txt"))
            self.assertIsInstance(result, str)
            self.assertIn("OK", result)
            self.assertNotIn(self.FILE_FOO.name, entries)
            self.assertIn("new_foo.txt", entries)
            self.assertEqual(entries["new_foo.txt"].operation, "rename_file")

            # Second rename
            result = self.tool_call_rename(Path("new_foo.txt"), Path("final_foo.txt"))
            self.assertIsInstance(result, str)
            self.assertIn("OK", result)
            self.assertNotIn("new_foo.txt", entries)
            self.assertIn("final_foo.txt", entries)
            self.assertEqual(entries["final_foo.txt"].text, FILE_DATA)
            self.assertEqual(entries["final_foo.txt"].operation, "rename_file")

            # Check file was actually renamed
            self.assertFalse(self.FILE_FOO.exists())
            self.assertFalse(Path("new_foo.txt").exists())
            self.assertTrue(Path("final_foo.txt").exists())
            self.epilogue()

            # Check that model sees it
            counts = 0
            for msg in msgs:
                if msg.get("role") == "tool" and (txt := msg.get("content")):
                    if FILE_DATA in txt:
                        self.assertIn("final_foo", txt)
                        counts += 1
            self.assertEqual(counts, 1)

        finally:
            Path("final_foo.txt").unlink(missing_ok=True)
            Path("new_foo.txt").unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
