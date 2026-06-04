import unittest

from agento.tool import io as tool_io
from agento.tests.test_helper import TestBase, tmpfilename
from agento.utils import TEMP_DIR
import os


class TestLs(TestBase):
    """Tests for ls tool with glob support."""

    TEST_DIR = tmpfilename("_test_ls")

    def setUp(self):
        super().setUp()

        tool_io.ToolMkDir()("_test_ls")
        tool_io.ToolMkDir()("_test_ls/foobar")
        tool_io.ToolMkDir()("_test_ls/barfoo")
        tool_io.ToolMkDir()("_test_ls/barfoo/deep")
        tool_io.ToolWriteFile()("_test_ls/foobar/f1.rs", "")
        tool_io.ToolWriteFile()("_test_ls/foobar/f2.rs", "")
        tool_io.ToolWriteFile()("_test_ls/foobar/f2.txt", "")
        tool_io.ToolWriteFile()("_test_ls/barfoo/f3.rs", "")
        tool_io.ToolWriteFile()("_test_ls/barfoo/f4.rs", "")
        tool_io.ToolWriteFile()("_test_ls/barfoo/deep/f5.rs", "")

    def tearDown(self):
        files_to_delete = [
            "_test_ls/foobar/f1.rs",
            "_test_ls/foobar/f2.rs",
            "_test_ls/foobar/f2.txt",
            "_test_ls/barfoo/f3.rs",
            "_test_ls/barfoo/f4.rs",
            "_test_ls/barfoo/deep/f5.rs",
        ]
        for f in files_to_delete:
            tool_io.ToolDeleteFile()(f)

        # Delete directories
        dirs_to_delete = [
            "_test_ls/barfoo/deep",
            "_test_ls/barfoo",
            "_test_ls/foobar",
            "_test_ls",
        ]
        for d in dirs_to_delete:
            tool_io.ToolRmDir()(d)

        super().tearDown()

    def test_glob_recursive_rs(self):
        """Test _test_ls/**/*.rs - should find 5 rs files (f1..f5)"""
        result = tool_io.ToolLs()(["_test_ls/**/*.rs"])

        # Check that all 5 .rs files are reported
        self.assertIn("f1.rs", result)
        self.assertIn("f2.rs", result)
        self.assertIn("f3.rs", result)
        self.assertIn("f4.rs", result)
        self.assertIn("f5.rs", result)

        # Check that full paths are reported
        self.assertIn("_test_ls/foobar/f1.rs", result)
        self.assertIn("_test_ls/foobar/f2.rs", result)
        self.assertIn("_test_ls/barfoo/f3.rs", result)
        self.assertIn("_test_ls/barfoo/f4.rs", result)
        self.assertIn("_test_ls/barfoo/deep/f5.rs", result)

        # Count .rs files - should be exactly 5 (count only actual file entries)
        # Filter to only count "File:" lines with .rs
        lines = result.split("\n")
        rs_files = [l for l in lines if "File:" in l and ".rs" in l]
        self.assertEqual(
            len(rs_files), 5, f"Expected 5 .rs files, found {len(rs_files)}"
        )

    def test_glob_non_recursive_rs(self):
        """Test _test_ls/*.rs - should find nothing (no .rs files directly in _test_ls)"""
        result = tool_io.ToolLs()(["_test_ls/*.rs"])

        # Should find no files (no .rs files directly in _test_ls)
        # The result should indicate no files found or be empty
        self.assertNotIn("f1.rs", result)
        self.assertNotIn("f2.rs", result)
        self.assertNotIn("f3.rs", result)
        self.assertNotIn("f4.rs", result)
        self.assertNotIn("f5.rs", result)

    def test_glob_barfoo_rs(self):
        """Test _test_ls/barfoo/*.rs - should find f3 and f4 only"""
        result = tool_io.ToolLs()(["_test_ls/barfoo/*.rs"])

        # Should find f3.rs and f4.rs (not f5.rs which is in deep/)
        self.assertIn("f3.rs", result)
        self.assertIn("f4.rs", result)
        self.assertIn("barfoo", result)
        self.assertNotIn("f5.rs", result)
        self.assertNotIn("f1.rs", result)
        self.assertNotIn("f2.rs", result)

        # Check full paths
        self.assertIn("_test_ls/barfoo/f3.rs", result)
        self.assertIn("_test_ls/barfoo/f4.rs", result)

        # Count .rs files - should be exactly 2 (count only actual file entries)
        lines = result.split("\n")
        rs_files = [l for l in lines if "File:" in l and ".rs" in l]
        self.assertEqual(
            len(rs_files), 2, f"Expected 2 .rs files, found {len(rs_files)}"
        )

    def test_glob_nonexistent_dir(self):
        """Test _test_ls/barbar - should report error"""
        result = tool_io.ToolLs()(["_test_ls/barbar"])

        # Should report an error for non-existent path
        self.assertIn("does not exist", result.lower())
        self.assertIn("barbar", result)

    def test_glob_barfoo_from_non_project_directory(self):
        result1 = tool_io.ToolLs()(["_test_ls/barfoo/*.rs"])
        os.chdir("..")
        result2 = tool_io.ToolLs()(["_test_ls/barfoo/*.rs"])
        self.assertEqual(result1, result2)

    def test_absolute_path_not_reported_without_glob(self):
        result = tool_io.ToolLs()(["."])
        self.assertNotIn(str(TEMP_DIR), result)

    def test_absolute_path_not_reported_with_glob(self):
        result = tool_io.ToolLs()(["**/*.rs"])
        self.assertNotIn(str(TEMP_DIR), result)


if __name__ == "__main__":
    unittest.main()
