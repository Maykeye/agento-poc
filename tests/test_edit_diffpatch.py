import unittest
import tool_io
from context import context_handler
from tests.test_helper import TestBase, tmpfilename


class TestDiffPatch(TestBase):
    """Tests for the edit_diff_patch tool."""

    def setUp(self):
        super().setUp()
        # Initialize LLM properly like other tests do
        self.llm, _ = self.init_llm_msgs()
        context_handler().prepare_current_llm(self.llm)

    def get_error_message(self, result) -> str:
        """Helper to extract error message from tool result."""
        if isinstance(result, dict):
            return result.get("error", "")
        return str(result)

    def normalize_patch(self, patch: str) -> str:
        """Normalize a patch by stripping trailing whitespace but keep newlines."""
        # Split lines but preserve trailing whitespace on deletion lines for matching
        raw_lines = patch.splitlines()
        lines = []
        for line in raw_lines:
            if line.startswith("-"):
                # Keep deletion lines as-is (preserve trailing whitespace for matching)
                lines.append(line)
            else:
                # Strip trailing whitespace from other lines
                lines.append(line.rstrip())

        # Add leading space to context lines (lines that don't start with -, +, or are headers)
        normalized_lines = []
        for line in lines:
            if line.startswith("--- ") or line.startswith("+++ "):
                # Header lines - keep as is
                normalized_lines.append(line)
            elif line.startswith("@@"):
                # Hunk header - keep as is
                normalized_lines.append(line)
            elif line.startswith("-") or line.startswith("+"):
                # Change lines (- for deletion, + for addition) - keep as is
                normalized_lines.append(line)
            else:
                # Context line - must start with space in unified diff format
                if line and not line.startswith(" "):
                    line = " " + line
                normalized_lines.append(line)

        return "\n".join(normalized_lines) + "\n"

    def test_edit_diff_patch_success(self):
        """Test successful patch application."""
        # Ensure file has trailing newline for patch to work
        self.FILE_FOO.write_text("foo\ntext\n")

        patch = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,2 +1,2 @@
 foo
-text
+modified
"""
        # Normalize the patch to ensure proper unified diff format
        patch = self.normalize_patch(patch)

        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

        self.assertIn("PATCH APPLIED", result)
        self.assertIn("edit_diff_patch", result)

        # Verify the file was actually modified (with trailing newline)
        self.assertEqual(self.FILE_FOO.read_text(), "foo\nmodified\n")

    def test_edit_diff_patch_empty_file(self):
        """Test patching an empty file."""
        # Create an empty file for this test
        empty_file = tmpfilename(".empty.test")
        empty_file.write_text("")

        try:
            patch = self.normalize_patch(f"""--- a/{empty_file.name}
+++ b/{empty_file.name}
@@ -0,0 +1 @@
+new content""")
            result = tool_io.ToolEditDiffPatch()(empty_file.name, patch)

            self.assertIn("PATCH APPLIED", result)
            # patch command adds trailing newline to file content
            self.assertEqual(empty_file.read_text(), "new content\n")
        finally:
            empty_file.unlink(True)

    def test_edit_diff_patch_invalid_format_no_header(self):
        """Test patch validation - missing header lines."""
        invalid_patch = "just some text\nno headers here"

        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, invalid_patch)

        self.assertIn("error", str(result))
        error_msg = self.get_error_message(result)
        self.assertIn("First line must start with '--- a/'", error_msg)

    def test_edit_diff_patch_invalid_format_wrong_first_line(self):
        """Test patch validation - wrong first line format."""
        invalid_patch = """@@ -1,2 +1,2 @@
-foo
+bar"""

        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, invalid_patch)

        self.assertIn("error", str(result))
        error_msg = self.get_error_message(result)
        self.assertIn("First line must start with '--- a/'", error_msg)

    def test_edit_diff_patch_invalid_format_wrong_second_line(self):
        """Test patch validation - wrong second line format."""
        invalid_patch = """--- a/foo
@@ -1,2 +1,2 @@
-foo
+bar"""

        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, invalid_patch)

        self.assertIn("error", str(result))
        error_msg = self.get_error_message(result)
        self.assertIn("Second line must start with '+++ b/'", error_msg)

    def test_edit_diff_patch_multiple_files(self):
        """Test patch validation - multiple files in patch."""
        # Create a patch that appears to have two files
        invalid_patch = self.normalize_patch(f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,2 +1,2 @@
-foo
+bar
--- a/bar
+++ b/bar
@@ -1 +1 @@
-bar
+baz""")

        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, invalid_patch)

        self.assertIn("error", str(result))
        error_msg = self.get_error_message(result)
        self.assertIn("Patch contains multiple files", error_msg)

    def test_edit_diff_patch_path_mismatch(self):
        """Test patch validation - path mismatch between patch and requested file."""
        # Create a patch for a different file
        patch = self.normalize_patch(f"""--- a/different_file.txt
+++ b/different_file.txt
@@ -1,2 +1,2 @@
-bar
+modified""")

        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

        self.assertIn("error", str(result))
        error_msg = self.get_error_message(result)
        self.assertIn("does not match requested path", error_msg)

    def test_edit_diff_patch_empty_patch(self):
        """Test patch validation - empty patch."""
        empty_patch = ""

        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, empty_patch)

        self.assertIn("error", str(result))
        error_msg = self.get_error_message(result)
        self.assertIn("Patch must have at least two lines", error_msg)

    def test_edit_diff_patch_single_line(self):
        """Test patch validation - single line patch."""
        single_line = "--- a/foo"

        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, single_line)

        self.assertIn("error", str(result))
        error_msg = self.get_error_message(result)
        self.assertIn("Patch must have at least two lines", error_msg)

    def test_edit_diff_patch_with_context(self):
        """Test patch application with context lines."""
        # Multi-line file for testing context - must have trailing newline
        original = "line1\nline2\nline3\nline4\nline5\n"
        self.FILE_FOO.write_text(original)

        # Patch changes line2 to "modified line2", with context lines around it
        patch = self.normalize_patch(f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,5 +1,5 @@
 line1
-line2
+modified line2
 line3
 line4
 line5""")

        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

        self.assertIn("PATCH APPLIED", result)
        self.assertEqual(
            self.FILE_FOO.read_text(), "line1\nmodified line2\nline3\nline4\nline5\n"
        )

    def test_edit_diff_patch_deletion(self):
        """Test patch with deletion."""
        original = "keep this\nremove this\nkeep also"
        self.FILE_FOO.write_text(original)

        patch = self.normalize_patch(f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,3 +1,2 @@
 keep this
-remove this
 keep also""")

        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

        self.assertIn("PATCH APPLIED", result)
        self.assertEqual(self.FILE_FOO.read_text(), "keep this\nkeep also")

    def test_edit_diff_patch_insertion(self):
        """Test patch with insertion."""
        original = "line1\nline3"
        self.FILE_FOO.write_text(original)

        patch = self.normalize_patch(f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,2 +1,3 @@
 line1
+inserted line
 line3""")

        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

        self.assertIn("PATCH APPLIED", result)
        self.assertEqual(self.FILE_FOO.read_text(), "line1\ninserted line\nline3")

    def test_edit_diff_patch_multiple_changes(self):
        """Test patch with multiple changes."""
        # Ensure file has trailing newline for patch to work
        original = "first\nsecond\nthird\n"
        self.FILE_FOO.write_text(original)

        new_content = "FIRST\nSECOND\nTHIRD\n"
        patch = self.normalize_patch(f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,3 +1,3 @@
-first
+FIRST
-second
+SECOND
-third
+THIRD""")

        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

        self.assertIn("PATCH APPLIED", result)
        self.assertEqual(self.FILE_FOO.read_text(), new_content)

    def test_edit_diff_patch_no_changes(self):
        """Test patch with no net changes (trivial change that results in same content)."""
        # File must have trailing newline for patch to work
        original = "same content\n"
        self.FILE_FOO.write_text(original)

        # Create a patch with a trivial change that results in the same content
        # (delete and add the same line)
        patch = self.normalize_patch(f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1 +1 @@
-same content
+same content""")

        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

        # Patch should apply successfully even though the content doesn't change
        self.assertIn("PATCH APPLIED", result)
        self.assertEqual(self.FILE_FOO.read_text(), original)

    def test_edit_diff_patch_invalid_hunk(self):
        """Test patch with invalid hunk that won't apply."""
        # Try to modify a line that doesn't exist
        original = "only one line"
        self.FILE_FOO.write_text(original)

        # This patch tries to delete line 5 which doesn't exist
        bad_patch = self.normalize_patch(f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,10 +1,10 @@
-incorrect match
+correct match""")

        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, bad_patch)

        self.assertIn("error", str(result))
        # The patch should fail because it can't find the line to delete
        error_msg = self.get_error_message(result)
        self.assertTrue(len(error_msg) > 0)

    def test_edit_diff_patch_whitespace_only(self):
        """Test patch with whitespace-only changes."""
        # File must have trailing newline for patch to work
        original = "  spaces  \n\ttabs\t\n"
        self.FILE_FOO.write_text(original)

        new_content = "no spaces\nno tabs\n"
        patch = self.normalize_patch(f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,2 +1,2 @@
-  spaces  
+no spaces
-\ttabs\t
+no tabs""")

        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

        self.assertIn("PATCH APPLIED", result)
        self.assertEqual(self.FILE_FOO.read_text(), new_content)

    def test_edit_diff_patch_prefix_context(self):
        """Test patch application in PREFIX context mode."""
        from context import set_context_mode, ContextMode

        # Save original mode and set to PREFIX
        original_handler = context_handler()
        set_context_mode(ContextMode.PREFIX, reset_ctx_id=True)

        try:
            # Ensure file has trailing newline for patch to work
            self.FILE_FOO.write_text("foo\ntext\n")

            patch = self.normalize_patch(f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,2 +1,2 @@
 foo
-text
+modified
""")

            result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

            # Verify the result contains PATCH APPLIED
            self.assertIn("PATCH APPLIED", result)
            self.assertIn("edit_diff_patch", result)

            # Verify the file was actually modified
            self.assertEqual(self.FILE_FOO.read_text(), "foo\nmodified\n")
        finally:
            # Restore original handler
            set_context_mode(original_handler.mode(), reset_ctx_id=False)

    def test_edit_diff_patch_suffix_context(self):
        """Test patch application in SUFFIX context mode."""
        from context import set_context_mode, ContextMode

        # Save original mode and set to SUFFIX
        original_handler = context_handler()
        set_context_mode(ContextMode.SUFFIX, reset_ctx_id=True)

        try:
            # Ensure file has trailing newline for patch to work
            self.FILE_FOO.write_text("foo\ntext\n")

            patch = self.normalize_patch(f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,2 +1,2 @@
 foo
-text
+modified
""")

            result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

            # Verify the result contains PATCH APPLIED in the tool response
            # and the SUFFIX context markers
            self.assertIn("PATCH APPLIED", result)
            self.assertIn("edit_diff_patch", result)

            # Verify the file was actually modified
            self.assertEqual(self.FILE_FOO.read_text(), "foo\nmodified\n")
        finally:
            # Restore original handler
            set_context_mode(original_handler.mode(), reset_ctx_id=False)


if __name__ == "__main__":
    unittest.main()
