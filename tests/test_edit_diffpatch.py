from context import context_handler
from context import set_context_mode, ContextMode
from context import set_context_mode, ContextMode
from llm import LLM
from llm import LLM, LlmInstace
from pathlib import Path
from tests.test_helper import TestBase, tmpfilename
import tool_io
import unittest


class TestDiffPatch(TestBase):
    """Tests for the edit_diff_patch tool."""

    def setUp(self):
        super().setUp()
        # Initialize LLM properly like other tests do
        self.llm, _ = self.init_llm_msgs()
        context_handler().prepare_current_llm(self.llm)

    def tool_call_patch(self, path: str, patch: str):
        llm = LLM.INSTANCES[-1].llm
        msgs = LLM.INSTANCES[-1].messages
        msgs.append(llm.msg_assistant(f"Patch {path}"))
        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, patch)
        self.append_tool_call_result("edit_diff_patch", msgs, result)
        return result

    def get_error_message(self, result) -> str:
        """Helper to extract error message from tool result."""
        if isinstance(result, dict):
            return result.get("error", "")
        return str(result)

    def test_edit_diff_patch_success(self):
        """Test successful patch application."""
        # Ensure file has trailing newline for patch to work
        self.FILE_FOO.write_text("foo\ntext\n")

        patch = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,2 +1,2 @@
 foo
-text
+modified"""
        # Normalize the patch to ensure proper unified diff format
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
            patch = f"""--- a/{empty_file.name}
+++ b/{empty_file.name}
@@ -0,0 +1 @@
+new content"""
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
        invalid_patch = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,2 +1,2 @@
-foo
+bar
--- a/bar
+++ b/bar
@@ -1 +1 @@
-bar
+baz"""

        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, invalid_patch)

        self.assertIn("error", str(result))
        error_msg = self.get_error_message(result)
        self.assertIn("Patch contains multiple files", error_msg)

    def test_edit_diff_patch_path_mismatch(self):
        """Test patch validation - path mismatch between patch and requested file."""
        # Create a patch for a different file
        patch = f"""--- a/different_file.txt
+++ b/different_file.txt
@@ -1,2 +1,2 @@
-bar
+modified"""

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
        patch = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,5 +1,5 @@
 line1
-line2
+modified line2
 line3
 line4
 line5"""

        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

        self.assertIn("PATCH APPLIED", result)
        self.assertEqual(
            self.FILE_FOO.read_text(), "line1\nmodified line2\nline3\nline4\nline5\n"
        )

    def test_edit_diff_patch_deletion(self):
        """Test patch with deletion."""
        original = "keep this\nremove this\nkeep also"
        self.FILE_FOO.write_text(original)

        patch = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,3 +1,2 @@
 keep this
-remove this
 keep also"""

        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

        self.assertIn("PATCH APPLIED", result)
        self.assertEqual(self.FILE_FOO.read_text(), "keep this\nkeep also")

    def test_edit_diff_patch_insertion(self):
        """Test patch with insertion."""
        original = "line1\nline3"
        self.FILE_FOO.write_text(original)

        patch = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,2 +1,3 @@
 line1
+inserted line
 line3"""

        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

        self.assertIn("PATCH APPLIED", result)
        self.assertEqual(self.FILE_FOO.read_text(), "line1\ninserted line\nline3")

    def test_edit_diff_patch_multiple_changes(self):
        """Test patch with multiple changes."""
        # Ensure file has trailing newline for patch to work
        original = "first\nsecond\nthird\n"
        self.FILE_FOO.write_text(original)

        new_content = "FIRST\nSECOND\nTHIRD\n"
        patch = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,3 +1,3 @@
-first
+FIRST
-second
+SECOND
-third
+THIRD"""

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
        patch = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1 +1 @@
-same content
+same content"""

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
        bad_patch = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,10 +1,10 @@
-incorrect match
+correct match"""

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
        patch = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,2 +1,2 @@
-  spaces  
+no spaces
-\ttabs\t
+no tabs"""

        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

        self.assertIn("PATCH APPLIED", result)
        self.assertEqual(self.FILE_FOO.read_text(), new_content)

    def test_edit_diff_patch_prefix_context(self):
        """Test patch application in PREFIX context mode."""

        # Ensure file has trailing newline for patch to work
        self.FILE_FOO.write_text("foo\ntext\n")

        patch = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,2 +1,2 @@
 foo
-text
+modified
"""
        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, patch)
        self.assertIn("PATCH APPLIED", result)
        self.assertIn("edit_diff_patch", result)
        self.assertEqual(self.FILE_FOO.read_text(), "foo\nmodified\n")

    def test_edit_diff_patch_suffix_context(self):
        """Test patch application in SUFFIX context mode."""

        set_context_mode(ContextMode.SUFFIX, reset_ctx_id=True)

        # Ensure file has trailing newline for patch to work
        self.FILE_FOO.write_text("foo\ntext\n")

        patch = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,2 +1,2 @@
 foo
-text
+modified
"""

        result = tool_io.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

        # Verify the result contains PATCH APPLIED in the tool response
        # and the SUFFIX context markers
        self.assertIn("PATCH APPLIED", result)
        self.assertIn("edit_diff_patch", result)

        # Verify the file was actually modified
        self.assertEqual(self.FILE_FOO.read_text(), "foo\nmodified\n")

    def test_edit_diff_patch_suffix_three_patches(self):
        """Test applying three patches in SUFFIX mode and verify context handling.

        This test creates a dummy file with 20 lines, applies three patches sequentially,
        and verifies that:
        1. The last patch (CTX(2)) contains the full file content
        2. The first two patches (CTX(0) and CTX(1)) are marked as out-of-date
        """

        # Save original mode and set to SUFFIX
        original_handler = context_handler()
        set_context_mode(ContextMode.SUFFIX, reset_ctx_id=True)

        # Create dummy file with 20 lines
        dummy_content = "\n".join([f"line{i}" for i in range(1, 21)]) + "\n"
        self.FILE_FOO.write_text(dummy_content)

        # Initialize LLM properly using the same pattern as test_context_suffix
        dummy_llm = LLM()
        dummy_llm.INSTANCES.append(LlmInstace(dummy_llm, []))
        msgs = dummy_llm.INSTANCES[-1].messages
        msgs.append(dummy_llm.msg_user("Please apply three patches to the file"))
        context_handler().prepare_current_llm(dummy_llm)

        # First patch: Change line1 to "LINE1"
        patch1 = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,5 +1,5 @@
-line1
+LINE1
 line2
 line3
 line4
 line5"""

        msgs.append(dummy_llm.msg_assistant("Applying first patch"))
        result1 = self.tool_call_patch(self.FILE_FOO.name, patch1)
        result1_idx = len(msgs) - 1
        self.assertIn("PATCH APPLIED", result1)
        self.assertIn("CTX(0)", result1)

        # Second patch: Change line10 to "LINE10"
        patch2 = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -8,7 +8,7 @@
 line8
 line9
-line10
+LINE10
 line11
 line12
 line13"""
        result2 = self.tool_call_patch(self.FILE_FOO.name, patch2)
        result2_idx = len(msgs) - 1
        self.assertIn("PATCH APPLIED", result2)
        self.assertIn("CTX(1)", result2)
        self.epilogue()

        # Patch1: gone
        self.assertEqual(msgs[result1_idx].get("role"), "tool")
        content = msgs[result1_idx]["content"]
        self.assertIn("ID: CTX(0)", content)
        self.assertNotIn("CTX(1)", content)
        self.assertIn("OUT OF DATE", content)
        self.assertNotIn("line", content)

        # Third patch: Change line20 to "LINE20"
        patch3 = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -17,4 +17,4 @@
 line17
 line18
 line19
-line20
+LINE20"""
        # Patch1: still gone
        self.assertEqual(msgs[result1_idx].get("role"), "tool")
        content = msgs[result1_idx]["content"]
        self.assertIn("ID: CTX(0)", content)
        self.assertIn("OUT OF DATE", content)
        self.assertNotIn("line", content)

        result3 = self.tool_call_patch(self.FILE_FOO.name, patch3)
        result3_idx = len(msgs) - 1
        self.epilogue()
        self.assertIn("PATCH APPLIED", result3)
        self.assertIn("CTX(2)", result3)

        # Patch2: gone
        self.assertEqual(msgs[result2_idx].get("role"), "tool")
        content = msgs[result2_idx]["content"]
        self.assertIn("ID: CTX(1)", content)
        self.assertIn("OUT OF DATE", content)
        self.assertNotIn("line", content)

        # Patch3: Up to date
        self.assertEqual(msgs[result3_idx].get("role"), "tool")
        content = msgs[result3_idx]["content"]
        self.assertNotIn("line1\n", content)
        self.assertNotIn("line10", content)
        self.assertNotIn("line20", content)
        self.assertIn("LINE1\n", content)
        self.assertIn("LINE10", content)
        self.assertIn("LINE20", content)


if __name__ == "__main__":
    unittest.main()
