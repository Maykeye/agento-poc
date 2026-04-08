from context import context_handler
from context import set_context_mode, ContextMode
from llm import LLM, LlmInstace
from tests.test_helper import TestBase
import tool_edit_patch
import unittest


class TestDiffPatch(TestBase):
    """Tests for the edit_diff_patch tool."""

    def setUp(self):
        super().setUp()
        # Initialize LLM properly like other tests do
        self.llm, self.messages = self.init_llm_msgs()
        context_handler().prepare_current_llm(self.llm)
        set_context_mode(ContextMode.SUFFIX, reset_ctx_id=True)

    def tool_call_patch(self, path: str, patch: str):
        llm = LLM.INSTANCES[-1].llm
        msgs = LLM.INSTANCES[-1].messages
        msgs.append(llm.msg_assistant(f"Patch {path}"))
        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, patch)
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
        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

        self.assertIn("PATCH APPLIED", result)
        self.assertIn("edit_diff_patch", result)

        # Verify the file was actually modified (with trailing newline)
        self.assertEqual(self.FILE_FOO.read_text(), "foo\nmodified\n")

    def test_edit_diff_patch_empty_file(self):
        """Test patching an empty file."""
        self.FILE_FOO.write_text("")
        patch = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -0,0 +1 @@
+new content"""
        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

        self.assertIn("PATCH APPLIED", result)
        # patch command adds trailing newline to file content
        self.assertEqual(self.FILE_FOO.read_text(), "new content\n")

    def test_edit_diff_patch_invalid_format_no_header(self):
        """Test patch validation - missing header lines."""
        invalid_patch = "just some text\nno headers here"

        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, invalid_patch)

        self.assertIn("error", str(result))
        error_msg = self.get_error_message(result)
        self.assertIn("First line must start with '--- a/'", error_msg)

    def test_edit_diff_patch_invalid_format_wrong_first_line(self):
        """Test patch validation - wrong first line format."""
        invalid_patch = """@@ -1,2 +1,2 @@
-foo
+bar"""

        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, invalid_patch)

        self.assertIn("error", str(result))
        error_msg = self.get_error_message(result)
        self.assertIn("First line must start with '--- a/'", error_msg)

    def test_edit_diff_patch_invalid_format_wrong_second_line(self):
        """Test patch validation - wrong second line format."""
        invalid_patch = """--- a/foo
@@ -1,2 +1,2 @@
-foo
+bar"""

        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, invalid_patch)

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

        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, invalid_patch)

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

        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

        self.assertIn("error", str(result))
        error_msg = self.get_error_message(result)
        self.assertIn("does not match requested path", error_msg)

    def test_edit_diff_patch_empty_patch(self):
        """Test patch validation - empty patch."""
        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, "")
        self.assertIn("error", str(result))
        error_msg = self.get_error_message(result)
        self.assertIn("Patch must have at least two lines", error_msg)

    def test_edit_diff_patch_single_line(self):
        """Test patch validation - single line patch."""
        single_line = "--- a/foo"

        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, single_line)

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

        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

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

        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

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

        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

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

        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

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

        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

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

        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, bad_patch)

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

        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

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
        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, patch)
        self.assertIn("PATCH APPLIED", result)
        self.assertIn("edit_diff_patch", result)
        self.assertEqual(self.FILE_FOO.read_text(), "foo\nmodified\n")

    def test_edit_diff_patch_suffix_context(self):
        self.FILE_FOO.write_text("foo\ntext\n")

        patch = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,2 +1,2 @@
 foo
-text
+modified
"""

        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

        # Verify the result contains PATCH APPLIED in the tool response
        # and the SUFFIX context markers
        self.assertIn("PATCH APPLIED", result)
        self.assertIn("edit_diff_patch", result)

        # Verify the file was actually modified
        self.assertEqual(self.FILE_FOO.read_text(), "foo\nmodified\n")

    def test_edit_diff_patch_suffix_three_patches(self):
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

        # Patch1: gone (outdated, references CTX(1))
        self.assertEqual(msgs[result1_idx].get("role"), "tool")
        content = msgs[result1_idx]["content"]
        self.assertIn("ID: CTX(0)", content)
        self.assertIn("out of date", content)
        self.assertNotIn("line1\n", content)
        self.assertNotIn("line2\n", content)

        # Third patch: Change line20 to "LINE20"
        patch3 = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -17,4 +17,4 @@
 line17
 line18
 line19
-line20
+LINE20"""
        # Patch1: still gone (outdated, still references CTX(1) since CTX(2) doesn't exist yet)
        self.assertEqual(msgs[result1_idx].get("role"), "tool")
        content = msgs[result1_idx]["content"]
        self.assertIn("ID: CTX(0)", content)
        self.assertIn("out of date", content)

        result3 = self.tool_call_patch(self.FILE_FOO.name, patch3)
        result3_idx = len(msgs) - 1
        self.epilogue()
        self.assertIn("PATCH APPLIED", result3)
        self.assertIn("CTX(2)", result3)

        # Patch2: gone (outdated, references CTX(2))
        self.assertEqual(msgs[result2_idx].get("role"), "tool")
        content = msgs[result2_idx]["content"]
        self.assertIn("ID: CTX(1)", content)
        self.assertIn("out of date", content)

        # Patch3: Up to date
        self.assertEqual(msgs[result3_idx].get("role"), "tool")
        content = msgs[result3_idx]["content"]
        self.assertNotIn("line1\n", content)
        self.assertNotIn("line10", content)
        self.assertNotIn("line20", content)
        self.assertIn("LINE1\n", content)
        self.assertIn("LINE10", content)
        self.assertIn("LINE20", content)

    def impl_count_fix(self, lhs, rhs):
        set_context_mode(ContextMode.SUFFIX)
        self.FILE_FOO.write_text("line1\nline2\nline3\nline4")
        self.tool_call_read(self.FILE_FOO)
        self.tool_call_patch(
            self.FILE_FOO.name,
            f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,{lhs} +1,{rhs} @@
 line1
 line2
-line3
+line_changed
 line4""",
        )
        res_idx = len(self.messages) - 1
        self.assertNotIn("error", self.messages[-1]["content"])
        self.epilogue()
        res = self.messages[res_idx]["content"]
        self.assertIn("line_changed", res)

    def test_lhs_fix(self):
        self.impl_count_fix(2, 4)

    def test_rhs_fix(self):
        self.impl_count_fix(4, 2)

    def test_empty_hunk_is_nop(self):
        set_context_mode(ContextMode.SUFFIX)
        self.FILE_FOO.write_text("line1\nline2\nline3\nline4")
        self.tool_call_read(self.FILE_FOO)
        self.tool_call_patch(
            self.FILE_FOO.name,
            f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,4 +1,4 @@
 line1
 line2
 line3
 line4""",
        )
        res_idx = len(self.messages) - 1
        self.assertNotIn("error", self.messages[-1]["content"])
        self.epilogue()
        res = self.messages[res_idx]["content"]
        self.assertIn("warning", res)

    def call_remove_line_test(self, patch_range: range, remove: int):
        orig = "\n".join([f"LINE{i:02d}" for i in range(1, 10)]) + "\n"
        self.FILE_FOO.write_text(orig)
        patch = f"--- a/{self.FILE_FOO.name}\n"
        patch += f"+++ b/{self.FILE_FOO.name}\n"
        patch += f"@@ -{patch_range.start},5 +{patch_range.start},5 @@\n"
        for i in patch_range:
            pfx = "-" if i == remove else " "
            patch += f"{pfx}LINE0{i}\n"
        return tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, patch)

    def test_edit_diff_patch_bof_visible(self):
        result = self.call_remove_line_test(range(1, 8), 5)
        self.assertIn("PATCH APPLIED", result)
        expected = "\n".join(f"LINE0{n}" for n in "12346789") + "\n"
        self.assertEqual(self.FILE_FOO.read_text(), expected)

    def test_edit_diff_patch_bof_remove(self):
        result = self.call_remove_line_test(range(1, 8), 1)
        self.assertIn("PATCH APPLIED", result)
        expected = "\n".join(f"LINE0{n}" for n in "23456789") + "\n"
        self.assertEqual(self.FILE_FOO.read_text(), expected)

    def test_edit_diff_patch_eof_visible(self):
        result = self.call_remove_line_test(range(3, 10), 7)
        self.assertIn("PATCH APPLIED", result)
        expected = "\n".join(f"LINE0{n}" for n in "12345689") + "\n"
        self.assertEqual(self.FILE_FOO.read_text(), expected)

    def test_edit_diff_patch_eof_remove(self):
        result = self.call_remove_line_test(range(3, 10), 9)
        self.assertIn("PATCH APPLIED", result)
        expected = "\n".join(f"LINE0{n}" for n in "12345678") + "\n"
        self.assertEqual(self.FILE_FOO.read_text(), expected)

    def test_edit_diff_patch_ambiguous_single_hunk(self):
        """Test that a single ambiguous hunk fails.

        If original has multiple identical occurrences and only one hunk
        is provided, the patch should fail.
        """
        # Original file with duplicate content
        original = "A\n{}\nB\n{}\n"
        self.FILE_FOO.write_text(original)

        # Single hunk trying to replace {} - ambiguous which {}
        # (no context to disambiguate)
        patch = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -2,1 +2,1 @@
-{{}}
+[]"""

        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, patch)
        self.assertIn("error", str(result))
        # Should report failure to find all hunks
        error_msg = self.get_error_message(result)
        self.assertIn("could not find all hunks", error_msg)

    def test_edit_diff_patch_ambiguous_two_hunks_success(self):
        """Test that two hunks can disambiguate identical content.

        If original has multiple identical occurrences, and we have two hunks
        that specify which one to change (by matching surrounding context),
        the patch should succeed.
        """
        # Original file with duplicate content
        original = "A\n{}\nB\n{}\n"
        self.FILE_FOO.write_text(original)

        # Two hunks: first changes B to C, second changes {} after B to []
        # The second {} is uniquely identified because it's after the first change
        patch1 = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -2,3 +2,3 @@
 {{}}
-B
+C
 {{}}"""

        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, patch1)
        self.assertIn("PATCH APPLIED", result)
        # After first patch: A\n{}\nC\n{}\n
        self.assertEqual(self.FILE_FOO.read_text(), "A\n{}\nC\n{}\n")

        # Now change the {} after C (which is uniquely identified)
        patch2 = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -2,4 +2,4 @@
 {{}}
 C
-{{}}
+[]"""

        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, patch2)
        self.assertIn("PATCH APPLIED", result)
        # Final: A\n{}\nC\n[]\n
        self.assertEqual(self.FILE_FOO.read_text(), "A\n{}\nC\n[]\n")

    def test_edit_diff_patch_ambiguous_two_hunks_fail(self):
        """Test that two hunks still fail when ambiguity remains.

        Even with two hunks, if the pattern matches multiple locations and
        the hunks don't provide enough context to disambiguate, it should fail.
        """
        # Original file with duplicate content
        original = "X\n{}\n{}\nY\n"
        self.FILE_FOO.write_text(original)

        # Two hunks: first changes X to Z, second tries to change {}
        # Both {} are between X and Y, so even after changing X, {} is still ambiguous
        patch = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,1 +1,1 @@
-X
+Z
@@ -2,1 +2,1 @@
-{{}}
+[]"""

        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, patch)
        self.assertIn("error", str(result))
        # Should report failure to find all hunks
        error_msg = self.get_error_message(result)
        self.assertIn("could not find all hunks", error_msg)

    def test_edit_diff_patch_overlapping_search(self):
        """Test patch fails when identical consecutive lines create overlap ambiguity.

        If original file has:
            AAA
            AAA
            AAA

        And patch tries to delete:
            -AAA
            -AAA

        We can't determine if AAA\nAAA are at lines 1-2 or 2-3.
        The patch should fail because the search is ambiguous.
        """
        # Original file with three identical lines
        original = "AAA\nAAA\nAAA\n"
        self.FILE_FOO.write_text(original)

        # Patch tries to delete two "AAA" lines
        # This is ambiguous - could be lines 1-2 or lines 2-3
        patch = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,3 +1,1 @@
-AAA
-AAA
"""

        result = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, patch)
        self.assertIn("error", str(result))
        # Should report failure to find all hunks due to overlapping matches
        error_msg = self.get_error_message(result)
        self.assertIn("could not find all hunks", error_msg)

    def test_edit_diff_patch_failed_logging(self):
        """Test that failed patches are logged to the database with correct history_id.

        When a patch fails, it should be logged to patch_fail table with history_id
        that corresponds to the tool call that generated it, not the max num from
        generation_history.
        """
        import utilsql

        # Create file with content
        self.FILE_FOO.write_text("line1\nline2\nline3\n")

        # Initialize LLM and messages
        dummy_llm, msgs = self.init_llm_msgs()
        context_handler().prepare_current_llm(dummy_llm)
        msgs.append(dummy_llm.msg_user("Apply two patches"))

        # First tool call - prepare messages
        msgs.append(dummy_llm.msg_assistant("First patch"))

        # Simulate tool call by setting last_tool_call_num
        # This simulates what LLM._generate does before calling tool
        dummy_llm.last_tool_call_num = utilsql.log_generation(
            utilsql.prompt_id(), dummy_llm.llm_id, msgs
        )
        first_call_history_id = dummy_llm.last_tool_call_num

        # Apply first broken patch (will fail)
        bad_patch1 = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,3 +1,3 @@
-wrong content
+correct content"""

        result1 = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, bad_patch1)
        self.assertIn("error", str(result1))

        # Second tool call - prepare messages
        msgs.append(dummy_llm.msg_assistant("Second patch"))

        # Simulate tool call by setting last_tool_call_num
        # This should be a new history_id but we'll use the SAME value
        # to test that the patch is linked to this call, not the previous one
        dummy_llm.last_tool_call_num = utilsql.log_generation(
            utilsql.prompt_id(), dummy_llm.llm_id, msgs
        )
        second_call_history_id = dummy_llm.last_tool_call_num

        # Apply second broken patch (will fail)
        bad_patch2 = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,3 +1,3 @@
-also wrong
+also correct"""

        result2 = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, bad_patch2)
        self.assertIn("error", str(result2))

        # Verify both patches are in the database
        with utilsql.sql_db() as db:
            # Check that both patches exist
            rows = db.execute(
                "SELECT history_id, orig, patch FROM patch_fail"
            ).fetchall()

            # We should have 2 failed patches
            self.assertEqual(len(rows), 2)

            # First patch should be linked to first_call_history_id
            first_patch_logged = [r for r in rows if r[0] == first_call_history_id]
            self.assertEqual(len(first_patch_logged), 1)
            self.assertIn(
                "wrong content", first_patch_logged[0][2]
            )  # patch contains "wrong content"

            # Second patch should be linked to second_call_history_id
            second_patch_logged = [r for r in rows if r[0] == second_call_history_id]
            self.assertEqual(len(second_patch_logged), 1)
            self.assertIn(
                "also wrong", second_patch_logged[0][2]
            )  # patch contains "also wrong"

            # Verify they are different history_ids
            self.assertNotEqual(first_call_history_id, second_call_history_id)

    def test_edit_diff_patch_failed_same_call(self):
        """Test that two failed patches in the same call use the same history_id.

        If LLM calls two patches at the same time and both fail, both should
        be linked to the same tool call history_id.
        """
        import utilsql

        # Create file with content
        self.FILE_FOO.write_text("line1\nline2\nline3\n")

        # Initialize LLM and messages
        dummy_llm, msgs = self.init_llm_msgs()
        context_handler().prepare_current_llm(dummy_llm)
        msgs.append(dummy_llm.msg_user("Apply two patches at once"))

        # Simulate a single tool call with last_tool_call_num set
        # This simulates what happens when multiple tool calls happen in the same generation
        msgs.append(dummy_llm.msg_assistant("Applying patches"))
        dummy_llm.last_tool_call_num = utilsql.log_generation(
            utilsql.prompt_id(), dummy_llm.llm_id, msgs
        )
        call_history_id = dummy_llm.last_tool_call_num

        # Apply first broken patch
        bad_patch1 = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,3 +1,3 @@
-wrong content
+correct content"""

        result1 = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, bad_patch1)
        self.assertIn("error", str(result1))

        # Apply second broken patch (same call, same history_id)
        bad_patch2 = f"""--- a/{self.FILE_FOO.name}
+++ b/{self.FILE_FOO.name}
@@ -1,3 +1,3 @@
-also wrong
+also correct"""

        result2 = tool_edit_patch.ToolEditDiffPatch()(self.FILE_FOO.name, bad_patch2)
        self.assertIn("error", str(result2))

        # Verify both patches are in the database with the SAME history_id
        with utilsql.sql_db() as db:
            rows = db.execute(
                "SELECT history_id, orig, patch FROM patch_fail"
            ).fetchall()

            # We should have 2 failed patches
            self.assertEqual(len(rows), 2)

            # Both patches should be linked to the same history_id
            patches_with_id = [r for r in rows if r[0] == call_history_id]
            self.assertEqual(len(patches_with_id), 2)

            # Verify both patches are present
            patches_text = [r[2] for r in patches_with_id]
            self.assertTrue(any("wrong content" in p for p in patches_text))
            self.assertTrue(any("also wrong" in p for p in patches_text))


if __name__ == "__main__":
    unittest.main()
