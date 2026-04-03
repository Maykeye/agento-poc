from context import ContextMode, context_handler
from context.suffix import SuffixHandler
import context
from tests.test_context import TestContextBase


class TestSuffix(TestContextBase):
    def setUp(self):
        super().setUp()
        context.set_context_mode(ContextMode.SUFFIX, reset_ctx_id=True)

    def assert_foo(self, res: str, ctx_id: int):
        self.assertIn("\n>>> === CONTENT START ===", res)
        self.assertIn("foo\ntext", res)
        self.assertIn(f"CTX({ctx_id})", res)

    def assert_ctx_references(self, msgs: list[dict], from_ctx_id: int, to_ctx_id: int):
        found = False
        all_expected = [
            f"ID: CTX({from_ctx_id})",
            f"CURRENT CONTENT IN CTX({to_ctx_id})",
        ]
        for msg in msgs:
            if msg.get("role") != "tool":
                continue
            content = str(msg.get("content", ""))
            if all(x in content for x in all_expected):
                found = True
                break
        self.assertTrue(found, f"CTX({from_ctx_id}) should reference CTX({to_ctx_id})")

    def assert_ctx_has_full_content(
        self, msgs: list[dict], ctx_id: int, expected_content: str
    ):
        found = False
        all_expected = [f"ID: CTX({ctx_id})", "\n>>> === CONTENT START ==="]
        all_expected += [expected_content]
        for msg in msgs:
            if msg.get("role") != "tool":
                continue
            content = str(msg.get("content", ""))
            if all(x in content for x in all_expected):
                found = True
                break
        self.assertTrue(
            found, f"CTX({ctx_id}) should have full content with '{expected_content}'"
        )

    def test_read_three_times(self):
        """Test reading file three times creates proper chain of references."""
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please read foo three times"))

        self.tool_call_read_with_check(self.FILE_FOO, 0)
        self.tool_call_read_with_check(self.FILE_FOO, 1)
        self.tool_call_read_with_check(self.FILE_FOO, 2)
        self.epilogue()

        self.assert_ctx_references(msgs, 0, 2)
        self.assert_ctx_references(msgs, 1, 2)
        self.assert_ctx_has_full_content(msgs, 2, "foo\ntext")

    def test_write_three_times(self):
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please write foo three times"))

        self.tool_call_write_with_check(self.FILE_FOO, 0, "FIRST_CONTENT")
        self.tool_call_write_with_check(self.FILE_FOO, 1, "SECOND_CONTENT")
        self.tool_call_write_with_check(self.FILE_FOO, 2, "THIRD_CONTENT")
        self.epilogue()

        self.assert_ctx_references(msgs, 0, 2)
        self.assert_ctx_references(msgs, 1, 2)
        self.assert_ctx_has_full_content(msgs, 2, "THIRD_CONTENT")

    def test_write_then_read(self):
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please write then read foo"))
        self.tool_call_write_with_check(self.FILE_FOO, 0, "WRITTEN_CONTENT")
        self.tool_call_read_with_check(self.FILE_FOO, 1, "WRITTEN_CONTENT")
        self.epilogue()

        self.assert_ctx_references(msgs, 0, 1)
        self.assert_ctx_has_full_content(msgs, 1, "WRITTEN_CONTENT")

    def test_read_then_write(self):
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please read then write foo"))

        self.tool_call_read_with_check(self.FILE_FOO, 0)
        self.tool_call_write_with_check(self.FILE_FOO, 1, "NEW_WRITTEN_CONTENT")
        self.epilogue()

        self.assert_ctx_references(msgs, 0, 1)
        self.assert_ctx_has_full_content(msgs, 1, "NEW_WRITTEN_CONTENT")

    def test_parallel_io(self):
        """Test that writing to two files maintains independent contexts.

        Changing one file should not affect the other file's context.
        """
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please write both foo and bar files"))

        # First round: Write both files with new content
        # CTX(0) = FILE_FOO with FOO_ROUND1
        # CTX(1) = FILE_BAR with BAR_ROUND1
        self.tool_call_write_with_check(self.FILE_FOO, 0, "FOO_ROUND1")
        self.tool_call_write_with_check(self.FILE_BAR, 1, "BAR_ROUND1")

        # Second round: Write both files with different content
        # CTX(2) = FILE_FOO with FOO_ROUND2
        # CTX(3) = FILE_BAR with BAR_ROUND2
        self.tool_call_write_with_check(self.FILE_FOO, 2, "FOO_ROUND2")
        self.tool_call_write_with_check(self.FILE_BAR, 3, "BAR_ROUND2")

        self.epilogue()

        # After epilogue, verify each file's contexts are independent
        # FILE_FOO: CTX(0) references CTX(2), CTX(2) has FOO_ROUND2 content
        # FILE_BAR: CTX(1) references CTX(3), CTX(3) has BAR_ROUND2 content

        # Verify FILE_FOO contexts: CTX(0) references CTX(2), and CTX(2) has full content
        self.assert_ctx_references(msgs, 0, 2)  # FOO: 0 references 2
        self.assert_ctx_has_full_content(msgs, 2, "FOO_ROUND2")

        # Verify FILE_BAR contexts: CTX(1) references CTX(3), and CTX(3) has full content
        self.assert_ctx_references(msgs, 1, 3)  # BAR: 1 references 3
        self.assert_ctx_has_full_content(msgs, 3, "BAR_ROUND2")

        # CRITICAL: Verify that FOO's content doesn't appear in BAR's contexts and vice versa
        # CTX(2) should have FOO_ROUND2 but NOT BAR_ROUND2 or BAR_ROUND1
        foo_has_no_bar_content = True
        for msg in msgs:
            content = str(msg.get("content", ""))
            if "ID: CTX(2)" in content and (
                "BAR_ROUND1" in content or "BAR_ROUND2" in content
            ):
                foo_has_no_bar_content = False
                break
        self.assertTrue(
            foo_has_no_bar_content, "CTX(2) for FOO should not contain BAR content"
        )

        # CTX(3) should have BAR_ROUND2 but NOT FOO_ROUND1 or FOO_ROUND2
        bar_has_no_foo_content = True
        for msg in msgs:
            content = str(msg.get("content", ""))
            if "ID: CTX(3)" in content and (
                "FOO_ROUND1" in content or "FOO_ROUND2" in content
            ):
                bar_has_no_foo_content = False
                break
        self.assertTrue(
            bar_has_no_foo_content, "CTX(3) for BAR should not contain FOO content"
        )

    def test_delete_file(self):
        """Test deleting a file creates a context entry with (file deleted) content."""
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please delete foo"))

        res = self.tool_call_delete_foo()
        self.assertIn("\n>>> === CONTENT START ===", res)
        self.assertIn("(file deleted)", res)
        self.assertIn("CTX(0)", res)
        self.epilogue()

        # Verify the deleted content is in the message
        self.assert_ctx_has_full_content(msgs, 0, "(file deleted)")

    def test_delete_then_write_then_read(self):
        """Test delete -> write -> read chain creates proper references."""
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please delete, write, and read foo"))

        # Delete: CTX(0) = (file deleted)
        res0 = self.tool_call_delete_foo()
        self.assertIn("CTX(0)", res0)
        self.assertIn("(file deleted)", res0)

        # Write: CTX(1) = NEW_CONTENT
        self.tool_call_write_with_check(self.FILE_FOO, 1, "NEW_CONTENT")

        # Read: CTX(2) = NEW_CONTENT (same as written)
        self.tool_call_read_with_check(self.FILE_FOO, 2, "NEW_CONTENT")

        self.epilogue()

        # CTX(0) and CTX(1) should reference CTX(2)
        self.assert_ctx_references(msgs, 0, 2)
        self.assert_ctx_references(msgs, 1, 2)
        # CTX(2) should have the final content
        self.assert_ctx_has_full_content(msgs, 2, "NEW_CONTENT")

    def test_edit_sample(self):
        """Test editing a file: read, then edit all content, verify old content is gone."""
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please read and then edit foo"))

        # Read foo: CTX(0) = foo\ntext
        self.tool_call_read_with_check(self.FILE_FOO, 0, "foo\ntext")

        # Edit: replace all content "foo\ntext" with "dummy\ncontent"
        # CTX(1) = dummy\ncontent
        res = self.tool_call_edit_foo("foo\ntext", "dummy\ncontent")
        self.assertIn("CTX(1)", res)
        self.assertIn("dummy\ncontent", res)

        self.epilogue()

        # Verify: original reading (CTX(0)) should now reference CTX(1)
        self.assert_ctx_references(msgs, 0, 1)
        # Verify: CTX(1) should have the edited content
        self.assert_ctx_has_full_content(msgs, 1, "dummy\ncontent")
        # Verify: original "foo\ntext" should NOT appear in CTX(0)'s full content block
        # (only in the reference)
        for msg in msgs:
            content = str(msg.get("content", ""))
            if "ID: CTX(0)" in content and "\n>>> === CONTENT START ===" in content:
                # CTX(0) should reference CTX(1), not contain full "foo\ntext" content
                self.assertIn("CURRENT CONTENT IN CTX(1)", content)
                self.assertNotIn("foo", content)
                self.assertNotIn("text", content)

    def test_edit_one_line_only(self):
        """Test editing only one line of a file, keeping rest of content intact."""
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please read and edit one line of foo"))

        # Read foo: CTX(0) = foo\ntext
        self.tool_call_read_with_check(self.FILE_FOO, 0, "foo\ntext")

        # Edit: replace only "foo" with "FOO" (not including "\ntext")
        # CTX(1) = FOO\ntext (only one line changed)
        res = self.tool_call_edit_foo("foo", "FOO")
        self.assertIn("CTX(1)", res)
        self.assertIn("FOO", res)
        # The edited content should still have "text" after FOO
        self.assertIn("text", res)

        self.epilogue()

        # Verify: original reading (CTX(0)) should now reference CTX(1)
        self.assert_ctx_references(msgs, 0, 1)
        # Verify: CTX(1) should have the edited content "FOO\ntext"
        self.assert_ctx_has_full_content(msgs, 1, "FOO\ntext")
        # Verify: the full content message for CTX(1) shows "FOO\ntext"
        for msg in msgs:
            content = str(msg.get("content", ""))
            # Check that CTX(1) has both FOO and text in its content block
            if "ID: CTX(1)" not in content:
                continue
            # Find the content block for CTX(1)
            pfx = "\n>>> CONTENT START ==="
            if pfx in content and "=== CONTENT END ===" in content:
                start = content.find(pfx) + len(pfx)
                end = content.find("\n>>> === CONTENT END ===")
                full_content = content[start:end].rstrip()
                lines = full_content.split("\n")
                full_content = "\n".join(lines).strip()
                self.assertEqual(full_content, "FOO\ntext")

    def test_write_no_nl(self):
        """Test write_file with content that has no trailing newline.

        When writing content without a trailing newline, the '>>>' markers
        should still start on their own lines in the response.
        """
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please write foo without newline"))

        text_no_nl = "NO_NEWLINE_AT_END"
        res = self.tool_call_write_with_check(self.FILE_FOO, 0, text_no_nl)

        # Verify that the '>>>' marker starts on a new line after content
        # The response should have proper structure with markers on separate lines
        self.assertIn(">>> ID: CTX(0)", res)
        self.assertIn("\n>>> === CONTENT START ===", res)
        self.assertIn(text_no_nl, res)
        self.assertIn("\n>>> === CONTENT END ===", res)

        # Critical: verify CONTENT END marker is on its own line (not concatenated)
        # The pattern should be: content\n>>> === CONTENT END ===
        lines = res.split("\n")
        content_end_idx = None
        for i, line in enumerate(lines):
            if ">>> === CONTENT END ===" in line:
                content_end_idx = i
                break
        self.assertIsNotNone(content_end_idx, "CONTENT END marker should exist")
        # The marker should be on its own line
        self.assertTrue(lines[content_end_idx].strip() == ">>> === CONTENT END ===")
        self.epilogue()

        # After epilogue, verify the content is properly stored
        self.assert_ctx_has_full_content(msgs, 0, text_no_nl)

    def test_edit_no_nl(self):
        """Test edit_file with content that has no trailing newline.

        When editing content without a trailing newline, the response should
        have '>>>' markers starting on their own lines, and the edited content
        should be properly formatted.
        """
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please edit foo without newline"))

        # First write content without trailing newline
        msgs.append(dummy_llm.msg_assistant("Writing foo without newline"))
        text_no_nl = "NO_NEWLINE"
        self.tool_call_write_with_check(self.FILE_FOO, 0, text_no_nl)

        # Now edit that content
        msgs.append(dummy_llm.msg_assistant("Editing foo without newline"))
        edited_text = "EDITED_NO_NEWLINE"
        res = self.tool_call_edit_foo("NO_NEWLINE", edited_text)

        # Verify response has proper structure with markers on separate lines
        self.assertIn(">>> ID: CTX(1)", res)
        self.assertIn("\n>>> === CONTENT START ===", res)
        self.assertIn(edited_text, res)
        self.assertIn("\n>>> === CONTENT END ===", res)

        # Critical: verify CONTENT END marker is on its own line
        lines = res.split("\n")
        content_end_idx = None
        for i, line in enumerate(lines):
            if ">>> === CONTENT END ===" in line:
                content_end_idx = i
                break
        self.assertIsNotNone(content_end_idx, "CONTENT END marker should exist")
        self.assertTrue(lines[content_end_idx].strip() == ">>> === CONTENT END ===")

        self.epilogue()

        # After epilogue, verify the edited content is properly stored
        self.assert_ctx_has_full_content(msgs, 1, edited_text)
        # And the original write (CTX(0)) references CTX(1)
        self.assert_ctx_references(msgs, 0, 1)

    def test_read_safe_from_asst(self):
        """Test that reading file is safe even when assistant message contains file content.

        This test verifies the context handling when an assistant message contains
        a replica of the tool result (file content). The system should correctly
        identify which context IDs belong to which operations.

        Flow:
        1. First read operation on file (creates CTX(0))
        2. Insert exact replica from tool result into assistant message content
        3. Execute tool call to read file again (creates CTX(1))
        4. Call epilogue to make sure handling is done

        Expected behavior:
        - Initial read message (CTX(0)) must redirect to second read message (CTX(1))
        - Assistant message should reference CTX(0) (as it was created after first read)
        - Second read message should be CTX(1) and not see the assistant message content
        """
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please read foo"))

        # Step 1: First read operation on file (creates CTX(0))
        msgs.append(dummy_llm.msg_assistant("Reading foo"))
        self.tool_call_read_with_check(self.FILE_FOO, 0, "foo\ntext")

        # Step 2: Insert exact replica from tool result into assistant message content
        # This simulates the case where assistant logs the file content in its message
        assistant_idx = len(msgs)
        msgs.append(dummy_llm.msg_assistant(msgs[-1]["content"]))
        assistant_msg = msgs[assistant_idx]["content"]

        # Step 3: Execute tool call to read file again (creates CTX(1))
        msgs.append(dummy_llm.msg_assistant("Reading foo again"))
        self.tool_call_read_with_check(self.FILE_FOO, 1, "foo\ntext")

        # Assistant message should not be changed
        # (Not expected to fail even if code fails: epilogue wasn't called)
        self.assertEqual(assistant_msg, msgs[assistant_idx]["content"])

        # Step 4: Call epilogue to make sure handling is done
        self.epilogue()

        # Test: Reference movement
        self.assert_ctx_references(msgs, 0, 1)
        self.assert_ctx_has_full_content(msgs, 1, "foo\ntext")

        # Assistant message should not be changed
        self.assertEqual(assistant_msg, msgs[assistant_idx]["content"])

    def test_file_contains_context_like_content(self):
        # TODO: simplify
        """Test that a file containing context-like content is properly handled.

        This test verifies that when a file contains content that looks like
        the context output format (with ID, OPERATION, CONTENT START/END markers),
        the read_file tool properly wraps it with its own context markers.

        The file content itself contains:
        - A header line with ID, OPERATION, and file path
        - CONTENT START marker
        - Actual content
        - CONTENT END marker

        When read through the tool, this should result in nested context markers:
        - Tool adds its own header and CONTENT START
        - File content (which includes its own header, CONTENT START, content, CONTENT END)
        - Tool adds its own CONTENT END

        This tests that the SuffixHandler doesn't get confused by context-like
        content inside a file.
        """
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please read file with context-like content"))

        # Step 1: Write a file with content that looks like context output
        # Using raw Path.write_text to bypass the tool and write exact content
        context_like_content = (
            ">>> ID: 0 OPERATION: read_file CTX-IO-FILE:  .agento.demo.foo\n"
            ">>> === CONTENT START ===\n"
            "THIS FILE HAS 4 LINES\n"
            ">>> === CONTENT END ==="
        )
        self.FILE_FOO.write_text(context_like_content)

        # Step 2: Read the file using the tool
        msgs.append(dummy_llm.msg_assistant("Reading file with context-like content"))
        res = self.tool_call_read(self.FILE_FOO)

        # Step 3: Verify the result contains properly nested context markers
        # The tool should add its own header and markers around the file content
        expected_result = (
            ">>> ID: CTX(0) OPERATION: read_file CTX-IO-FILE:  .agento.demo.foo\n"
            ">>> === CONTENT START ===\n"
            ">>> ID: 0 OPERATION: read_file CTX-IO-FILE:  .agento.demo.foo\n"
            ">>> === CONTENT START ===\n"
            "THIS FILE HAS 4 LINES\n"
            ">>> === CONTENT END ===\n"
            ">>> === CONTENT END ==="
        )

        self.assertEqual(res, expected_result)

        # Verify that the result has the correct structure:
        # - First two lines are the tool's header (added by context)
        # - Next 4 lines are from the file (which looks like context)
        # - Last line is the tool's CONTENT END marker

        lines = res.split("\n")
        self.assertEqual(len(lines), 7)  # 7 lines total

        # Line 0: Tool header with CTX(0)
        self.assertTrue(lines[0].startswith(">>> ID: CTX(0)"))
        self.assertIn("OPERATION: read_file", lines[0])

        # Line 1: Tool's CONTENT START marker
        self.assertEqual(lines[1], ">>> === CONTENT START ===")

        # Lines 2-5: File content (which includes its own context-like structure)
        self.assertTrue(lines[2].startswith(">>> ID: 0"))
        self.assertEqual(lines[3], ">>> === CONTENT START ===")
        self.assertEqual(lines[4], "THIS FILE HAS 4 LINES")
        self.assertEqual(lines[5], ">>> === CONTENT END ===")

        # Line 6: Tool's CONTENT END marker
        self.assertEqual(lines[6], ">>> === CONTENT END ===")

        # Call epilogue to verify the context is properly stored
        self.epilogue()

        # Verify that the full content is stored in CTX(0)
        self.assert_ctx_has_full_content(msgs, 0, context_like_content)

    def test_custom_prefix_with_context_like_content(self):
        """Test that custom prefix works correctly when file contains context-like content.

        This test verifies that when a custom prefix (e.g., '~~~~~') is set,
        the SuffixHandler properly distinguishes between:
        1. Context markers added by the tool (using custom prefix)
        2. Content inside files that happens to look like context (using default '>>>')

        The file content contains the default '>>>' prefix, but tool responses
        should use the custom '~~~~~' prefix.

        Flow:
        1. Set custom prefix to '~~~~~'
        2. Write file with '>>>' context-like content using Path.write_text
        3. Read the file - tool response should use '~~~~~' prefix
        4. Write new content - should use '~~~~~' prefix
        5. Edit content - should use '~~~~~' prefix
        6. Delete file - should use '~~~~~' prefix
        7. Verify epilogue properly handles all operations with custom prefix
        """
        # Set custom prefix
        ctx = context_handler()
        assert isinstance(ctx, SuffixHandler)
        ctx.prefix = "~~~~~"

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Test custom prefix with context-like content"))

        # Step 1: Write a file with content that looks like context output (using '>>>')
        context_like_content = (
            ">>> ID: 0 OPERATION: read_file CTX-IO-FILE:  .agento.demo.foo\n"
            ">>> === CONTENT START ===\n"
            "THIS FILE HAS 4 LINES\n"
            ">>> === CONTENT END ==="
        )
        self.FILE_FOO.write_text(context_like_content)

        # Step 2: Read the file - response should use custom '~~~~~' prefix
        msgs.append(dummy_llm.msg_assistant("Reading file with context-like content"))
        res_read = self.tool_call_read(self.FILE_FOO)

        # Verify read response uses custom prefix
        self.assertIn("~~~~~ ID: CTX(0) OPERATION: read_file", res_read)
        self.assertIn("~~~~~ === CONTENT START ===", res_read)
        self.assertIn("~~~~~ === CONTENT END ===", res_read)
        # Verify file content (with '>>>') is preserved inside
        self.assertIn(">>> ID: 0 OPERATION: read_file", res_read)
        self.assertIn(">>> === CONTENT START ===", res_read)
        self.assertIn("THIS FILE HAS 4 LINES", res_read)
        self.assertIn(">>> === CONTENT END ===", res_read)

        # Step 3: Write new content - should use custom prefix
        msgs.append(dummy_llm.msg_assistant("Writing new content"))
        res_write = self.tool_call_write(self.FILE_FOO, "NEW_CONTENT_HERE")

        self.assertIn("~~~~~ ID: CTX(1) OPERATION: write_file", res_write)
        self.assertIn("~~~~~ OK: write_file", res_write)
        self.assertIn("~~~~~ === CONTENT START ===", res_write)
        self.assertIn("NEW_CONTENT_HERE", res_write)
        self.assertIn("~~~~~ === CONTENT END ===", res_write)

        # Step 4: Edit content - should use custom prefix
        msgs.append(dummy_llm.msg_assistant("Editing content"))
        res_edit = self.tool_call_edit_foo("NEW_CONTENT_HERE", "EDITED_CONTENT")

        self.assertIn("~~~~~ ID: CTX(2) OPERATION: edit_file", res_edit)
        self.assertIn("~~~~~ OK: edit", res_edit)
        self.assertIn("~~~~~ === CONTENT START ===", res_edit)
        self.assertIn("EDITED_CONTENT", res_edit)
        self.assertIn("~~~~~ === CONTENT END ===", res_edit)

        # Step 5: Delete file - should use custom prefix
        msgs.append(dummy_llm.msg_assistant("Deleting file"))
        res_delete = self.tool_call_delete_foo()

        self.assertIn("~~~~~ ID: CTX(3) OPERATION: delete_file", res_delete)
        self.assertIn("~~~~~ OK: delete_file", res_delete)
        self.assertIn("~~~~~ === CONTENT START ===", res_delete)
        self.assertIn("(file deleted)", res_delete)
        self.assertIn("~~~~~ === CONTENT END ===", res_delete)

        # Step 6: Call epilogue and verify references are updated correctly
        self.epilogue()

        # Verify all old context IDs reference the latest one (CTX(3))
        self.assert_ctx_references(msgs, 0, 3)  # Read references delete
        self.assert_ctx_references(msgs, 1, 3)  # Write references delete
        self.assert_ctx_references(msgs, 2, 3)  # Edit references delete

        # Verify CTX(3) has full content with custom prefix
        for msg in msgs:
            if msg.get("role") != "tool":
                continue
            content = str(msg.get("content", ""))
            if "ID: CTX(3)" in content and "(file deleted)" in content:
                # CTX(3) should use custom prefix
                self.assertIn("~~~~~ ID: CTX(3)", content)
                self.assertIn("~~~~~ === CONTENT START ===", content)
                self.assertIn("~~~~~ === CONTENT END ===", content)
                break
        else:
            self.fail("CTX(3) with delete content not found")

        # Verify no '>>>' prefix appears in tool-added markers (only in file content)
        # The '>>>' should only appear in the original read content (CTX(0)'s file content)
        for msg in msgs:
            if msg.get("role") != "tool":
                continue
            content = str(msg.get("content", ""))
            # Tool markers should use '~~~~~', not '>>>'
            # '>>>' can only appear as part of the original file content in CTX(0)
            lines = content.split("\n")
            for line in lines:
                # If line starts with '>>>' and contains ID or CONTENT, it's a marker
                if line.startswith(">>> ID:") or line.startswith(">>> === CONTENT"):
                    # This should only be valid for CTX(0)'s file content
                    # Check if this is inside CTX(0)'s content block
                    if "CTX(0)" in content:
                        # This is expected - it's the file content
                        pass
                    else:
                        self.fail(
                            f"Found '>>>' marker outside CTX(0) file content: {line}"
                        )
