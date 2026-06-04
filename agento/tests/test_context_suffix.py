from agento.context import ContextMode, context_handler
from agento.context.suffix import SuffixHandler
from agento import context
from agento.tests.test_context import TestContextBase


class TestSuffix(TestContextBase):
    def setUp(self):
        super().setUp()
        context.set_context_mode(ContextMode.SUFFIX, reset_ctx_id=True)

    def assert_foo(self, res: str, ctx_id: int):
        self.assertIn("\n>>> === CONTENT START ===", res)
        self.assertIn("foo\ntext", res)
        self.assertIn(f"CTX({ctx_id})", res)

    def assert_ctx_out_of_date(self, msgs: list[dict], ctx_id: int):
        all_expected = [f"ID: CTX({ctx_id})", "out of date"]
        for msg in msgs:
            if msg.get("role") != "tool":
                continue
            content = str(msg.get("content", ""))
            if all(x in content for x in all_expected):
                return
        raise ValueError(f"CTX({ctx_id}) not found out of date")

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

        self.assert_ctx_out_of_date(msgs, 0)
        self.assert_ctx_out_of_date(msgs, 1)
        self.assert_ctx_has_full_content(msgs, 2, "foo\ntext")

    def test_write_three_times(self):
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please write foo three times"))

        self.tool_call_write_with_check(self.FILE_FOO, 0, "FIRST_CONTENT")
        self.tool_call_write_with_check(self.FILE_FOO, 1, "SECOND_CONTENT")
        self.tool_call_write_with_check(self.FILE_FOO, 2, "THIRD_CONTENT")
        self.epilogue()

        self.assert_ctx_out_of_date(msgs, 0)
        self.assert_ctx_out_of_date(msgs, 1)
        self.assert_ctx_has_full_content(msgs, 2, "THIRD_CONTENT")

    def test_write_then_read(self):
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please write then read foo"))
        self.tool_call_write_with_check(self.FILE_FOO, 0, "WRITTEN_CONTENT")
        self.tool_call_read_with_check(self.FILE_FOO, 1, "WRITTEN_CONTENT")
        self.epilogue()

        self.assert_ctx_out_of_date(msgs, 0)
        self.assert_ctx_has_full_content(msgs, 1, "WRITTEN_CONTENT")

    def test_read_then_write(self):
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please read then write foo"))

        self.tool_call_read_with_check(self.FILE_FOO, 0)
        self.tool_call_write_with_check(self.FILE_FOO, 1, "NEW_WRITTEN_CONTENT")
        self.epilogue()

        self.assert_ctx_out_of_date(msgs, 0)
        self.assert_ctx_has_full_content(msgs, 1, "NEW_WRITTEN_CONTENT")

    def test_parallel_io(self):
        """Test that writing to two files maintains independent contexts.

        Changing one file should not affect the other file's context.
        """
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please write both foo and bar files"))

        self.tool_call_write_with_check(self.FILE_FOO, 0, "FOO_ROUND1")
        self.tool_call_write_with_check(self.FILE_BAR, 1, "BAR_ROUND1")
        self.tool_call_write_with_check(self.FILE_FOO, 2, "FOO_ROUND2")
        self.tool_call_write_with_check(self.FILE_BAR, 3, "BAR_ROUND2")

        self.epilogue()

        self.assert_ctx_out_of_date(msgs, 0)
        self.assert_ctx_out_of_date(msgs, 1)
        self.assert_ctx_has_full_content(msgs, 2, "FOO_ROUND2")
        self.assert_ctx_has_full_content(msgs, 3, "BAR_ROUND2")

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

        self.assert_ctx_out_of_date(msgs, 0)
        self.assert_ctx_out_of_date(msgs, 1)
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
        self.assert_ctx_out_of_date(msgs, 0)
        # Verify: CTX(1) should have the edited content
        self.assert_ctx_has_full_content(msgs, 1, "dummy\ncontent")
        # Verify: original "foo\ntext" should NOT appear in CTX(0)'s full content block
        # (only in the reference)
        for msg in msgs:
            content = str(msg.get("content", ""))
            if "ID: CTX(0)" in content and "\n>>> === CONTENT START ===" in content:
                # CTX(0) should reference CTX(1), not contain full "foo\ntext" content
                self.assertIn("CONTENT IS OUT OF DATE", content)
                # Check that the original content "foo\ntext" is not in the message
                # (the file path may contain "foo" as substring, but the content shouldn't)
                self.assertNotIn("foo\ntext", content)

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
        self.assert_ctx_out_of_date(msgs, 0)
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
        content_end_exist = any(
            line.startswith(">>> === CONTENT END") for line in lines
        )
        self.assertIsNotNone(content_end_exist, "CONTENT END marker should exist")
        self.epilogue()
        self.assert_ctx_out_of_date(msgs, 0)
        self.assert_ctx_has_full_content(msgs, 1, edited_text)

    def test_read_safe_from_asst(self):
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please read foo"))
        msgs.append(dummy_llm.msg_assistant("Reading foo"))
        self.tool_call_read_with_check(self.FILE_FOO, 0, "foo\ntext")
        assistant_idx = len(msgs)
        msgs.append(dummy_llm.msg_assistant(msgs[-1]["content"]))
        assistant_msg = msgs[assistant_idx]["content"]
        msgs.append(dummy_llm.msg_assistant("Reading foo again"))
        self.tool_call_read_with_check(self.FILE_FOO, 1, "foo\ntext")
        self.assertEqual(assistant_msg, msgs[assistant_idx]["content"])
        self.epilogue()
        self.assert_ctx_out_of_date(msgs, 0)
        self.assert_ctx_has_full_content(msgs, 1, "foo\ntext")

        self.assertEqual(assistant_msg, msgs[assistant_idx]["content"])

    def test_file_contains_context_like_content(self):
        # TODO: simplify
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please read file with context-like content"))

        # Step 1: Write a file with content that looks like context output
        # Using raw Path.write_text to bypass the tool and write exact content
        context_like_content = (
            ">>> ID: 0 OPERATION: read_file CTX-IO-FILE: .agento.demo.foo\n"
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
            ">>> ID: CTX(0) OPERATION: read_file CTX-IO-FILE: .agento.demo.foo\n"
            ">>> === CONTENT START ===\n"
            ">>> ID: 0 OPERATION: read_file CTX-IO-FILE: .agento.demo.foo\n"
            ">>> === CONTENT START ===\n"
            "THIS FILE HAS 4 LINES\n"
            ">>> === CONTENT END ===\n"
            ">>> === CONTENT END ==="
        )

        self.assertEqual(res, expected_result)
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
        # Set custom prefix
        ctx = context_handler()
        assert isinstance(ctx, SuffixHandler)
        ctx.prefix = "~~~~~"

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Test custom prefix with context-like content"))

        # Step 1: Write a file with content that looks like context output (using '>>>')
        context_like_content = (
            ">>> ID: 0 OPERATION: read_file CTX-IO-FILE: .agento.demo.foo\n"
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
        self.assert_ctx_out_of_date(msgs, 0)
        self.assert_ctx_out_of_date(msgs, 1)
        self.assert_ctx_out_of_date(msgs, 2)

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
            for line in content.splitlines():
                # If line starts with '>>>' and contains ID or CONTENT, it's a marker
                our_line = line.startswith(">>> ID:") or line.startswith(
                    ">>> === CONTENT"
                )
                if not our_line or "CTX(0)" in content:
                    continue
                else:
                    self.fail(f"Found '>>>' marker outside CTX(0) file content: {line}")
