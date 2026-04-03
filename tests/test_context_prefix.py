from tests.test_context import TestContextBase
import context
from context import ContextMode, context_handler


class TestPrefix(TestContextBase):
    def setUp(self):
        super().setUp()
        context.set_context_mode(ContextMode.PREFIX, reset_ctx_id=True)

    def foo_content(self, oper, ctx_id):
        return f"=== Path: .agento.demo.foo (last id: CTX({ctx_id}); last operation: {oper}) CONTEXT START ===\nfoo\ntext\n=== Path: .agento.demo.foo (last id: CTX({ctx_id}); last operation: {oper})"

    def ctx_change(self, oper, id):
        return {
            "context-change": {
                "path": ".agento.demo.foo",
                "operation": f"{oper}",
                "id": f"CTX({id})",
            }
        }

    def test_read_twice(self):
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please read foo twice"))
        context_handler().prepare_current_llm(dummy_llm)

        msgs.append(dummy_llm.msg_assistant("Calling reading foo"))
        res0 = self.tool_call_read(self.FILE_FOO)
        self.assertEqual(res0, self.ctx_change("read_file", 0))
        self.epilogue()
        self.assertIn(self.foo_content("read_file", 0), msgs[0]["content"])

        msgs.append(dummy_llm.msg_assistant("Calling reading foo"))
        res1 = self.tool_call_read(self.FILE_FOO)
        self.assertEqual(res1, self.ctx_change("read_file", 1))
        self.epilogue()
        self.assertIn(self.foo_content("read_file", 1), msgs[0]["content"])
        self.assertNotIn(self.foo_content("read_file", 0), msgs[0]["content"])

    def test_write_file_twice(self):
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please write foo twice"))

        # TODO: Make it shorter
        msgs.append(dummy_llm.msg_assistant("Calling writing foo"))
        res = self.tool_call_write(self.FILE_FOO, "NEWFOO")
        assert res == self.ctx_change("write_file", 0)
        self.epilogue()
        assert self.foo_content("write_file", 0) not in msgs[0]["content"]
        new_ctx = self.foo_content("write_file", 0).replace("foo\ntext", "NEWFOO")
        self.assertIn(new_ctx, msgs[0]["content"])

        msgs.append(dummy_llm.msg_assistant("Calling writing foo"))
        res = self.tool_call_write(self.FILE_FOO, "FOONEW")
        assert res == self.ctx_change("write_file", 1)
        self.epilogue()
        assert self.foo_content("write_file", 0) not in msgs[0]["content"]
        new_ctx = self.foo_content("write_file", 1).replace("foo\ntext", "FOONEW")
        self.assertIn(new_ctx, msgs[0]["content"])

    def test_delete_file(self):
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please delete foo"))

        msgs.append(dummy_llm.msg_assistant("Calling deleting foo"))
        res = self.tool_call_delete_foo()
        assert res == self.ctx_change("delete_file", 0)
        self.epilogue()
        deleted = self.foo_content("delete_file", 0).replace(
            "foo\ntext", "(file deleted)"
        )
        self.assertIn(deleted, msgs[0]["content"])

    def test_edit_twice(self):
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please edit foo twice"))

        # First edit: change "foo" to "FOO"
        msgs.append(dummy_llm.msg_assistant("Calling first edit"))
        res = self.tool_call_edit_foo("foo", "FOO")
        assert res == self.ctx_change("edit_file", 0)
        self.epilogue()

        # After first epilogue, foo must be replaced with FOO, BARR should not exist
        foo_content_0 = (
            f"=== Path: .agento.demo.foo (last id: CTX(0); last operation: edit_file) CONTEXT START ===\n"
            "FOO\n"
            "text\n"
            f"=== Path: .agento.demo.foo (last id: CTX(0); last operation: edit_file) CONTEXT END ==="
        )
        self.assertIn(foo_content_0, msgs[0]["content"])
        self.assertNotIn("BARR", msgs[0]["content"])

        # Second edit: change "FOO" to "BARR"
        msgs.append(dummy_llm.msg_assistant("Calling second edit"))
        res = self.tool_call_edit_foo("FOO", "BARR")
        assert res == self.ctx_change("edit_file", 1)
        self.epilogue()

        # After second epilogue, FOO must be replaced with BARR, FOO must not exist
        foo_content_1 = (
            f"=== Path: .agento.demo.foo (last id: CTX(1); last operation: edit_file) CONTEXT START ===\n"
            "BARR\n"
            "text\n"
            f"=== Path: .agento.demo.foo (last id: CTX(1); last operation: edit_file) CONTEXT END ==="
        )
        self.assertIn(foo_content_1, msgs[0]["content"])
        self.assertNotIn("FOO", msgs[0]["content"])

    def test_write_no_nl(self):
        """Test write_file with content that has no trailing newline.

        Prefix handler stores content and formats it with === markers.
        Content without newline should be properly stored and displayed.
        """
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please write foo without newline"))

        msgs.append(dummy_llm.msg_assistant("Calling writing foo without newline"))
        text_no_nl = "NO_NEWLINE_AT_END"
        res = self.tool_call_write(self.FILE_FOO, text_no_nl)
        assert res == self.ctx_change("write_file", 0)
        self.epilogue()

        # Verify content without newline is stored correctly
        new_ctx = f"=== Path: .agento.demo.foo (last id: CTX(0); last operation: write_file) CONTEXT START ===\n"
        new_ctx += "NO_NEWLINE_AT_END\n"
        new_ctx += f"=== Path: .agento.demo.foo (last id: CTX(0); last operation: write_file) CONTEXT END ==="
        self.assertIn(new_ctx, msgs[0]["content"])

    def test_edit_no_nl(self):
        """Test edit_file with content that has no trailing newline.

        When editing content without newline, the context should be updated
        properly with the new content.
        """
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please edit foo without newline"))

        # First write content without trailing newline
        msgs.append(dummy_llm.msg_assistant("Writing foo without newline"))
        text_no_nl = "NO_NEWLINE"
        self.tool_call_write(self.FILE_FOO, text_no_nl)
        self.epilogue()

        # Now edit that content
        msgs.append(dummy_llm.msg_assistant("Editing foo without newline"))
        res = self.tool_call_edit_foo("NO_NEWLINE", "EDITED_NO_NEWLINE")
        assert res == self.ctx_change("edit_file", 1)
        self.epilogue()

        # Verify the edited content is in context
        edited_ctx = (
            f"=== Path: .agento.demo.foo (last id: CTX(1); last operation: edit_file) CONTEXT START ===\n"
            "EDITED_NO_NEWLINE\n"
            f"=== Path: .agento.demo.foo (last id: CTX(1); last operation: edit_file) CONTEXT END ==="
        )
        self.assertIn(edited_ctx, msgs[0]["content"])
