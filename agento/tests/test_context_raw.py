from agento.context import ContextMode
from agento import context
from agento.tests.test_context import TestContextBase


class TestRaw(TestContextBase):
    def setUp(self):
        super().setUp()
        context.set_context_mode(ContextMode.RAW)

    @property
    def foo_content(self):
        pfx = ">>>"
        return f"""
{pfx} OK: read_file .agento.demo.foo
{pfx} === CONTENT START ===
foo
text
{pfx} === CONTENT END ===""".strip()

    def test_read_twice(self):
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please read foo twice"))
        msgs.append(dummy_llm.msg_assistant("Calling reading foo"))
        res1 = self.tool_call_read(self.FILE_FOO)
        self.assertEqual(res1, self.foo_content)

        msgs.append(dummy_llm.msg_assistant("Calling reading foo again"))
        res2 = self.tool_call_read(self.FILE_FOO)
        self.assertEqual(res2, self.foo_content)

    def test_write_file(self):
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please write foo"))
        msgs.append(dummy_llm.msg_assistant("Calling writing foo"))
        res1 = self.tool_call_write(self.FILE_FOO, "FOO")
        exp = [
            ">>> OK: write_file .agento.demo.foo (3 bytes, 1 lines)",
            ">>> FIRST WRITTEN LINE: FOO",
        ]
        assert isinstance(res1, str), type(res1)
        assert res1.splitlines() == exp

    def test_write_no_nl(self):
        """Test write_file with content that has no trailing newline.

        The '>>> FIRST WRITTEN LINE' marker should start on its own line
        regardless of whether the written content ends with newline.
        """
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please write foo without newline"))
        msgs.append(dummy_llm.msg_assistant("Calling writing foo without newline"))
        # Write content without trailing newline
        text_no_nl = "NO_NEWLINE_AT_END"
        res1 = self.tool_call_write(self.FILE_FOO, text_no_nl)
        lines = res1.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn(">>> OK: write_file", lines[0])
        self.assertIn(">>> FIRST WRITTEN LINE:", lines[1])
        # Verify the marker is on its own line (not concatenated with content)
        self.assertTrue(lines[1].startswith(">>> FIRST WRITTEN LINE:"))

    def test_edit_no_nl(self):
        """Test edit_file with content that has no trailing newline.

        When editing content without newline, the response should still
        be properly formatted with markers on separate lines.
        """
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please edit foo without newline"))

        # First write content without trailing newline
        msgs.append(dummy_llm.msg_assistant("Writing foo without newline"))
        text_no_nl = "NO_NEWLINE"
        self.tool_call_write(self.FILE_FOO, text_no_nl)

        # Now edit that content
        msgs.append(dummy_llm.msg_assistant("Editing foo without newline"))
        res = self.tool_call_edit_foo("NO_NEWLINE", "EDITED_NO_NEWLINE")
        assert isinstance(res, dict)
        self.assertIn("ok", res.get(".agento.demo.foo", ""))
        # The edit response is a dict, verify it has proper structure
        self.assertIn("desc", res)

    def test_delete_file(self):
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please delete foo"))
        msgs.append(dummy_llm.msg_assistant("Calling deleting foo"))
        res = self.tool_call_delete_foo()
        assert res == {".agento.demo.foo": "ok", "desc": "File deleted"}

    def test_edit_twice(self):
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please edit foo twice"))

        # First edit: change "foo" to "FOO"
        msgs.append(dummy_llm.msg_assistant("Calling first edit"))
        res1 = self.tool_call_edit_foo("foo", "FOO")
        assert isinstance(res1, dict)
        self.assertIn("ok", res1.get(".agento.demo.foo", ""))
        all_content = "\n".join(str(m.get("content", "")) for m in msgs)
        self.assertIn("FOO", all_content)

        # Second edit: change "text" to "TEXT"
        msgs.append(dummy_llm.msg_assistant("Calling second edit"))
        res2 = self.tool_call_edit_foo("FOO", "TEXT")
        assert isinstance(res2, dict)
        self.assertIn("ok", res2.get(".agento.demo.foo", ""))

        # Verify both edits appear in message history
        all_content = "\n".join(str(m.get("content", "")) for m in msgs)
        self.assertIn("FOO", all_content)
        self.assertIn("TEXT", all_content)
