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
{pfx} OK: read_file: .agento.demo.foo
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
            ">>> OK: write_file: .agento.demo.foo",
            ">>> === CONTENT START ===",
            "FOO",
            ">>> === CONTENT END ===",
        ]
        assert isinstance(res1, str), type(res1)
        self.assertEqual(res1.splitlines(), exp)

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
        self.assertEqual(len(lines), 4)
        self.assertIn(">>> OK: write_file: ", lines[0])
        self.assertEqual(">>> === CONTENT START ===", lines[1])

    def test_delete_file(self):
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please delete foo"))
        msgs.append(dummy_llm.msg_assistant("Calling deleting foo"))
        res = self.tool_call_delete_foo()
        exp = """\
>>> OK: delete_file: .agento.demo.foo
>>> === CONTENT START ===
(file deleted)
>>> === CONTENT END ==="""
        self.assertEqual(res, exp)
