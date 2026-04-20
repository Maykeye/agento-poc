import json
import context
import tool_io
from context import ContextMode
from context.suffix import SuffixHandler
from context.suffix import SUFFIX_CONTEXTS
from context.context_handler import ContextEntry
from tests.test_context import TestContextBase


class TestSuffixPrune(TestContextBase):
    """Test aggressive pruning of old tool calls that edit files."""

    def setUp(self):
        super().setUp()
        context.set_context_mode(ContextMode.SUFFIX, reset_ctx_id=True)
        # Set keep_old_edits to 3 for testing
        SuffixHandler.keep_old_edits = 3

    def tearDown(self):
        # Reset to default value
        SuffixHandler.keep_old_edits = 5
        super().tearDown()

    def _get_tool_calls_for_path(self, msgs: list[dict], path: str) -> list[dict]:
        """Extract all tool calls that edit a specific path."""
        tool_calls = []
        for msg in msgs:
            if msg.get("role") != "assistant":
                continue
            for tc in msg.get("tool_calls", []):
                func_info = tc.get("function", {})
                args_str = func_info.get("arguments", "{}")
                try:
                    args = json.loads(args_str)
                    if args.get("path") == path:
                        tool_calls.append({"name": func_info.get("name"), "args": args})
                except (json.JSONDecodeError, TypeError):
                    pass
        return tool_calls

    def test_edit_three_times_no_prune(self):
        """Test editing file 3 times keeps all tool calls (within keep_old_edits)."""
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please edit foo three times"))

        # First edit: CTX(0)
        msgs.append(dummy_llm.msg_assistant("Editing foo first time"))
        self.tool_call_edit_foo("foo\ntext", "EDIT1")

        # Second edit: CTX(1)
        msgs.append(dummy_llm.msg_assistant("Editing foo second time"))
        self.tool_call_edit_foo("EDIT1", "EDIT2")

        # Third edit: CTX(2)
        msgs.append(dummy_llm.msg_assistant("Editing foo third time"))
        self.tool_call_edit_foo("EDIT2", "EDIT3")

        self.epilogue()

        # Verify all 3 tool calls are preserved (keep_old_edits = 3)
        edit_calls = self._get_tool_calls_for_path(msgs, self.FILE_FOO.name)
        self.assertEqual(len(edit_calls), 3)

        # None should be pruned
        for call in edit_calls:
            self.assertNotIn("cleanup", call["args"])

    def test_edit_four_times_one_pruned(self):
        """Test editing file 4 times prunes the 1st tool call."""
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please edit foo four times"))

        # First edit: CTX(0)
        msgs.append(dummy_llm.msg_assistant("Editing foo first time"))
        self.tool_call_edit_foo("foo\ntext", "EDIT1")

        # Second edit: CTX(1)
        msgs.append(dummy_llm.msg_assistant("Editing foo second time"))
        self.tool_call_edit_foo("EDIT1", "EDIT2")

        # Third edit: CTX(2)
        msgs.append(dummy_llm.msg_assistant("Editing foo third time"))
        self.tool_call_edit_foo("EDIT2", "EDIT3")

        # Fourth edit: CTX(3) - this triggers pruning
        msgs.append(dummy_llm.msg_assistant("Editing foo fourth time"))
        self.tool_call_edit_foo("EDIT3", "EDIT4")

        self.epilogue()

        # Verify all 4 tool calls exist
        edit_calls = self._get_tool_calls_for_path(msgs, self.FILE_FOO.name)
        self.assertEqual(len(edit_calls), 4)

        # First one should be pruned (cleanup message)
        self.assertIn("cleanup", edit_calls[0]["args"])
        self.assertEqual(edit_calls[0]["args"]["path"], self.FILE_FOO.name)
        self.assertEqual(edit_calls[0]["args"]["cleanup"], "the call is removed")

        # Others should be preserved
        for i, call in enumerate(edit_calls[1:], 1):
            self.assertNotIn("cleanup", call["args"], f"Call {i} should not be pruned")

    def test_edit_five_times_two_pruned(self):
        """Test editing file 5 times prunes the 2 oldest tool calls."""
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please edit foo five times"))

        # Edits 1-5
        for i in range(5):
            prev_content = "foo\ntext" if i == 0 else f"EDIT{i}"
            new_content = f"EDIT{i+1}"
            msgs.append(dummy_llm.msg_assistant(f"Editing foo {i+1} time"))
            self.tool_call_edit_foo(prev_content, new_content)

        self.epilogue()

        # Verify all 5 tool calls exist
        edit_calls = self._get_tool_calls_for_path(msgs, self.FILE_FOO.name)
        self.assertEqual(len(edit_calls), 5)

        # First 2 should be pruned (keep_old_edits = 3, so 5-3 = 2 pruned)
        for i in range(2):
            self.assertIn(
                "cleanup", edit_calls[i]["args"], f"Call {i} should be pruned"
            )
            self.assertEqual(edit_calls[i]["args"]["cleanup"], "the call is removed")

        # Last 3 should be preserved
        for i in range(2, 5):
            self.assertNotIn(
                "cleanup", edit_calls[i]["args"], f"Call {i} should not be pruned"
            )

    def test_write_then_edit_prunes_writes(self):
        """Test that write_file and edit_file are both tracked and pruned."""
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Write and edit foo"))

        # Write: CTX(0)
        msgs.append(dummy_llm.msg_assistant("Writing foo"))
        self.tool_call_write(self.FILE_FOO, "INITIAL")

        # Edit 1: CTX(1)
        msgs.append(dummy_llm.msg_assistant("Editing foo 1"))
        self.tool_call_edit_foo("INITIAL", "EDIT1")

        # Edit 2: CTX(2)
        msgs.append(dummy_llm.msg_assistant("Editing foo 2"))
        self.tool_call_edit_foo("EDIT1", "EDIT2")

        # Edit 3: CTX(3)
        msgs.append(dummy_llm.msg_assistant("Editing foo 3"))
        self.tool_call_edit_foo("EDIT2", "EDIT3")

        # Edit 4: CTX(4) - triggers pruning
        msgs.append(dummy_llm.msg_assistant("Editing foo 4"))
        self.tool_call_edit_foo("EDIT3", "EDIT4")

        self.epilogue()

        # Get all editing tool calls
        edit_calls = self._get_tool_calls_for_path(msgs, self.FILE_FOO.name)
        self.assertEqual(len(edit_calls), 5)

        # First 2 should be pruned (keep_old_edits = 3, so 5-3=2 pruned)
        self.assertEqual(edit_calls[0]["name"], "write_file")
        self.assertIn("cleanup", edit_calls[0]["args"])
        self.assertIn("cleanup", edit_calls[1]["args"])

        # Last 3 should be preserved
        for i in range(2, 5):
            self.assertNotIn("cleanup", edit_calls[i]["args"])

    def test_non_editing_tool_survives(self):
        """Test that non-editing tools (read_file) are not pruned."""
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Read and edit foo"))

        # Read: not an editing tool
        msgs.append(dummy_llm.msg_assistant("Reading foo"))
        self.tool_call_read(self.FILE_FOO)

        # Edit 1
        msgs.append(dummy_llm.msg_assistant("Editing foo 1"))
        self.tool_call_edit_foo("foo\ntext", "EDIT1")

        # Edit 2
        msgs.append(dummy_llm.msg_assistant("Editing foo 2"))
        self.tool_call_edit_foo("EDIT1", "EDIT2")

        # Edit 3
        msgs.append(dummy_llm.msg_assistant("Editing foo 3"))
        self.tool_call_edit_foo("EDIT2", "EDIT3")

        # Edit 4 - triggers pruning
        msgs.append(dummy_llm.msg_assistant("Editing foo 4"))
        self.tool_call_edit_foo("EDIT3", "EDIT4")

        self.epilogue()

        # Get all tool calls
        all_calls = self._get_tool_calls_for_path(msgs, self.FILE_FOO.name)

        # First should be read_file, not pruned
        self.assertEqual(all_calls[0]["name"], "read_file")
        self.assertNotIn("cleanup", all_calls[0]["args"])

        # Edit 1 should be pruned
        self.assertEqual(all_calls[1]["name"], "search_replace_once")
        self.assertIn("cleanup", all_calls[1]["args"])

    def test_different_file_not_pruned(self):
        """Test that edits to different files are tracked separately."""
        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Edit both foo and bar"))

        # Edit foo 4 times
        foo_content = "foo\ntext"
        for i in range(4):
            prev_content = foo_content if i == 0 else f"FOO_EDIT{i}"
            new_content = f"FOO_EDIT{i+1}"
            msgs.append(dummy_llm.msg_assistant(f"Editing foo {i+1}"))
            self.tool_call_edit_foo(prev_content, new_content)

        # Edit bar 4 times
        bar_content = "bar\nvalue"
        for i in range(4):
            prev_content = bar_content if i == 0 else f"BAR_EDIT{i}"
            new_content = f"BAR_EDIT{i+1}"
            msgs.append(dummy_llm.msg_assistant(f"Editing bar {i+1}"))
            self.tool_call(
                tool_io.ToolEditFile(),
                path=self.FILE_BAR.name,
                replace_from=prev_content,
                replace_with=new_content,
            )

        self.epilogue()

        # Check foo calls
        foo_calls = self._get_tool_calls_for_path(msgs, self.FILE_FOO.name)
        self.assertEqual(len(foo_calls), 4)
        # First foo edit should be pruned
        self.assertIn("cleanup", foo_calls[0]["args"])

        # Check bar calls
        bar_calls = self._get_tool_calls_for_path(msgs, self.FILE_BAR.name)
        self.assertEqual(len(bar_calls), 4)
        # First bar edit should be pruned
        self.assertIn("cleanup", bar_calls[0]["args"])

    def test_multiple_tool_calls_same_message(self):
        """Test pruning when a message has multiple tool calls for same file."""
        llm, msgs = self.init_llm_msgs()
        msgs.append(llm.msg_user("Edit foo with multiple calls per message"))

        # First message with 2 edit calls for foo
        msgs.append(llm.msg_assistant("Editing foo with 2 calls"))
        edit = tool_io.ToolEditFile()

        llm.append_tool_call(
            edit.name,
            path=self.FILE_FOO.name,
            replace_from="text1",
            replace_with="EDIT1",
        )
        llm.append_tool_call(
            edit.name,
            path=self.FILE_FOO.name,
            replace_from="EDIT1",
            replace_with="EDIT2",
        )

        last_tool_call = llm.messages().pop()["tool_calls"]
        llm.messages()[-1]["tool_calls"].extend(last_tool_call)

        self.FILE_FOO.write_text("EDIT2")

        new_id = "unused"
        SUFFIX_CONTEXTS[self.FILE_FOO.name] = ContextEntry(
            self.FILE_FOO.name, "EDIT2", new_id, "edit_file"
        )
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": -2,
                "name": "search_replace_once",
                "content": f">>> ID: {new_id} OPERATION: search_replace_once CTX-IO-FILE: {self.FILE_FOO.name}\n>>> OK: edit {self.FILE_FOO.name}\n>>> === CONTENT START ===\nEDIT2\n>>> === CONTENT END ===",
            }
        )
        msgs.append(
            {
                "role": "tool",
                "tool_call_id": -1,
                "name": "search_replace_once",
                "content": f">>> ID: {new_id} OPERATION: search_replace_once CTX-IO-FILE: {self.FILE_FOO.name}\n>>> OK: edit {self.FILE_FOO.name}\n>>> === CONTENT START ===\nEDIT2\n>>> === CONTENT END ===",
            }
        )

        # Second edit
        msgs.append(llm.msg_assistant("Editing foo again"))
        self.tool_call_edit_foo("EDIT2", "EDIT3")

        # Third edit
        msgs.append(llm.msg_assistant("Editing foo again"))
        self.tool_call_edit_foo("EDIT3", "EDIT4")

        # Fourth edit - triggers pruning
        msgs.append(llm.msg_assistant("Editing foo again"))
        self.tool_call_edit_foo("EDIT4", "EDIT5")

        self.epilogue()

        # Get all edit calls
        edit_calls = self._get_tool_calls_for_path(msgs, self.FILE_FOO.name)
        self.assertEqual(
            len(edit_calls), 5
        )  # 2 from first msg + 3 subsequent = 5 total

        # First 2 should be pruned (keep_old_edits = 3, so 5-3=2 pruned)
        self.assertIn("cleanup", edit_calls[0]["args"])
        self.assertIn("cleanup", edit_calls[1]["args"])

        # Last 3 should be preserved
        for i in range(2, 5):
            self.assertNotIn("cleanup", edit_calls[i]["args"])
