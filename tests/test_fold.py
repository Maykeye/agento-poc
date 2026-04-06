"""Tests for file folding functionality."""

import unittest
from tests.test_helper import TestBase
from context import ContextMode, context


class TestFold(TestBase):
    """Test cases for file folding functionality."""

    def setUp(self):
        """Set up test fixtures with SUFFIX context mode for folding."""
        super().setUp()
        context.set_context_mode(ContextMode.SUFFIX, reset_ctx_id=True)
        # Clear SUFFIX_CONTEXTS to ensure clean state for each test
        from context.suffix import SUFFIX_CONTEXTS

        SUFFIX_CONTEXTS.clear()

    def test_add_fold_basic(self):
        """Test adding a basic fold to a file."""
        content = "line1\nline2\nline3\nline4\nline5"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please add a fold"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold"))
        res = self.tool_call_add_fold(
            self.FILE_FOO,
            fold_from_line_num=2,
            fold_from_line="line2",
            fold_to_line_num=3,
            fold_to_line="line3",
            name="test_fold",
        )

        self.assertIn("OK: fold added 'test_fold'", res)
        self.assertIn(">>> >> FOLD: lines 2..3 (test_fold)", res)
        for n in "145":
            self.assertIn(f"line{n}", res)
        self.assertNotIn("line2", res)
        self.assertNotIn("line3", res)

    def test_add_fold_single_line(self):
        """Test folding a single line."""
        content = "line1\nline2\nline3"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please add a single line fold"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold"))
        res = self.tool_call_add_fold(
            self.FILE_FOO,
            fold_from_line_num=2,
            fold_from_line="line2",
            fold_to_line_num=2,
            fold_to_line="line2",
            name="single_line_fold",
        )

        self.assertIn(">>> >> FOLD: lines 2..2 (single_line_fold)", res)
        self.assertIn("line1", res)
        self.assertIn("line3", res)
        self.assertNotIn("line2", res)

    def test_add_fold_invalid_line_numbers(self):
        """Test that invalid line numbers are rejected."""
        content = "line1\nline2\nline3"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please add fold with invalid line numbers"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        # Test negative line number
        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold with invalid line"))
        res = self.tool_call_add_fold(
            self.FILE_FOO,
            fold_from_line_num=0,
            fold_from_line="line1",
            fold_to_line_num=2,
            fold_to_line="line2",
            name="invalid",
        )
        self.assertIn("error", res)

        # Test line number exceeds file
        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold with line exceeds"))
        res = self.tool_call_add_fold(
            self.FILE_FOO,
            fold_from_line_num=1,
            fold_from_line="line1",
            fold_to_line_num=10,
            fold_to_line="nonexistent",
            name="invalid",
        )
        self.assertIn("error", res)

        # Test from > to
        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold with from > to"))
        res = self.tool_call_add_fold(
            self.FILE_FOO,
            fold_from_line_num=3,
            fold_from_line="line3",
            fold_to_line_num=1,
            fold_to_line="line1",
            name="invalid",
        )
        self.assertIn("error", res)

    def test_add_fold_content_mismatch(self):
        """Test that content mismatch is detected."""
        content = "line1\nline2\nline3"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please add fold with mismatched content"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        # Wrong from_line content
        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold with mismatch"))
        res = self.tool_call_add_fold(
            self.FILE_FOO,
            fold_from_line_num=2,
            fold_from_line="wrong_content",
            fold_to_line_num=3,
            fold_to_line="line3",
            name="mismatch",
        )
        self.assertIn("error", res)

    def test_unfold(self):
        """Test unfolding a specific fold."""
        content = "line1\nline2\nline3\nline4\nline5"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please add and remove fold"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        # Add a fold
        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold"))
        self.tool_call_add_fold(
            self.FILE_FOO,
            fold_from_line_num=2,
            fold_from_line="line2",
            fold_to_line_num=3,
            fold_to_line="line3",
            name="test_fold",
        )

        # Unfold
        msgs.append(dummy_llm.msg_assistant("Calling file_unfold"))
        res = self.tool_call_unfold(self.FILE_FOO, "test_fold")

        self.assertIn("OK: fold removed 'test_fold'", res)
        for n in "12345":
            self.assertIn(f"line{n}", res)
        self.assertNotIn("FOLD:", res)

    def test_unfold_nonexistent(self):
        """Test unfolding a fold that doesn't exist."""
        content = "line1\nline2\nline3"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please unfold nonexistent fold"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        msgs.append(dummy_llm.msg_assistant("Calling file_unfold nonexistent"))
        res = self.tool_call_unfold(self.FILE_FOO, "nonexistent")
        self.assertIn("error", res)
        self.assertIn("No folds exist for file", res["error"])

    def test_unfold_all(self):
        """Test removing all folds."""
        content = "line1\nline2\nline3\nline4\nline5\nline6"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please add and remove all folds"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        # Add multiple folds
        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold first"))
        self.tool_call_add_fold(
            self.FILE_FOO,
            fold_from_line_num=2,
            fold_from_line="line2",
            fold_to_line_num=3,
            fold_to_line="line3",
            name="fold1",
        )

        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold second"))
        self.tool_call_add_fold(
            self.FILE_FOO,
            fold_from_line_num=4,
            fold_from_line="line4",
            fold_to_line_num=5,
            fold_to_line="line5",
            name="fold2",
        )

        # Unfold all
        msgs.append(dummy_llm.msg_assistant("Calling file_unfold_all"))
        res = self.tool_call_unfold_all(self.FILE_FOO)

        self.assertIn("OK: 2 folds removed", res)
        for n in "123456":
            self.assertIn(f"line{n}", res)
        self.assertNotIn("FOLD:", res)

    def test_multiple_folds(self):
        """Test multiple non-overlapping folds."""
        content = "line1\nline2\nline3\nline4\nline5\nline6\nline7"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please add multiple folds"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        # Add first fold
        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold first"))
        self.tool_call_add_fold(
            self.FILE_FOO,
            fold_from_line_num=2,
            fold_from_line="line2",
            fold_to_line_num=3,
            fold_to_line="line3",
            name="first_fold",
        )

        # Add second fold
        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold second"))
        res = self.tool_call_add_fold(
            self.FILE_FOO,
            fold_from_line_num=5,
            fold_from_line="line5",
            fold_to_line_num=6,
            fold_to_line="line6",
            name="second_fold",
        )
        # TODO: test: content removed in first message

        self.assertIn("OK: fold added 'second_fold'", res)
        self.assertIn(">>> >> FOLD: lines 2..3 (first_fold)", res)
        self.assertIn(">>> >> FOLD: lines 5..6 (second_fold)", res)
        for n in "147":
            self.assertIn(f"line{n}", res)

    def test_multiple_folds_reload(self):
        """Test multiple non-overlapping folds after reloading."""

        def do_fold(start, end):
            return self.tool_call_add_fold(
                self.FILE_FOO,
                fold_from_line_num=start,
                fold_from_line=f"line{start}",
                fold_to_line_num=end,
                fold_to_line=f"line{end}",
                name=f"fold_{start}_{end}",
            )

        content = "line1\nline2\nline3\nline4\nline5\nline6\nline7"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please add multiple folds"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)
        do_fold(2, 3)
        do_fold(5, 6)
        self.tool_call_read(self.FILE_FOO)
        res_idx = len(msgs) - 1
        self.epilogue()

        res = msgs[res_idx]["content"]
        self.assertIn(">>> >> FOLD: lines 2..3 (fold_2_3)", res)
        self.assertIn(">>> >> FOLD: lines 5..6 (fold_5_6)", res)
        for n in "147":
            self.assertIn(f"line{n}", res)

    def test_overlapping_folds_rejected(self):
        """Test that overlapping folds are rejected."""
        content = "line1\nline2\nline3\nline4\nline5"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please try overlapping folds"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        # Add first fold
        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold first"))
        self.tool_call_add_fold(
            self.FILE_FOO,
            fold_from_line_num=2,
            fold_from_line="line2",
            fold_to_line_num=3,
            fold_to_line="line3",
            name="fold1",
        )

        # Try to add overlapping fold
        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold overlapping"))
        res = self.tool_call_add_fold(
            self.FILE_FOO,
            fold_from_line_num=3,
            fold_from_line="line3",
            fold_to_line_num=4,
            fold_to_line="line4",
            name="fold2",
        )

        self.assertIn("error", res)
        self.assertIn("Fold 'fold2' overlaps with existing fold", res["error"])

    def test_duplicate_fold_name_rejected(self):
        """Test that duplicate fold names are rejected."""
        content = "line1\nline2\nline3\nline4\nline5"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please try duplicate fold name"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        # Add first fold
        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold first"))
        self.tool_call_add_fold(
            self.FILE_FOO,
            fold_from_line_num=2,
            fold_from_line="line2",
            fold_to_line_num=3,
            fold_to_line="line3",
            name="duplicate",
        )

        # Try to add fold with same name
        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold duplicate name"))
        res = self.tool_call_add_fold(
            self.FILE_FOO,
            fold_from_line_num=4,
            fold_from_line="line4",
            fold_to_line_num=5,
            fold_to_line="line5",
            name="duplicate",
        )

        self.assertIn("error", res)
        self.assertIn(
            "Fold with name 'duplicate' already exists for .agento.demo.foo",
            res["error"],
        )

    def test_has_folds(self):
        """Test has_folds method."""
        content = "line1\nline2\nline3"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please check folds"))

        from context import context_handler

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        # No folds yet
        self.assertFalse(context_handler().has_folds(self.FILE_FOO.name))

        # Add a fold
        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold"))
        self.tool_call_add_fold(
            self.FILE_FOO,
            fold_from_line_num=2,
            fold_from_line="line2",
            fold_to_line_num=2,
            fold_to_line="line2",
            name="test",
        )

        self.assertTrue(context_handler().has_folds(self.FILE_FOO.name))

    def test_get_folds(self):
        """Test get_folds method."""
        content = "line1\nline2\nline3\nline4\nline5"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please check folds"))

        from context import context_handler

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        # Add a fold
        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold"))
        self.tool_call_add_fold(
            self.FILE_FOO,
            fold_from_line_num=2,
            fold_from_line="line2",
            fold_to_line_num=3,
            fold_to_line="line3",
            name="test_fold",
        )

        folds = context_handler().get_folds(self.FILE_FOO.name)
        self.assertEqual(len(folds), 1)
        self.assertEqual(folds[0].name, "test_fold")
        self.assertEqual(folds[0].start_line, 2)
        self.assertEqual(folds[0].end_line, 3)

    def test_file_not_read(self):
        """Test that folding a file that hasn't been read fails."""
        content = "line1\nline2\nline3\nline4\nline5"
        # Use unique filename to avoid context from other tests
        self.FILE_FOO.write_text(content)

        # Intentionally NOT reading the file first to test error case

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please fold unread file"))

        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold on unread file"))
        res = self.tool_call_add_fold(
            self.FILE_FOO,
            fold_from_line_num=1,
            fold_from_line="line1",
            fold_to_line_num=1,
            fold_to_line="line1",
            name="test",
        )

        self.assertIn("error", res)
        self.assertIn("has not been read", res["error"])

    def test_fold_at_start(self):
        """Test folding from the first line."""
        content = "line1\nline2\nline3"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please fold from start"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold"))
        res = self.tool_call_add_fold(
            self.FILE_FOO,
            fold_from_line_num=1,
            fold_from_line="line1",
            fold_to_line_num=2,
            fold_to_line="line2",
            name="start_fold",
        )

        self.assertIn(">>> >> FOLD: lines 1..2 (start_fold)", res)
        self.assertIn("line3", res)
        self.assertNotIn("line1", res)
        self.assertNotIn("line2", res)

    def test_fold_at_end(self):
        """Test folding to the last line."""
        content = "line1\nline2\nline3"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please fold to end"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold"))
        res = self.tool_call_add_fold(
            self.FILE_FOO,
            fold_from_line_num=2,
            fold_from_line="line2",
            fold_to_line_num=3,
            fold_to_line="line3",
            name="end_fold",
        )

        self.assertIn(">>> >> FOLD: lines 2..3 (end_fold)", res)
        self.assertIn("line1", res)
        self.assertNotIn("line2", res)
        self.assertNotIn("line3", res)

    def test_fold_whole_file(self):
        """Test folding the entire file."""
        content = "line1\nline2\nline3"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please fold whole file"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold"))
        res = self.tool_call_add_fold(
            self.FILE_FOO,
            fold_from_line_num=1,
            fold_from_line="line1",
            fold_to_line_num=3,
            fold_to_line="line3",
            name="whole_fold",
        )

        self.assertIn(">>> >> FOLD: lines 1..3 (whole_fold)", res)
        self.assertNotIn("line1", res)
        self.assertNotIn("line2", res)
        self.assertNotIn("line3", res)

    def test_read_then_fold(self):
        _, msgs = self.init_llm_msgs()
        self.FILE_FOO.write_text("line1\nline2\nline3\nline4")
        self.tool_call_read(self.FILE_FOO)
        read_idx = len(msgs) - 1
        self.epilogue()
        self.tool_call_add_fold(self.FILE_FOO, 2, "line2", 3, "line3", "<2_3>")
        fold_idx = len(msgs) - 1
        self.epilogue()

        res_read = msgs[read_idx]["content"]
        res_fold = msgs[fold_idx]["content"]
        # read is no longer valid
        self.assertIn("out of date", res_read)
        # fold is now up to date
        self.assertNotIn("out of date", res_fold)
        # fold has fold name
        self.assertIn("<2_3>", res_fold)
        # fold has raw data
        self.assertIn("line1", res_fold)
        self.assertIn("line4", res_fold)
        # fold has no folded data
        self.assertNotIn("line2", res_fold)
        self.assertNotIn("line3", res_fold)


if __name__ == "__main__":
    unittest.main()
