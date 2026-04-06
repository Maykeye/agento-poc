"""Tests for regex-based file folding functionality."""

import unittest
from context.suffix import SuffixHandler
from tests.test_helper import TestBase
from context import ContextMode, context


class TestFoldRegex(TestBase):
    """Test cases for regex-based file folding functionality."""

    def setUp(self):
        """Set up test fixtures with SUFFIX context mode for folding."""
        super().setUp()
        context.set_context_mode(ContextMode.SUFFIX, reset_ctx_id=True)
        # Clear SUFFIX_CONTEXTS to ensure clean state for each test
        from context.suffix import SUFFIX_CONTEXTS

        SUFFIX_CONTEXTS.clear()
        suffix: SuffixHandler = context.context_handler()  # type: ignore
        self.prefix = suffix.prefix

    def test_add_fold_regex_basic(self):
        """Test adding a fold using regex patterns."""
        content = "line1\nline2\nline3\nline4\nline5"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please add a fold using regex"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        # Use regex patterns to fold lines 2-3
        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold with regex"))
        res = self.tool_call_add_fold_regex(
            self.FILE_FOO,
            start_pattern="^line2$",
            end_pattern="^line3$",
            name="regex_fold",
        )

        self.assertIn("OK: fold added 'regex_fold'", res)
        self.assertIn(f"{self.prefix} >> FOLD: lines 2..3 (regex_fold)", res)
        for n in "145":
            self.assertIn(f"line{n}", res)
        self.assertNotIn("line2", res)
        self.assertNotIn("line3", res)

    def test_add_fold_regex_multiline(self):
        """Test adding a fold using multiline regex patterns."""
        content = "def my_function():\n    pass\n"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please add a fold using multiline regex"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        # Use multiline pattern to fold from def to pass
        msgs.append(
            dummy_llm.msg_assistant("Calling file_add_fold with multiline regex")
        )
        res = self.tool_call_add_fold_regex(
            self.FILE_FOO,
            start_pattern="^def my_function\\(\\):",
            end_pattern="^\\s*pass$",
            name="function_fold",
        )

        self.assertIn("OK: fold added 'function_fold'", res)
        self.assertIn(f"{self.prefix} >> FOLD: lines 1..2 (function_fold)", res)
        self.assertNotIn("def my_function():", res)
        self.assertNotIn("pass", res)

    def test_add_fold_regex_multiline_block(self):
        """Test adding a fold using multiline regex for a code block."""
        content = "class MyClass:\n    def __init__(self):\n        self.value = 0\n"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please add a fold using multiline regex"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        # Use multiline pattern to fold the __init__ method
        msgs.append(
            dummy_llm.msg_assistant("Calling file_add_fold with multiline regex")
        )
        res = self.tool_call_add_fold_regex(
            self.FILE_FOO,
            start_pattern="^\\s*def __init__\\(self\\):",
            end_pattern="^\\s*self\\.value = 0$",
            name="init_fold",
        )

        self.assertIn("OK: fold added 'init_fold'", res)
        self.assertIn(f"{self.prefix} >> FOLD: lines 2..3 (init_fold)", res)
        self.assertIn("class MyClass:", res)
        self.assertNotIn("def __init__", res)

    def test_add_fold_regex_not_found(self):
        """Test error when start pattern not found."""
        content = "line1\nline2\nline3"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please add fold with not found pattern"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        # Pattern doesn't exist
        msgs.append(
            dummy_llm.msg_assistant("Calling file_add_fold with not found pattern")
        )
        res = self.tool_call_add_fold_regex(
            self.FILE_FOO,
            start_pattern="^nonexistent$",
            end_pattern="^line3$",
            name="not_found_fold",
        )

        self.assertIn("error", res)
        self.assertIn("not found in visible content", res["error"])

    def test_add_fold_regex_multiple_matches(self):
        """Test error when pattern matches multiple times."""
        content = "line1\nline2\nline2\nline3"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please add fold with multiple matches"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        # Pattern matches twice
        msgs.append(
            dummy_llm.msg_assistant("Calling file_add_fold with multiple matches")
        )
        res = self.tool_call_add_fold_regex(
            self.FILE_FOO,
            start_pattern="^line2$",
            end_pattern="^line3$",
            name="multiple_matches_fold",
        )

        self.assertIn("error", res)
        self.assertIn("matches 2 times", res["error"])

    def test_add_fold_regex_invalid_pattern(self):
        """Test error when pattern is invalid regex."""
        content = "line1\nline2\nline3"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please add fold with invalid pattern"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        # Invalid regex pattern
        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold with invalid regex"))
        res = self.tool_call_add_fold_regex(
            self.FILE_FOO,
            start_pattern="[invalid(regex",
            end_pattern="^line3$",
            name="invalid_pattern_fold",
        )

        self.assertIn("error", res)
        self.assertIn("Invalid start_pattern regex", res["error"])

    def test_add_fold_regex_with_existing_folds(self):
        """Test adding a fold when other folds already exist."""
        content = "line1\nline2\nline3\nline4\nline5\nline6\nline7"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please add folds"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        # Add first fold using regex
        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold first"))
        res1 = self.tool_call_add_fold_regex(
            self.FILE_FOO,
            start_pattern="^line2$",
            end_pattern="^line3$",
            name="first_fold",
        )
        self.assertIn("OK: fold added 'first_fold'", res1)

        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold second"))
        res2 = self.tool_call_add_fold_regex(
            self.FILE_FOO,
            start_pattern="^line5$",
            end_pattern="^line6$",
            name="second_fold",
        )
        self.assertIn("OK: fold added 'second_fold'", res2)
        self.assertIn(f"{self.prefix} >> FOLD: lines 2..3 (first_fold)", res2)
        # The fold should be on lines 5..6 (visible lines 4-5 map to actual lines 5-6)
        self.assertIn(f"{self.prefix} >> FOLD: lines 5..6 (second_fold)", res2)
        # Only lines 1, 4, 7 should be visible (lines 5 and 6 are folded)
        for n in "147":
            self.assertIn(f"line{n}", res2)
        self.assertNotIn("line5", res2)
        self.assertNotIn("line6", res2)

    def test_add_fold_regex_end_pattern_not_after_start(self):
        """Test error when end pattern is before start pattern."""
        content = "line1\nline2\nline3\nline4"
        self.FILE_FOO.write_text(content)

        dummy_llm, msgs = self.init_llm_msgs()
        msgs.append(dummy_llm.msg_user("Please add fold with end before start"))

        # Read the file first
        msgs.append(dummy_llm.msg_assistant("Calling read_file"))
        self.tool_call_read(self.FILE_FOO)

        # Try to fold with end pattern before start pattern
        msgs.append(dummy_llm.msg_assistant("Calling file_add_fold with wrong order"))
        res = self.tool_call_add_fold_regex(
            self.FILE_FOO,
            start_pattern="^line3$",
            end_pattern="^line2$",
            name="wrong_order_fold",
        )

        self.assertIn("error", res)
        self.assertIn("not found after start_pattern", res["error"])

    def test_several_folds(self):
        self.init_llm_msgs()

        self.FILE_FOO.write_text("""## Section 1
This is a test section 1
## Section 2
Line A
Line B
## Section 3
Line 1
Line 2
## Section 4
There is something to say fill this section with, I dunno,
Ignore context, it exists just to test api result\n\n""")

        self.tool_call_read(self.FILE_FOO)
        res = self.tool_call_add_fold_regex(
            self.FILE_FOO, "^## Section 1$", "^This is a test section 1$", "Section 1"
        )
        self.assertNotIn("error", res)

        res = self.tool_call_add_fold_regex(
            self.FILE_FOO, "^## Section 2$", "^Line B$", "Section 2"
        )
        self.assertNotIn("error", res)

        res = self.tool_call_add_fold_regex(
            self.FILE_FOO,
            "^## Section 4$",
            "^Ignore context, it exists just to test api result$",
            "Section 4",
        )
        self.assertNotIn("error", res)


if __name__ == "__main__":
    unittest.main()
