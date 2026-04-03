from pathlib import Path

from context import ContextMode, context_handler
from tests.test_helper import TestBase
import context


class TestFold(TestBase):
    """Test suite for file folding operations."""

    def setUp(self):
        super().setUp()
        context.set_context_mode(ContextMode.SUFFIX, reset_ctx_id=True)

    def init_test_file(self):
        """Create a test file with known content."""
        test_content = """use a::{b,c};

struct Foo(i32);
struct FooBar(i32);

// The main
fn main() {
    if lorem() > ipsum() {
        return;
    }
}

fn lorem() -> i32 {1}
fn ipsum() -> i32 {1}
"""
        self.FILE_FOO.write_text(test_content)
        return test_content

    def test_add_fold_head(self):
        """Test adding a fold at the head of the file."""
        _, messages = self.init_test_llm()
        self.init_test_file()

        # Read the file first to load it into context
        res = self.tool_call_read_with_check(self.FILE_FOO, 0)
        read_msg_idx = len(messages) - 1
        self.assertIn("struct FooBar(i32);", res)

        # Add a fold at the head using the tool
        fold_result = self.tool_call_add_fold(
            self.FILE_FOO, "head", "struct FooBar(i32);", "Helper functions structs"
        )

        # Check fold was added successfully
        self.assertIsInstance(fold_result, str)
        self.assertIn("Added fold 'Helper functions structs'", fold_result)
        self.assertIn("lines 1..4", fold_result)

        # Call epilogue to trigger prepare_current_llm which applies folding
        self.epilogue()

        # Find the read_file tool message and check it contains the fold
        read_msg = messages[read_msg_idx]
        content = read_msg["content"]

        # Check that the content now contains the fold marker
        self.assertIn(">>> FOLD: Helper functions structs (lines 1..4)", content)
        # Original hidden content should not be in the message
        self.assertNotIn("use a::{b,c};", content)
        self.assertNotIn("struct FooBar(i32);", content)
        # Visible content should still be there
        self.assertIn("fn main()", content)

    def test_add_fold_tail(self):
        """Test adding a fold at the tail of the file."""
        _, messages = self.init_test_llm()
        self.init_test_file()

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)
        read_msg_idx = len(messages) - 1

        # Add a fold at the tail using the tool
        fold_result = self.tool_call_add_fold(
            self.FILE_FOO, "tail", "fn lorem()", "Lorem/ipsum implementation"
        )

        # Check fold was added successfully
        self.assertIsInstance(fold_result, str)
        self.assertIn("Added fold 'Lorem/ipsum implementation'", fold_result)
        self.assertIn("lines 13..14", fold_result)

        # Call epilogue to trigger prepare_current_llm which applies folding
        self.epilogue()

        # Find the read_file tool message and check it contains the fold
        read_msg = messages[read_msg_idx]
        content = read_msg["content"]

        # Check that the content now contains the fold marker
        self.assertIn(">>> FOLD: Lorem/ipsum implementation (lines 13..14)", content)
        # Original hidden content should not be in the message
        self.assertNotIn("fn lorem() -> i32 {1}", content)
        self.assertNotIn("fn ipsum() -> i32 {1}", content)
        # Visible content should still be there
        self.assertIn("fn main()", content)

    def test_add_multiple_folds(self):
        """Test adding multiple folds sequentially."""
        _, messages = self.init_test_llm()
        self.init_test_file()

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)
        read_msg_idx = len(messages) - 1

        # Add head fold using the tool
        fold_result1 = self.tool_call_add_fold(
            self.FILE_FOO, "head", "struct FooBar(i32);", "Helper functions structs"
        )
        self.assertIn("Added fold 'Helper functions structs'", fold_result1)

        # Add tail fold using the tool
        fold_result2 = self.tool_call_add_fold(
            self.FILE_FOO, "tail", "fn lorem()", "Lorem/ipsum implementation"
        )
        self.assertIn("Added fold 'Lorem/ipsum implementation'", fold_result2)

        # Add another head fold using the tool
        fold_result3 = self.tool_call_add_fold(
            self.FILE_FOO, "head", "// The main", "Main() comments"
        )
        self.assertIn("Added fold 'Main() comments'", fold_result3)
        # The fold should start after the previous head fold (line 4) + 1 buffer line (line 5)
        # So the fold should be lines 6..6
        self.assertIn("lines 6..6", fold_result3)

        # Call epilogue to trigger prepare_current_llm which applies folding
        self.epilogue()

        # Find the read_file tool message and check it contains all folds
        read_msg = messages[read_msg_idx]
        content = read_msg["content"]

        # Check all folds are applied
        self.assertIn(">>> FOLD: Helper functions structs (lines 1..4)", content)
        self.assertIn(">>> FOLD: Main() comments (lines 6..6)", content)
        self.assertIn(">>> FOLD: Lorem/ipsum implementation (lines 13..14)", content)
        # Visible content should still be there
        self.assertIn("fn main()", content)
        # Hidden content should not be visible
        self.assertNotIn("struct FooBar(i32);", content)
        self.assertNotIn("fn lorem() -> i32 {1}", content)

    def test_add_fold_overlapping_head_tail(self):
        """Test that overlapping head and tail folds are rejected."""
        self.init_test_llm()
        self.init_test_file()

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)

        # Add a head fold that goes deep using the tool
        fold_result1 = self.tool_call_add_fold(
            self.FILE_FOO, "head", "return;", "Top section"
        )
        self.assertIn("Added fold 'Top section'", fold_result1)

        # Try to add a tail fold that would overlap (pattern before the head fold ends)
        fold_result2 = self.tool_call_add_fold(
            self.FILE_FOO, "tail", "fn main()", "Main function"
        )

        # This should fail due to overlap
        self.assertIsInstance(fold_result2, dict)
        self.assertIn("error", fold_result2)

    def test_add_fold_overlapping_same_position(self):
        """Test that overlapping folds at the same position are rejected."""
        self.init_test_llm()
        self.init_test_file()

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)

        # Add a head fold using the tool
        fold_result1 = self.tool_call_add_fold(
            self.FILE_FOO, "head", "struct FooBar(i32);", "First head fold"
        )
        self.assertIn("Added fold 'First head fold'", fold_result1)

        # Try to add another head fold that would overlap
        # Pattern "struct Foo" appears before "struct FooBar", so would overlap
        fold_result2 = self.tool_call_add_fold(
            self.FILE_FOO, "head", "struct Foo(i32);", "Second head fold"
        )

        # This should fail due to overlap
        self.assertIsInstance(fold_result2, dict)
        self.assertIn("error", fold_result2)

    def test_add_fold_no_space_between_folds(self):
        """Test that folds with no raw line between them are rejected."""
        self.init_test_llm()
        # Create a file where head and tail folds would touch
        test_content = """line1
line2
line3
line4
fn main() {
    return;
}
fn helper() {
    return;
}
"""
        self.FILE_FOO.write_text(test_content)

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)

        # Add a head fold ending at line 4 using the tool
        fold_result1 = self.tool_call_add_fold(
            self.FILE_FOO, "head", "line4", "First fold"
        )
        self.assertIn("Added fold 'First fold'", fold_result1)

        # Try to add a tail fold starting at line 5 (no buffer line)
        fold_result2 = self.tool_call_add_fold(
            self.FILE_FOO, "tail", "fn main()", "Second fold"
        )

        # This should fail due to no buffer line between folds
        self.assertIsInstance(fold_result2, dict)
        self.assertIn("error", fold_result2)

    def test_add_fold_duplicate_name(self):
        """Test that adding a fold with a duplicate name is rejected."""
        self.init_test_llm()
        self.init_test_file()

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)

        # Add a fold using the tool
        fold_result1 = self.tool_call_add_fold(
            self.FILE_FOO, "head", "struct FooBar(i32);", "MyFold"
        )
        self.assertIn("Added fold 'MyFold'", fold_result1)

        # Try to add another fold with the same name using the tool
        fold_result2 = self.tool_call_add_fold(
            self.FILE_FOO, "tail", "fn lorem()", "MyFold"
        )

        # This should fail due to duplicate name
        self.assertIsInstance(fold_result2, dict)
        self.assertIn("error", fold_result2)
        self.assertIn("already exists", fold_result2["error"])

    def test_add_fold_nonexistent_pattern(self):
        """Test that adding a fold with a non-existent pattern fails."""
        self.init_test_llm()
        self.init_test_file()

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)

        # Try to add a fold with a pattern that doesn't exist
        fold_result = self.tool_call_add_fold(
            self.FILE_FOO, "head", "nonexistent_pattern_xyz", "MyFold"
        )

        # This should fail
        self.assertIsInstance(fold_result, dict)
        self.assertIn("error", fold_result)
        self.assertIn("No lines matching pattern", fold_result["error"])

    def test_add_fold_file_not_in_context(self):
        """Test that adding a fold to a file not in context fails."""
        # Use a file that is not in SUFFIX_CONTEXTS
        self.init_test_llm()
        self.init_test_file()

        # Read the file first to set up context
        self.tool_call_read_with_check(self.FILE_FOO, 0)

        nonexistent_path = Path("nonexistent_file.txt")
        fold_result = self.tool_call_add_fold(
            nonexistent_path, "head", "some_pattern", "MyFold"
        )

        # This should fail
        self.assertIsInstance(fold_result, dict)
        self.assertIn("error", fold_result)
        self.assertIn("not loaded into context", fold_result["error"])

    def test_unfold(self):
        """Test removing a specific fold by name."""
        _, messages = self.init_test_llm()
        self.init_test_file()

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)
        read_msg_idx = len(messages) - 1

        # Add two folds using the tool
        fold_result1 = self.tool_call_add_fold(
            self.FILE_FOO, "head", "struct FooBar(i32);", "HeadFold"
        )
        self.assertIn("Added fold 'HeadFold'", fold_result1)

        fold_result2 = self.tool_call_add_fold(
            self.FILE_FOO, "tail", "fn lorem()", "TailFold"
        )
        self.assertIn("Added fold 'TailFold'", fold_result2)

        # Unfold the tail fold using the tool
        unfold_result = self.tool_call_unfold(self.FILE_FOO, "TailFold")
        self.assertIn("Removed fold 'TailFold'", unfold_result)

        # Call epilogue to trigger prepare_current_llm which applies folding
        self.epilogue()

        # Find the read_file tool message and check the unfold result
        read_msg = messages[read_msg_idx]
        content = read_msg["content"]

        # Check that only head fold remains
        self.assertIn(">>> FOLD: HeadFold", content)
        self.assertNotIn(">>> FOLD: TailFold", content)
        # Tail content should be visible now
        self.assertIn("fn lorem() -> i32 {1}", content)

    def test_unfold_nonexistent_fold(self):
        """Test that unfolding a non-existent fold fails."""
        self.init_test_llm()
        self.init_test_file()

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)

        # Try to unfold without any folds
        unfold_result = self.tool_call_unfold(self.FILE_FOO, "NonExistentFold")

        # This should fail
        self.assertIsInstance(unfold_result, dict)
        self.assertIn("error", unfold_result)

    def test_unfold_all(self):
        """Test removing all folds from a file."""
        _, messages = self.init_test_llm()
        self.init_test_file()

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)
        read_msg_idx = len(messages) - 1

        # Add multiple folds using the tool
        fold_result1 = self.tool_call_add_fold(
            self.FILE_FOO, "head", "struct FooBar(i32);", "HeadFold"
        )
        self.assertIn("Added fold 'HeadFold'", fold_result1)

        fold_result2 = self.tool_call_add_fold(
            self.FILE_FOO, "tail", "fn lorem()", "TailFold"
        )
        self.assertIn("Added fold 'TailFold'", fold_result2)

        # Unfold all using the tool
        unfold_all_result = self.tool_call_unfold_all(self.FILE_FOO)
        self.assertIn("Removed all folds", unfold_all_result)

        # Call epilogue to trigger prepare_current_llm which applies folding
        self.epilogue()

        # Find the read_file tool message and check all folds are removed
        read_msg = messages[read_msg_idx]
        content = read_msg["content"]

        # Check that all folds are removed
        self.assertNotIn(">>> FOLD:", content)
        # All content should be visible now
        self.assertIn("struct FooBar(i32);", content)
        self.assertIn("fn lorem() -> i32 {1}", content)

    def test_unfold_all_no_folds(self):
        """Test that unfold_all on a file with no folds fails gracefully."""
        self.init_test_llm()
        self.init_test_file()

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)

        # Try to unfold all without any folds
        unfold_all_result = self.tool_call_unfold_all(self.FILE_FOO)

        # This should fail
        self.assertIsInstance(unfold_all_result, dict)
        self.assertIn("error", unfold_all_result)

    def test_has_folds(self):
        """Test checking if a file has folds."""
        self.init_test_llm()
        self.init_test_file()

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)

        handler = context_handler()

        # Initially no folds
        self.assertFalse(handler.has_folds(self.FILE_FOO.name))

        # Add a fold using the tool
        fold_result = self.tool_call_add_fold(
            self.FILE_FOO, "head", "struct FooBar(i32);", "MyFold"
        )
        self.assertIn("Added fold", fold_result)

        # Now has folds
        self.assertTrue(handler.has_folds(self.FILE_FOO.name))

        # Unfold all using the tool
        self.tool_call_unfold_all(self.FILE_FOO)

        # No more folds
        self.assertFalse(handler.has_folds(self.FILE_FOO.name))

    def test_format_folded_content_no_folds(self):
        """Test that content without folds is unchanged."""
        _, messages = self.init_test_llm()
        self.init_test_file()

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)
        read_msg_idx = len(messages) - 1

        # Call epilogue to trigger prepare_current_llm
        self.epilogue()

        # Find the read_file tool message
        read_msg = messages[read_msg_idx]
        content = read_msg["content"]

        # Should not have any fold markers
        self.assertNotIn(">>> FOLD:", content)
        # All content should be visible
        self.assertIn("struct FooBar(i32);", content)
        self.assertIn("fn lorem() -> i32 {1}", content)

    def test_fold_line_numbers(self):
        """Test that fold line numbers are correct."""
        _, messages = self.init_test_llm()
        # Create a file with exact line count
        test_content = """line1
line2
line3
line4
line5
line6
line7
line8
line9
line10
"""
        self.FILE_FOO.write_text(test_content)

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)
        read_msg_idx = len(messages) - 1

        # Add a head fold ending at line 3 using the tool
        fold_result = self.tool_call_add_fold(
            self.FILE_FOO, "head", "line3", "First three lines"
        )
        self.assertIn("lines 1..3", fold_result)

        # Add a tail fold starting at line 8 using the tool
        fold_result = self.tool_call_add_fold(
            self.FILE_FOO, "tail", "line8", "Last three lines"
        )
        self.assertIn("lines 8..10", fold_result)

        # Call epilogue to trigger prepare_current_llm which applies folding
        self.epilogue()

        # Find the read_file tool message and check fold markers
        read_msg = messages[read_msg_idx]
        content = read_msg["content"]

        # Check fold markers with correct line numbers
        self.assertIn(">>> FOLD: First three lines (lines 1..3)", content)
        self.assertIn(">>> FOLD: Last three lines (lines 8..10)", content)
        # Visible content
        for n in "4567":
            self.assertIn(f"line{n}", content)
        # Hidden test_add_fold_no_space_between_foldsent should not be visible
        for n in ["1", "3", "8", "10"]:
            self.assertNotIn(f"line{n}", content)

    def test_get_folds(self):
        """Test getting all folds for a file."""
        self.init_test_llm()
        self.init_test_file()

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)

        handler = context_handler()

        # Initially no folds
        folds = handler.get_folds(self.FILE_FOO.name)
        self.assertEqual(len(folds), 0)

        # Add folds using the tool
        self.tool_call_add_fold(
            self.FILE_FOO, "head", "struct FooBar(i32);", "HeadFold"
        )
        self.tool_call_add_fold(self.FILE_FOO, "tail", "fn lorem()", "TailFold")

        # Check folds
        folds = handler.get_folds(self.FILE_FOO.name)
        self.assertEqual(len(folds), 2)

        fold_names = [f.name for f in folds]
        self.assertIn("HeadFold", fold_names)
        self.assertIn("TailFold", fold_names)

    def test_edit_file_with_folds_visible_target(self):
        """Test editing a file with folds, editing visible content."""
        _, messages = self.init_test_llm()
        self.init_test_file()

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)
        read_msg_idx = len(messages) - 1

        # Add a head fold to hide the imports and struct definitions
        fold_result = self.tool_call_add_fold(
            self.FILE_FOO, "head", "struct FooBar(i32);", "Helper functions structs"
        )
        self.assertIn("Added fold 'Helper functions structs'", fold_result)

        # Add a tail fold to hide the helper functions
        fold_result = self.tool_call_add_fold(
            self.FILE_FOO, "tail", "fn lorem()", "Lorem/ipsum implementation"
        )
        self.assertIn("Added fold 'Lorem/ipsum implementation'", fold_result)

        # Epilogue to apply folds
        self.epilogue()

        # Now edit visible content - the main function
        edit_result = self.tool_call_edit_foo("fn main() {", "fn main(){")
        self.assertIsInstance(edit_result, str)
        self.assertIn("OK: edit", edit_result)

        # Check the file was actually edited
        actual_content = self.FILE_FOO.read_text()
        self.assertIn("fn main(){", actual_content)

        # Epilogue to update context
        self.epilogue()

        # Check the message shows the edit
        read_msg = messages[read_msg_idx]
        content = read_msg["content"]
        self.assertIn("fn main(){", content)

    def test_edit_file_with_folds_hidden_target(self):
        """Test editing a file with folds, trying to edit hidden content should fail."""
        self.init_test_llm()
        self.init_test_file()

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)

        # Add a head fold to hide the imports and struct definitions
        fold_result = self.tool_call_add_fold(
            self.FILE_FOO, "head", "struct FooBar(i32);", "Helper functions structs"
        )
        self.assertIn("Added fold 'Helper functions structs'", fold_result)

        # Epilogue to apply folds
        self.epilogue()

        # Try to edit hidden content - should fail
        edit_result = self.tool_call_edit_foo("struct FooBar(i32);", "struct FooBar2(i32);")

        # This should fail because the target is hidden
        self.assertIsInstance(edit_result, dict)
        self.assertIn("error", edit_result)
        self.assertIn("folded", edit_result["error"])

    def test_edit_file_with_folds_unique_in_visible(self):
        """Test editing a file with folds where target is unique in visible content."""
        _, messages = self.init_test_llm()
        # Create a file with duplicate content
        test_content = """fn foo() -> i32 {
    return 1;
}
fn bar() -> i32 {
    return 1;
}
fn foobar() -> i32 {
    return 1;
}
"""
        self.FILE_FOO.write_text(test_content)

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)
        read_msg_idx = len(messages) - 1

        # Fold foo function (head fold to the closing brace on line 3)
        fold_result1 = self.tool_call_add_fold(
            self.FILE_FOO, "head", "}", "foo function"
        )
        self.assertIn("Added fold 'foo function'", fold_result1)

        # Fold foobar function (tail fold from line 7 to end)
        fold_result2 = self.tool_call_add_fold(
            self.FILE_FOO, "tail", "fn foobar()", "foobar function"
        )
        self.assertIn("Added fold 'foobar function'", fold_result2)

        # Epilogue to apply folds
        self.epilogue()

        # Now "return 1;" is unique in visible content (only in bar function)
        # So this edit should succeed
        edit_result = self.tool_call_edit_foo("    return 1;", "    return 2;")
        self.assertIsInstance(edit_result, str)
        self.assertIn("OK: edit", edit_result)

        # Check the file was edited
        actual_content = self.FILE_FOO.read_text()
        lines = actual_content.splitlines()
        # The bar function should have "return 2;"
        self.assertIn("    return 2;", actual_content)

    def test_edit_file_with_folds_not_unique_in_visible(self):
        """Test editing a file with folds where target is not unique in visible content."""
        _, messages = self.init_test_llm()
        # Create a file with duplicate content in visible section
        test_content = """fn foo() -> i32 {
    return 1;
}
fn bar() -> i32 {
    return 1;
}
fn foobar() -> i32 {
    return 1;
}
"""
        self.FILE_FOO.write_text(test_content)

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)

        # Fold only foobar function
        fold_result = self.tool_call_add_fold(
            self.FILE_FOO, "tail", "fn foobar()", "foobar function"
        )
        self.assertIn("Added fold 'foobar function'", fold_result)

        # Epilogue to apply folds
        self.epilogue()

        # Now "return 1;" appears in both foo and bar (visible), so edit should fail
        edit_result = self.tool_call_edit_foo("    return 1;", "    return 2;")

        # This should fail because target is not unique in visible content
        self.assertIsInstance(edit_result, dict)
        self.assertIn("error", edit_result)
        self.assertIn("must be unique", edit_result["error"])

    def test_edit_file_with_folds_updates_line_numbers(self):
        """Test that fold line numbers are updated after editing."""
        _, messages = self.init_test_llm()
        # Create a simple file
        test_content = """line1
line2
line3
line4
line5
line6
line7
line8
line9
line10
"""
        self.FILE_FOO.write_text(test_content)

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)
        read_msg_idx = len(messages) - 1

        # Add a tail fold
        fold_result = self.tool_call_add_fold(
            self.FILE_FOO, "tail", "line8", "Bottom lines"
        )
        self.assertIn("lines 8..10", fold_result)

        # Epilogue to apply folds
        self.epilogue()

        # Edit visible content, adding a line
        edit_result = self.tool_call_edit_foo("line5", "line5a\nline5b")
        self.assertIsInstance(edit_result, str)
        self.assertIn("OK: edit", edit_result)

        # Epilogue to update context
        self.epilogue()

        # Check the fold line numbers were updated
        handler = context_handler()
        folds = handler.get_folds(self.FILE_FOO.name)
        self.assertEqual(len(folds), 1)
        # Tail fold should now start at line 9 (was 8, +1 for added line)
        self.assertEqual(folds[0].start_line, 9)
        self.assertEqual(folds[0].end_line, 11)

        # Check the message shows updated fold markers
        read_msg = messages[read_msg_idx]
        content = read_msg["content"]
        self.assertIn(">>> FOLD: Bottom lines (lines 9..11)", content)

    def test_edit_file_with_folds_deletes_line(self):
        """Test that fold line numbers are updated when deleting a line."""
        _, messages = self.init_test_llm()
        # Create a simple file
        test_content = """line1
line2
line3
line4
line5
line6
line7
line8
line9
line10
"""
        self.FILE_FOO.write_text(test_content)

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)
        read_msg_idx = len(messages) - 1

        # Add a tail fold
        fold_result = self.tool_call_add_fold(
            self.FILE_FOO, "tail", "line8", "Bottom lines"
        )
        self.assertIn("lines 8..10", fold_result)

        # Epilogue to apply folds
        self.epilogue()

        # Edit visible content, deleting a line
        edit_result = self.tool_call_edit_foo("line5\nline6", "line5")
        self.assertIsInstance(edit_result, str)
        self.assertIn("OK: edit", edit_result)

        # Epilogue to update context
        self.epilogue()

        # Check the fold line numbers were updated
        handler = context_handler()
        folds = handler.get_folds(self.FILE_FOO.name)
        self.assertEqual(len(folds), 1)
        # Tail fold should now start at line 7 (was 8, -1 for deleted line)
        self.assertEqual(folds[0].start_line, 7)
        self.assertEqual(folds[0].end_line, 9)

        # Check the message shows updated fold markers
        read_msg = messages[read_msg_idx]
        content = read_msg["content"]
        self.assertIn(">>> FOLD: Bottom lines (lines 7..9)", content)

    def test_edit_file_no_folds_still_works(self):
        """Test that editing without folds still works as before."""
        _, messages = self.init_test_llm()
        test_content = """fn foo() {
    return 1;
}
"""
        self.FILE_FOO.write_text(test_content)

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)

        # Edit without any folds
        edit_result = self.tool_call_edit_foo("return 1;", "return 2;")
        self.assertIsInstance(edit_result, str)
        self.assertIn("OK: edit", edit_result)

        # Check the file was edited
        actual_content = self.FILE_FOO.read_text()
        self.assertIn("return 2;", actual_content)

    def test_edit_file_no_folds_not_unique_fails(self):
        """Test that editing without folds still requires unique target."""
        _, messages = self.init_test_llm()
        test_content = """fn foo() {
    return 1;
}
fn bar() {
    return 1;
}
"""
        self.FILE_FOO.write_text(test_content)

        # Read the file first
        self.tool_call_read_with_check(self.FILE_FOO, 0)

        # Try to edit non-unique content
        edit_result = self.tool_call_edit_foo("return 1;", "return 2;")

        # This should fail
        self.assertIsInstance(edit_result, dict)
        self.assertIn("error", edit_result)
        self.assertIn("must exists exactly once", edit_result["error"])
