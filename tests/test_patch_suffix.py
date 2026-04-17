import threading
from tests.test_helper import TestBase
import tool_suffix_patch
import sys


class TestPatchSuffix(TestBase):
    def setUp(self):
        super().setUp()
        # Setup timeout thread for safety
        self.timeout_thread = threading.Thread(
            target=self._timeout_handler, daemon=True
        )
        self.timeout_abort = threading.Event()
        self.timeout_thread.start()

    def _timeout_handler(self):
        """Wait 0.3 seconds and quit with error if not aborted."""
        self.timeout_abort.wait(0.3)
        if self.timeout_abort.is_set():
            return
        print("ERROR: Test took too long (0.3s timeout)")
        sys.exit(1)

    def tearDown(self):
        # Signal to abort timeout thread
        self.timeout_abort.set()
        self.timeout_thread.join(timeout=1.0)
        super().tearDown()

    def test_basic_remove(self):
        """Test basic line removal with <<remove>> suffix."""
        self.FILE_FOO.write_text("line1\nline2\nline3\n")
        patch = """<<context>>
|line1
|line2<<remove>>
|line3
"""
        result = tool_suffix_patch.ToolPatchSuffix()(self.FILE_FOO.name, patch)
        assert isinstance(result, str)
        self.assertIn("FIRST WRITTEN LINE", result)
        content = self.FILE_FOO.read_text()
        self.assertEqual(content, "line1\nline3\n")

    def test_basic_add(self):
        """Test basic line addition with <<add>> section."""
        self.FILE_FOO.write_text("line1\nline3\n")
        patch = """<<context>>
|line1<<add>>
|line3
<<add>>
|line2
"""
        result = tool_suffix_patch.ToolPatchSuffix()(self.FILE_FOO.name, patch)
        assert isinstance(result, str)
        self.assertIn("FIRST WRITTEN LINE", result)
        content = self.FILE_FOO.read_text()
        self.assertEqual(content, "line1\nline2\nline3\n")

    def test_remove_add(self):
        """Test remove and add in one operation."""
        self.FILE_FOO.write_text("line1\nline2\nline3\n")
        patch = """<<context>>
|line1
|line2<<remove>><<add>>
|line3
<<add>>
|line2_new
"""
        result = tool_suffix_patch.ToolPatchSuffix()(self.FILE_FOO.name, patch)
        assert isinstance(result, str)
        assert "FIRST WRITTEN LINE" in result
        content = self.FILE_FOO.read_text()
        self.assertEqual(content, "line1\nline2_new\nline3\n")

    def test_example_from_spec(self):
        """Test the exact example from the specification."""
        original = """    # run
    lang = config.guess_project_language()
    print(f"{lang=}")
    node = AgencyNode(lang=lang)
    node.simple(prompt)
    if node.lang == "rust":
        tool_sh.rustfmt()
    end = time.monotonic()
"""
        self.FILE_FOO.write_text(original)
        patch = """<<context>>
|    # run<<remove>>
|    lang = config.guess_project_language()<<remove>><<add>>
|    print(f"{lang=}")
|    node = AgencyNode(lang=lang)
|    node.simple(prompt)<<add>>
|    if node.lang == "rust":<<remove>>
|        tool_sh.rustfmt()<<remove>>
|    end = time.monotonic()
<<add>>
|    lang_level = config.guess_project_language_level()
|    lang = config.guess_project_language(lang_level)
<<add>>
|    node.recursive()
|
"""
        result = tool_suffix_patch.ToolPatchSuffix()(self.FILE_FOO.name, patch)
        assert isinstance(result, str)
        self.assertIn("FIRST WRITTEN LINE", result)
        content = self.FILE_FOO.read_text()
        expected = """    lang_level = config.guess_project_language_level()
    lang = config.guess_project_language(lang_level)
    print(f"{lang=}")
    node = AgencyNode(lang=lang)
    node.simple(prompt)
    node.recursive()

    end = time.monotonic()
"""
        self.assertEqual(content, expected)

    def test_multiple_add_sections(self):
        """Test multiple <<add>> sections."""
        self.FILE_FOO.write_text("a\nb\nc\n")
        patch = """<<context>>
|a<<add>>
|b<<add>>
|c
<<add>>
|a_new
<<add>>
|b_new
"""
        result = tool_suffix_patch.ToolPatchSuffix()(self.FILE_FOO.name, patch)
        assert isinstance(result, str)
        self.assertIn("FIRST WRITTEN LINE", result)
        content = self.FILE_FOO.read_text()
        self.assertEqual(content, "a\na_new\nb\nb_new\nc\n")

    def test_empty_file(self):
        """Test patching empty file."""
        self.FILE_FOO.write_text("")
        # For empty file, context must be empty or add section only
        patch = """<<add>>
|new content
"""
        result = tool_suffix_patch.ToolPatchSuffix()(self.FILE_FOO.name, patch)
        assert isinstance(result, str)
        self.assertIn("FIRST WRITTEN LINE", result)
        content = self.FILE_FOO.read_text()
        self.assertEqual(content, "new content\n")

    def test_context_not_found(self):
        """Test error when context not found."""
        self.FILE_FOO.write_text("line1\nline2\nline3\n")
        patch = """<<context>>
|not_found
"""
        result = tool_suffix_patch.ToolPatchSuffix()(self.FILE_FOO.name, patch)
        assert isinstance(result, dict)
        self.assertIn("error", result)

    def test_missing_add_section(self):
        """Test error when <<add>> suffix but no add section."""
        self.FILE_FOO.write_text("line1\nline2\n")
        patch = """<<context>>
|line1<<add>>
|line2
"""
        result = tool_suffix_patch.ToolPatchSuffix()(self.FILE_FOO.name, patch)
        assert isinstance(result, dict)
        self.assertIn("error", result)

    def test_unconsumed_add_section(self):
        """Test error when <<add>> section is not consumed."""
        self.FILE_FOO.write_text("line1\nline2\n")
        patch = """<<context>>
|line1
|line2
<<add>>
|new_line
"""
        result = tool_suffix_patch.ToolPatchSuffix()(self.FILE_FOO.name, patch)
        assert isinstance(result, dict)
        self.assertIn("error", result)

    def test_safety_check_suffix_in_file(self):
        """Test that file with forbidden suffixes is rejected."""
        self.FILE_FOO.write_text("line1<<remove>>\nline2\n")
        patch = """<<context>>
|line1
|line2
"""
        result = tool_suffix_patch.ToolPatchSuffix()(self.FILE_FOO.name, patch)
        assert isinstance(result, dict)
        self.assertIn("error", result)

    def test_custom_suffixes(self):
        """Test with custom suffix values."""
        self.FILE_FOO.write_text("line1\nline2\nline3\n")
        tool = tool_suffix_patch.ToolPatchSuffix(
            sfx_remove="<<DEL>>", sfx_add="<<INS>>"
        )
        patch = """<<context>>
|line1
|line2<<DEL>>
|line3
"""
        result = tool(self.FILE_FOO.name, patch)
        assert isinstance(result, str)
        self.assertIn("FIRST WRITTEN LINE", result)
        content = self.FILE_FOO.read_text()
        self.assertEqual(content, "line1\nline3\n")

    def test_keep_only(self):
        """Test patch with only keep operations (no changes)."""
        self.FILE_FOO.write_text("line1\nline2\n")
        patch = """<<context>>
|line1
|line2
"""
        result = tool_suffix_patch.ToolPatchSuffix()(self.FILE_FOO.name, patch)
        assert isinstance(result, str)
        self.assertIn("FIRST WRITTEN LINE", result)
        content = self.FILE_FOO.read_text()
        self.assertEqual(content, "line1\nline2\n")

    def test_context_in_middle(self):
        """Test context found in the middle of file."""
        self.FILE_FOO.write_text("before\nline1\nline2\nline3\nafter\n")
        patch = """<<context>>
|line1
|line2<<remove>>
|line3
"""
        result = tool_suffix_patch.ToolPatchSuffix()(self.FILE_FOO.name, patch)
        assert isinstance(result, str)
        self.assertIn("FIRST WRITTEN LINE", result)
        content = self.FILE_FOO.read_text()
        self.assertEqual(content, "before\nline1\nline3\nafter\n")

    def test_multiline_add_section(self):
        """Test <<add>> section with multiple lines."""
        self.FILE_FOO.write_text("line1\nline3\n")
        patch = """<<context>>
|line1<<add>>
|line3
<<add>>
|line2a
|line2b
|line2c
"""
        result = tool_suffix_patch.ToolPatchSuffix()(self.FILE_FOO.name, patch)
        assert isinstance(result, str)
        self.assertIn("FIRST WRITTEN LINE", result)
        content = self.FILE_FOO.read_text()
        self.assertEqual(content, "line1\nline2a\nline2b\nline2c\nline3\n")

    def test_empty_add_section(self):
        """Test <<add>> section with empty content."""
        self.FILE_FOO.write_text("line1\nline2\n")
        patch = """<<context>>
|line1<<add>>
|line2
<<add>>
|
"""
        result = tool_suffix_patch.ToolPatchSuffix()(self.FILE_FOO.name, patch)
        assert isinstance(result, str)
        self.assertIn("FIRST WRITTEN LINE", result)
        content = self.FILE_FOO.read_text()
        self.assertEqual(content, "line1\n\nline2\n")

    def test_no_trailing_newline_preserved(self):
        """Test that lack of trailing newline is preserved."""
        self.FILE_FOO.write_text("line1\nline2")
        patch = """<<context>>
|line1
|line2
"""
        result = tool_suffix_patch.ToolPatchSuffix()(self.FILE_FOO.name, patch)
        assert isinstance(result, str)
        self.assertIn("FIRST WRITTEN LINE", result)
        content = self.FILE_FOO.read_text()
        self.assertEqual(content, "line1\nline2")

    def test_complex_mixed_operations(self):
        """Test complex patch with mixed operations."""
        self.FILE_FOO.write_text("a\nb\nc\nd\ne\n")
        patch = """<<context>>
|a
|b<<remove>>
|c<<add>>
|d<<remove>><<add>>
|e
<<add>>
|c_new
<<add>>
|d_new
"""
        result = tool_suffix_patch.ToolPatchSuffix()(self.FILE_FOO.name, patch)
        assert isinstance(result, str)
        self.assertIn("FIRST WRITTEN LINE", result)
        content = self.FILE_FOO.read_text()
        self.assertEqual(content, "a\nc\nc_new\nd_new\ne\n")
