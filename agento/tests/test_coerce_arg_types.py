"""Tests for coerce_arg_types function in llm_fix.py"""

import unittest
from typing import Annotated
from dataclasses import dataclass

from agento.tool import Tool
from agento.llm_fix import coerce_arg_types
from agento.tests.test_helper import TestBase


@dataclass
class TestToolWithInt(Tool):
    """Test tool with integer parameter."""

    def __init__(self):
        super().__init__("test_int", "Test tool with int param")

    def __call__(self, num: Annotated[int, "An integer"]):
        return f"Got int: {num}"


@dataclass
class TestToolWithFloat(Tool):
    """Test tool with float/number parameter."""

    def __init__(self):
        super().__init__("test_float", "Test tool with float param")

    def __call__(self, num: Annotated[float, "A number"]):
        return f"Got float: {num}"


@dataclass
class TestToolWithBool(Tool):
    """Test tool with boolean parameter."""

    def __init__(self):
        super().__init__("test_bool", "Test tool with bool param")

    def __call__(self, flag: Annotated[bool, "A boolean flag"]):
        return f"Got bool: {flag}"


@dataclass
class TestToolWithArray(Tool):
    """Test tool with array parameter."""

    def __init__(self):
        super().__init__("test_array", "Test tool with array param")

    def __call__(self, items: Annotated[list, "An array of strings"]):
        return f"Got array: {items}"


@dataclass
class TestToolWithMixedTypes(Tool):
    """Test tool with multiple parameter types."""

    def __init__(self):
        super().__init__("test_mixed", "Test tool with mixed params")

    def __call__(
        self,
        count: Annotated[int, "An integer"],
        ratio: Annotated[float, "A float"],
        enabled: Annotated[bool, "A boolean"],
        items: Annotated[list, "An array"],
        name: Annotated[str, "A string"],
    ):
        return f"Got: count={count}, ratio={ratio}, enabled={enabled}, items={items}, name={name}"


class TestCoerceArgTypes(TestBase):
    """Tests for coerce_arg_types function."""

    def test_coerce_string_to_integer(self):
        """Test that string values are coerced to integers when expected."""
        tool = TestToolWithInt()
        args = {"num": "42"}
        result = coerce_arg_types(args, tool)
        assert isinstance(
            result["num"], int
        ), f"Expected int, got {type(result['num'])}"
        self.assertEqual(result["num"], 42)

    def test_coerce_string_to_negative_integer(self):
        """Test that negative string values are coerced to integers."""
        tool = TestToolWithInt()
        args = {"num": "-123"}
        result = coerce_arg_types(args, tool)
        assert isinstance(result["num"], int)
        self.assertEqual(result["num"], -123)

    def test_coerce_string_to_float(self):
        """Test that string values are coerced to floats when expected."""
        tool = TestToolWithFloat()
        args = {"num": "3.14"}
        result = coerce_arg_types(args, tool)
        assert isinstance(
            result["num"], float
        ), f"Expected float, got {type(result['num'])}"
        self.assertAlmostEqual(result["num"], 3.14, places=2)

    def test_coerce_string_to_integer_as_float(self):
        """Test that integer string is coerced to float when float expected."""
        tool = TestToolWithFloat()
        args = {"num": "42"}
        result = coerce_arg_types(args, tool)
        assert isinstance(result["num"], float)
        self.assertEqual(result["num"], 42.0)

    def test_coerce_string_to_boolean_true(self):
        """Test that 'true' string is coerced to boolean True."""
        tool = TestToolWithBool()
        args = {"flag": "true"}
        result = coerce_arg_types(args, tool)
        assert isinstance(result["flag"], bool)
        self.assertEqual(result["flag"], True)

    def test_coerce_string_to_boolean_true_uppercase(self):
        """Test that 'TRUE' string is coerced to boolean True."""
        tool = TestToolWithBool()
        args = {"flag": "TRUE"}
        result = coerce_arg_types(args, tool)
        assert isinstance(result["flag"], bool)
        self.assertEqual(result["flag"], True)

    def test_coerce_string_to_boolean_one(self):
        """Test that '1' string is coerced to boolean True."""
        tool = TestToolWithBool()
        args = {"flag": "1"}
        result = coerce_arg_types(args, tool)
        assert isinstance(result["flag"], bool)
        self.assertEqual(result["flag"], True)

    def test_coerce_string_to_boolean_yes(self):
        """Test that 'yes' string is coerced to boolean True."""
        tool = TestToolWithBool()
        args = {"flag": "yes"}
        result = coerce_arg_types(args, tool)
        assert isinstance(result["flag"], bool)
        self.assertEqual(result["flag"], True)

    def test_coerce_string_to_boolean_false(self):
        """Test that 'false' string is coerced to boolean False."""
        tool = TestToolWithBool()
        args = {"flag": "false"}
        result = coerce_arg_types(args, tool)
        assert isinstance(result["flag"], bool)
        self.assertEqual(result["flag"], False)

    def test_coerce_string_to_array(self):
        """Test that JSON array string is coerced to list."""
        tool = TestToolWithArray()
        args = {"items": '["a", "b", "c"]'}
        result = coerce_arg_types(args, tool)
        assert isinstance(result["items"], list)
        self.assertEqual(result["items"], ["a", "b", "c"])

    def test_coerce_string_to_array_empty(self):
        """Test that empty JSON array string is coerced to empty list."""
        tool = TestToolWithArray()
        args = {"items": "[]"}
        result = coerce_arg_types(args, tool)
        assert isinstance(result["items"], list)
        self.assertEqual(result["items"], [])

    def test_coerce_mixed_types(self):
        """Test coercion of multiple parameter types at once."""
        tool = TestToolWithMixedTypes()
        args = {
            "count": "42",
            "ratio": "3.14",
            "enabled": "true",
            "items": '["x", "y"]',
            "name": "test",
        }
        result = coerce_arg_types(args, tool)
        assert isinstance(result["count"], int)
        assert isinstance(result["ratio"], float)
        assert isinstance(result["enabled"], bool)
        assert isinstance(result["items"], list)
        assert isinstance(result["name"], str)
        self.assertEqual(result["count"], 42)
        self.assertAlmostEqual(result["ratio"], 3.14, places=2)
        self.assertEqual(result["enabled"], True)
        self.assertEqual(result["items"], ["x", "y"])
        self.assertEqual(result["name"], "test")

    def test_no_coercion_when_already_correct_type(self):
        """Test that values already of correct type are not modified."""
        tool = TestToolWithInt()
        args = {"num": 42}  # Already an int
        result = coerce_arg_types(args, tool)
        self.assertEqual(result["num"], 42)
        assert isinstance(result["num"], int)

    def test_no_coercion_when_invalid_integer(self):
        """Test that invalid integer strings are not coerced."""
        tool = TestToolWithInt()
        args = {"num": "not_a_number"}
        result = coerce_arg_types(args, tool)
        # Should remain as string since it can't be converted
        self.assertEqual(result["num"], "not_a_number")
        assert isinstance(result["num"], str)

    def test_no_coercion_when_invalid_boolean(self):
        """Test that invalid boolean strings remain as strings."""
        tool = TestToolWithBool()
        args = {"flag": "maybe"}
        result = coerce_arg_types(args, tool)
        self.assertEqual(result["flag"], "maybe")
        assert isinstance(result["flag"], str)

    def test_no_coercion_when_invalid_array(self):
        """Test that invalid JSON strings remain as strings."""
        tool = TestToolWithArray()
        args = {"items": "not json"}
        result = coerce_arg_types(args, tool)
        self.assertEqual(result["items"], "not json")
        assert isinstance(result["items"], str)

    def test_tool_with_no_parameters(self):
        """Test that tools with no parameters work correctly."""

        @dataclass
        class NoParamTool(Tool):
            def __init__(self):
                super().__init__("no_params", "Tool with no params")

            def __call__(self):
                return "done"

        tool = NoParamTool()
        args = {}
        result = coerce_arg_types(args, tool)
        self.assertEqual(result, {})

    def test_extra_args_not_in_tool_schema(self):
        """Test that extra arguments not in tool schema are passed through unchanged."""
        tool = TestToolWithInt()
        args = {"num": "42", "extra": "should_remain"}
        result = coerce_arg_types(args, tool)
        self.assertEqual(result["num"], 42)
        self.assertEqual(result["extra"], "should_remain")
        assert isinstance(result["extra"], str)

    def test_original_args_not_modified(self):
        """Test that original args dict is not modified."""
        tool = TestToolWithInt()
        args = {"num": "42"}
        original_num = args["num"]
        result = coerce_arg_types(args, tool)
        # Original should still be string
        self.assertEqual(args["num"], "42")
        # Result should be int
        self.assertEqual(result["num"], 42)


if __name__ == "__main__":
    unittest.main()
