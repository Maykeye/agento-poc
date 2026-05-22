import unittest
import sqlite3
from pathlib import Path

from config import CONFIG
from tool import Tool, ToolCall
from tool import io as tool_io
import utilsql
from tests.test_helper import TestBase


class TestToolErrorLogging(TestBase):
    """Tests for tool error logging functionality."""

    def test_tool_error_table_exists(self):
        """Test that tool_error table is created."""
        with utilsql.sql_db() as db:
            result = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='tool_error'"
            ).fetchone()
            self.assertIsNotNone(result)
            self.assertEqual(result[0], "tool_error")

    def test_tool_error_log_non_existing_function(self):
        """Test logging when a non-existing function is called."""
        result = utilsql.log_tool_error(
            llm_id=1,
            reason="non-existing function",
            name_tag="q1",
            tool_list="['read_file', 'write_file']",
            function_name="non_existing_tool",
            function_args='{"path": "test.txt"}',
        )
        self.assertIsNotNone(result)

        with utilsql.sql_db() as db:
            row = db.execute(
                "SELECT reason, name_tag, tool_list, function_name, function_args "
                "FROM tool_error WHERE id=?",
                (result,),
            ).fetchone()
            self.assertEqual(row[0], "non-existing function")
            self.assertEqual(row[1], "q1")
            self.assertIn("read_file", row[2])
            self.assertEqual(row[3], "non_existing_tool")
            self.assertIn("path", row[4])

    def test_tool_error_log_execution_error(self):
        """Test logging when tool execution fails."""
        result = utilsql.log_tool_error(
            llm_id=2,
            reason="execution error",
            name_tag="q2",
            tool_list="['edit_file']",
            function_name="edit_file",
            function_args='{"path": "test.txt", "replace_from": "foo"}',
            exception_traceback="Traceback (most recent call last):\n  File 'test.py', line 1\n    raise ValueError('test error')\nValueError: test error",
        )
        self.assertIsNotNone(result)

        with utilsql.sql_db() as db:
            row = db.execute(
                "SELECT reason, exception_traceback FROM tool_error WHERE id=?",
                (result,),
            ).fetchone()
            self.assertEqual(row[0], "execution error")
            self.assertIn("ValueError", row[1])

    def test_tool_error_log_without_traceback(self):
        """Test logging error without traceback (non-existing function)."""
        result = utilsql.log_tool_error(
            llm_id=3,
            reason="non-existing function",
            name_tag="q3",
            tool_list="['read_file']",
            function_name="missing_tool",
        )
        self.assertIsNotNone(result)

        with utilsql.sql_db() as db:
            row = db.execute(
                "SELECT reason, function_name, exception_traceback FROM tool_error WHERE id=?",
                (result,),
            ).fetchone()
            self.assertEqual(row[0], "non-existing function")
            self.assertEqual(row[1], "missing_tool")
            # traceback should be empty or minimal when no exception occurred
            self.assertIsNotNone(row[2])

    def test_tool_error_llm_index(self):
        """Test that tool_error table has llm_id index."""
        with utilsql.sql_db() as db:
            result = db.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name='tool_error_llm_idx'"
            ).fetchone()
            self.assertIsNotNone(result)

    def test_tool_error_query_by_llm_id(self):
        """Test querying tool errors by llm_id."""
        # Insert multiple errors with different llm_ids
        utilsql.log_tool_error(
            llm_id=10,
            reason="non-existing function",
            name_tag="q10",
            tool_list="['read_file']",
            function_name="bad_tool",
        )
        utilsql.log_tool_error(
            llm_id=11,
            reason="execution error",
            name_tag="q11",
            tool_list="['write_file']",
            function_name="write_file",
            exception_traceback="Some error",
        )

        with utilsql.sql_db() as db:
            # Query for llm_id=10
            rows = db.execute(
                "SELECT reason, name_tag FROM tool_error WHERE llm_id=?", (10,)
            ).fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], "non-existing function")
            self.assertEqual(rows[0][1], "q10")

            # Query for llm_id=11
            rows = db.execute(
                "SELECT reason, name_tag FROM tool_error WHERE llm_id=?", (11,)
            ).fetchall()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0][0], "execution error")
            self.assertEqual(rows[0][1], "q11")

    def test_tool_error_preserves_all_fields(self):
        """Test that all fields are preserved correctly."""
        tool_list_str = "['tool1', 'tool2', 'tool3']"
        function_args = '{"path": "/some/path", "option": true}'

        result = utilsql.log_tool_error(
            llm_id=99,
            reason="execution error",
            name_tag="test_tag",
            tool_list=tool_list_str,
            function_name="complex_tool",
            function_args=function_args,
            exception_traceback="File 'x.py', line 5\nException: something bad",
        )

        with utilsql.sql_db() as db:
            row = db.execute(
                "SELECT llm_id, reason, name_tag, tool_list, function_name, "
                "function_args, exception_traceback, created_at "
                "FROM tool_error WHERE id=?",
                (result,),
            ).fetchone()
            self.assertEqual(row[0], 99)
            self.assertEqual(row[1], "execution error")
            self.assertEqual(row[2], "test_tag")
            self.assertEqual(row[3], tool_list_str)
            self.assertEqual(row[4], "complex_tool")
            self.assertEqual(row[5], function_args)
            self.assertIn("Exception", row[6])
            self.assertIsNotNone(row[7])  # created_at


if __name__ == "__main__":
    unittest.main()
