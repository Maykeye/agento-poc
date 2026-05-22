"""Tests for SQL utilities (utilsql)."""

import unittest
import sqlite3
from tests.test_helper import TestBase
import utilsql


class TestUtilSQL(TestBase):
    """Test cases for SQL utilities."""

    def test_generation_history_has_tools_id_column(self):
        """Test that generation_history table has tools_id column."""
        with utilsql.sql_db() as db:
            # Get table info for generation_history
            result = db.execute(
                "PRAGMA table_info(generation_history)"
            ).fetchall()
            column_names = [row[1] for row in result]
            self.assertIn("tools_id", column_names)

    def test_llm_tools_table_exists(self):
        """Test that llm_tools table exists (renamed from llm_tools_info)."""
        with utilsql.sql_db() as db:
            # Check if llm_tools table exists
            result = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='llm_tools'"
            ).fetchall()
            self.assertEqual(len(result), 1)

    def test_llm_tools_info_table_does_not_exist(self):
        """Test that llm_tools_info table does not exist (was renamed to llm_tools)."""
        with utilsql.sql_db() as db:
            # Check that old table name does not exist
            result = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='llm_tools_info'"
            ).fetchall()
            self.assertEqual(len(result), 0)

    def test_llm_tools_table_has_text_column(self):
        """Test that llm_tools table has text column with unique constraint."""
        with utilsql.sql_db() as db:
            result = db.execute(
                "PRAGMA table_info(llm_tools)"
            ).fetchall()
            columns = {row[1]: row for row in result}
            self.assertIn("text", columns)
            # text column should be NOT NULL
            self.assertEqual(columns["text"][3], 1)  # notnull flag

    def test_log_tools_returns_id(self):
        """Test that log_tools returns a valid tools_id."""
        tools = {"tool1": {"name": "Tool 1"}, "tool2": {"name": "Tool 2"}}
        with utilsql.sql_db() as db:
            tools_id = utilsql.log_tools(db, tools)
            self.assertIsInstance(tools_id, int)
            self.assertGreater(tools_id, 0)

    def test_log_tools_idempotent(self):
        """Test that logging same tools returns same id."""
        tools = {"tool1": {"name": "Tool 1"}, "tool2": {"name": "Tool 2"}}
        with utilsql.sql_db() as db:
            tools_id_1 = utilsql.log_tools(db, tools)
            tools_id_2 = utilsql.log_tools(db, tools)
            self.assertEqual(tools_id_1, tools_id_2)

    def test_log_tools_different_tools_different_id(self):
        """Test that logging different tools returns different id."""
        tools1 = {"tool1": {"name": "Tool 1"}}
        tools2 = {"tool2": {"name": "Tool 2"}}
        with utilsql.sql_db() as db:
            tools_id_1 = utilsql.log_tools(db, tools1)
            tools_id_2 = utilsql.log_tools(db, tools2)
            self.assertNotEqual(tools_id_1, tools_id_2)

    def test_log_generation_stores_tools_id(self):
        """Test that log_generation stores tools_id in generation_history."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = {"read_file": {"name": "ReadFile"}}

        num = utilsql.log_generation(1, 1, messages, tools)

        with utilsql.sql_db() as db:
            result = db.execute(
                "SELECT tools_id FROM generation_history WHERE num = ?", (num,)
            ).fetchone()
            self.assertIsNotNone(result)
            self.assertIsNotNone(result[0])  # tools_id should not be None
            self.assertIsInstance(result[0], int)

    def test_log_generation_tools_logged_first(self):
        """Test that tools are logged before generation_history entry."""
        messages = [{"role": "user", "content": "Hello"}]
        tools = {"special_tool": {"name": "SpecialTool"}}

        num = utilsql.log_generation(1, 1, messages, tools)

        with utilsql.sql_db() as db:
            # Get the tools_id from generation_history
            result = db.execute(
                "SELECT tools_id FROM generation_history WHERE num = ?", (num,)
            ).fetchone()
            tools_id = result[0]

            # Verify the tools entry exists in llm_tools table
            tools_result = db.execute(
                "SELECT text FROM llm_tools WHERE id = ?", (tools_id,)
            ).fetchone()
            self.assertIsNotNone(tools_result)
            self.assertIn("special_tool", tools_result[0])

    def test_no_llm_tools_linking_table(self):
        """Test that the old llm_tools linking table does not exist."""
        with utilsql.sql_db() as db:
            # Check that the old llm_tools table (linking generations to tools) does not exist
            result = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='llm_tools' "
                "AND sql LIKE '%generation_id%'"
            ).fetchall()
            # The current llm_tools table should not have generation_id column
            self.assertEqual(len(result), 0)


if __name__ == "__main__":
    unittest.main()
