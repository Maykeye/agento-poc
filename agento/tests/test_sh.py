import unittest
from agento.tool.sh import ToolBash


class TestBash(unittest.TestCase):
    def test_bash_echo(self):
        """Test basic echo command"""
        tool = ToolBash()
        result = tool(command="echo hello world")
        self.assertEqual(result["exitcode"], 0)
        self.assertIn("hello world", result["stdout"])

    def test_bash_pipe_tail(self):
        """Test pipe with tail command"""
        tool = ToolBash()
        # Create a sequence of numbers and get last 2
        result = tool(command="seq 1 10 | tail -n2")
        self.assertEqual(result["exitcode"], 0)
        self.assertIn("9\n10", result["stdout"])

    def test_bash_ls(self):
        """Test ls command"""
        tool = ToolBash()
        result = tool(command="ls -la")
        self.assertEqual(result["exitcode"], 0)
        self.assertIn("total", result["stdout"])

    def test_bash_command_not_found(self):
        """Test command that doesn't exist"""
        tool = ToolBash()
        result = tool(command="nonexistent_command_12345")
        self.assertNotEqual(result["exitcode"], 0)

    def test_bash_grep(self):
        """Test grep command"""
        tool = ToolBash()
        result = tool(command="echo -e 'foo\\nbar\\nbaz' | grep bar")
        self.assertEqual(result["exitcode"], 0)
        self.assertIn("bar", result["stdout"])
        self.assertNotIn("foo", result["stdout"])


if __name__ == "__main__":
    unittest.main()
