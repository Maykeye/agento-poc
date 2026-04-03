"""Test fork tool for proper subtask isolation"""

from tool_fork import ToolFork
from llm import LLM


def foo():
    # Create LLM with fork tool
    llm = LLM()
    llm.add_tool(ToolFork())
    msgs = llm.prepend_system_message(
        [
            llm.msg_user(
                """We need to calculate two values using fork and then combine them.

Please use the fork tool to:
1. First fork: Calculate the SUM of numbers from 1 to 6 (including 1 and 6). Report the result.
2. Second fork: Calculate the PRODUCT (multiplication) of numbers from 1 to 6 (including 1 and 6). Report the result.

After both forks complete, the main task should:
3. Add the two results together (sum + product)
4. Verify the final answer is correct
"""
            ),
        ]
    )
    llm.generate(msgs)


if __name__ == "__main__":
    foo()
