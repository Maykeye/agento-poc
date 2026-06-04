from typing import Annotated

from agento.config import real_path
from agento.llm import LLM
from agento.tool import Tool
from agento.tool.editor.editor import ToolEditor
from agento.tool.editor.tool_list import EDITOR_TOOLS
from agento.tool.io import ToolWriteFile
from agento import utils


class EditorToolInsertBefore(Tool):
    """Insert text before a pattern in the file."""

    def __init__(self):
        super().__init__(
            name="insert_before",
            description="""Insert text before the first line of a pattern match.

FINDS the text-to-find in the entire file. Must find exactly one occurrence.
- 0 occurrences => error
- >1 occurrences => error

Then inserts text-to-insert BEFORE the first line of the match.
Supports multiline patterns and multiline insert text.

Example:
  insert_before "LINE102" "Line101.5"
  inserts "Line101.5" on the line before "LINE102"
""",
        )

    def __call__(
        self,
        text_to_find: Annotated[str, "Text to find (must exist exactly once in file)"],
        text_to_insert: Annotated[str, "Text to insert before the match"],
    ):
        # Get current LLM instance
        if not LLM.INSTANCES:
            return {"error": "No LLM instance available"}

        llm = LLM.INSTANCES[-1].llm
        llm_id = id(llm)

        # Get current state
        if llm_id not in ToolEditor._state:
            return {"error": "Not in editor mode"}

        path = ToolEditor._state[llm_id].path

        try:
            match_start, match_end, matched_text, total_matches = (
                _editor_find_unique_pattern(path, text_to_find)
            )
            del (match_end, matched_text, total_matches)
        except ValueError as e:
            return {"error": str(e)}

        # Insert before the match (at match_start line)
        return _editor_insert_text(path, match_start, text_to_insert)


class EditorToolInsertAfter(Tool):
    """Insert text after a pattern in the file."""

    def __init__(self):
        super().__init__(
            name="insert_after",
            description="""Insert text after the last line of a pattern match.

FINDS the text-to-find in the entire file. Must find exactly one occurrence.
- 0 occurrences => error
- >1 occurrences => error

Then inserts text-to-insert AFTER the last line of the match.
Supports multiline patterns and multiline insert text.

Example:
  insert_after "LINE102" "Line102.5"
  inserts "Line102.5" on the line after "LINE102"
""",
        )

    def __call__(
        self,
        text_to_find: Annotated[str, "Text to find (must exist exactly once in file)"],
        text_to_insert: Annotated[str, "Text to insert after the match"],
    ):
        # Get current LLM instance
        if not LLM.INSTANCES:
            return {"error": "No LLM instance available"}

        llm = LLM.INSTANCES[-1].llm
        llm_id = id(llm)

        # Get current state
        if llm_id not in ToolEditor._state:
            return {"error": "Not in editor mode"}

        path = ToolEditor._state[llm_id].path

        try:
            match_start, match_end, matched_text, total_matches = (
                _editor_find_unique_pattern(path, text_to_find)
            )
            del (match_start, matched_text, total_matches)
        except ValueError as e:
            return {"error": str(e)}

        # Insert after the match (at match_end + 1 line)
        return _editor_insert_text(path, match_end + 1, text_to_insert)


def _editor_find_unique_pattern(path: str, pattern: str) -> tuple[int, int, str, int]:
    """Find exactly one occurrence of a pattern in the entire file.

    Args:
        path: Path to file
        pattern: Text pattern to search for (not regex, plain text)

    Returns:
        Tuple of (match_start_line, match_end_line, matched_text, total_matches)
        where lines are 1-indexed.

    Raises:
        ValueError: If pattern is not found or found multiple times
    """
    p = real_path(path)
    text = p.read_text()

    # Find all occurrences of the pattern in the file
    # We need to search for multiline matches
    matches = []
    start_idx = 0
    while True:
        idx = text.find(pattern, start_idx)
        if idx == -1:
            break
        # Calculate which line this starts on
        lines_before = text[:idx].count("\n")
        start_line = lines_before + 1  # 1-indexed

        # Calculate which line this ends on
        matched_text = pattern
        end_line = start_line + matched_text.count("\n")

        matches.append((start_line, end_line, matched_text))
        start_idx = idx + 1

    total_matches = len(matches)

    if total_matches == 0:
        raise ValueError(f"Pattern '{pattern}' not found in file. Total occurrences: 0")

    if total_matches > 1:
        raise ValueError(
            f"Pattern '{pattern}' found {total_matches} times in file. "
            f"Must be exactly once."
        )

    match_start, match_end, matched_text = matches[0]
    return match_start, match_end, matched_text, total_matches


def _editor_insert_text(
    path: str,
    insert_line: int,
    text_to_insert: str,
) -> str:
    """Insert text at a specific line in the file.

    Args:
        path: Path to file
        insert_line: Line number to insert before (1-indexed)
        text_to_insert: Text to insert (can be multiline)
        llm_id: LLM instance ID for updating state

    Returns:
        Formatted output string with confirmation and buffer
    """
    p = real_path(path)
    full_text = p.read_text()
    full_lines = full_text.splitlines()
    total_lines = len(full_lines)

    # Validate insert_line
    if insert_line < 1:
        insert_line = 1
    if insert_line > total_lines + 1:
        insert_line = total_lines + 1

    # Split text to insert into lines
    insert_lines = text_to_insert.splitlines()

    # Insert the lines
    # insert_line is 1-indexed, so we insert at index insert_line - 1
    insert_idx = insert_line - 1
    new_lines = full_lines[:insert_idx] + insert_lines + full_lines[insert_idx:]

    # Write updated content
    new_text = "\n".join(new_lines)
    if full_text and not full_text.endswith("\n"):
        # Preserve no-trailing-newline if original didn't have it
        new_text = new_text.rstrip("\n")
    elif full_text.endswith("\n") and insert_lines:
        # Ensure trailing newline if original had it
        new_text = new_text + "\n"

    write_tool = ToolWriteFile()
    write_result = write_tool(path, new_text)

    if isinstance(write_result, dict) and "error" in write_result:
        return write_result  # type: ignore

    # Format success output
    output_lines = []
    output_lines.append(f"Inserted {len(insert_lines)} line(s) at line {insert_line}")
    output_lines.append("```diff")
    output_lines.extend(utils.diff_gen(full_text, new_text, path))
    output_lines.append("```")

    # Prune old buffers
    messages = LLM.INSTANCES[-1].messages
    ToolEditor._prune_old_buffers(messages)

    return "\n".join(output_lines)


EDITOR_TOOLS.append(EditorToolInsertAfter)
EDITOR_TOOLS.append(EditorToolInsertBefore)
