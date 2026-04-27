import re
from tool import Tool
from typing import Annotated
from tool.editor.editor import ToolEditor
from tool.editor.tool_list import EDITOR_TOOLS
from config import real_path
from llm import LLM


class EditorToolFindPrev(Tool):
    """Find previous occurrence of pattern by searching backwards from current line."""

    def __init__(self):
        super().__init__(
            name="find_prev",
            description="""Search backwards from current line (starting from current_line-1) for a multiline regex pattern.

When pattern is found, displays matching lines in format:
  00123|line content
  00124|line content

And shows pattern info: Pattern: [current_occurrence]/[total_occurrences]

After finding, sets current line to pattern's first line - 5 (doesn't wrap to negative).
Then prints buffer and prunes old messages.""",
        )

    def __call__(
        self, pattern: Annotated[str, "Multiline regex pattern to search for"]
    ):
        try:
            llm_id, current_line, path = _editor_get_current_state()
        except ValueError as e:
            return {"error": str(e)}

        match_start, match_end, _, occurrence_num, total_matches = (
            _editor_search_pattern(path, pattern, current_line, search_backward=True)
        )

        if match_start is None or match_end is None:
            return {
                "status": "not_found",
                "message": f"Pattern '{pattern}' not found before line {current_line}",
                "total_occurrences": total_matches,
            }

        p = real_path(path)
        text = p.read_text()
        lines = text.splitlines()

        return _editor_format_search_result(
            path,
            pattern,
            match_start,
            match_end,
            lines,
            occurrence_num,
            total_matches,
            llm_id,
        )


class EditorToolFindNext(Tool):
    """Find next occurrence of pattern by searching forwards from current line."""

    def __init__(self):
        super().__init__(
            name="find_next",
            description="""Search forwards from current line (starting from current_line+1) for a multiline regex pattern.

When pattern is found, displays matching lines in format:
  00123|line content
  00124|line content

And shows pattern info: Pattern: [current_occurrence]/[total_occurrences]

After finding, sets current line to pattern's first line - 5 (doesn't wrap to negative).
Then prints buffer and prunes old messages.""",
        )

    def __call__(
        self, pattern: Annotated[str, "Multiline regex pattern to search for"]
    ):
        try:
            llm_id, current_line, path = _editor_get_current_state()
            match_start, match_end, _, occurrence_num, total_matches = (
                _editor_search_pattern(
                    path, pattern, current_line, search_backward=False
                )
            )
        except ValueError as e:
            return {"error": str(e)}

        if match_start is None or match_end is None:
            return {
                "status": "not_found",
                "message": f"Pattern '{pattern}' not found after line {current_line}",
                "total_occurrences": total_matches,
            }

        p = real_path(path)
        text = p.read_text()
        lines = text.splitlines()

        return _editor_format_search_result(
            path,
            pattern,
            match_start,
            match_end,
            lines,
            occurrence_num,
            total_matches,
            llm_id,
        )


def _editor_search_pattern(
    path: str,
    pattern: str,
    current_line: int,
    search_backward: bool,
) -> tuple[int | None, int | None, str | None, int, int]:
    """Search for a pattern in file, either forward or backward from current line.

    Args:
        path: Path to file
        pattern: Regex pattern to search for
        current_line: Current line number (1-indexed)
        search_backward: If True, search backward; if False, search forward

    Returns:
        Tuple of (match_start, match_end, matched_text, occurrence_num, total_matches)
        If no match found, returns (None, None, None, 0, total_matches)
    """
    p = real_path(path)
    text = p.read_text()
    lines = text.splitlines()
    total_lines = len(lines)

    try:
        regex = re.compile(pattern, re.MULTILINE)
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {e}")

    # Count total occurrences
    total_matches = len(list(regex.finditer(text)))

    # Determine search range
    if search_backward:
        # Search backward from current_line - 1
        search_start_idx = current_line - 2
        search_range = range(search_start_idx, -1, -1)
    else:
        # Search forward from current_line + 1
        search_start_idx = current_line
        search_range = range(search_start_idx, total_lines)

    # Find match
    match_found = None
    for i in search_range:
        remaining_text = "\n".join(lines[i:])
        match = regex.search(remaining_text)
        if match:
            # Calculate the actual line where the match occurs within remaining_text
            # Count newlines before the match position to find the line offset
            match_start_pos = match.start()
            lines_before_match = remaining_text[:match_start_pos].count("\n")
            match_line = i + 1 + lines_before_match  # Convert to 1-indexed
            matched_text = match.group()
            matched_lines = matched_text.count("\n") + 1
            match_end_line = match_line + matched_lines - 1
            match_found = (match_line, match_end_line, matched_text)
            break

    if not match_found:
        return None, None, None, 0, total_matches

    match_start, match_end, matched_text = match_found

    # Find occurrence number
    occurrence_num = 0
    for i, line in enumerate(lines, 1):
        if regex.search(line):
            occurrence_num += 1
            if i >= match_start:
                break

    return match_start, match_end, matched_text, occurrence_num, total_matches


def _editor_format_search_result(
    path: str,
    pattern: str,
    match_start: int,
    match_end: int,
    lines: list[str],
    occurrence_num: int,
    total_matches: int,
    llm_id: int,
) -> str:
    """Format search result and update editor state.

    Returns formatted output string with matched lines and buffer.
    """
    # Format matched lines
    matched_lines_output = []
    for i in range(match_start - 1, match_end):
        line_num = i + 1
        line_content = lines[i]
        matched_lines_output.append(f"{line_num:05d}|{line_content}")

    # Update current line
    new_current_line = max(1, match_start - 5)
    ToolEditor._state[llm_id].current_line = new_current_line

    # Format output
    output_lines = []
    output_lines.append(f"Found pattern '{pattern}':")
    output_lines.append("")
    output_lines.extend(matched_lines_output)
    output_lines.append("")
    output_lines.append(f"Pattern: {occurrence_num}/{total_matches}")
    output_lines.append("")

    # Print buffer
    text = "\n".join(lines)
    buffer_output = ToolEditor._format_buffer(path, new_current_line, text)
    output_lines.append(buffer_output)

    # Prune old buffers
    messages = LLM.INSTANCES[-1].messages
    ToolEditor._prune_old_buffers(messages)

    return "\n".join(output_lines)


def _editor_get_current_state():
    """Get current editor state (llm_id, current_line, path).

    Returns tuple of (llm_id, current_line, path) or raises ValueError if not in editor mode.
    TODO: replace with _state
    """
    if not LLM.INSTANCES:
        raise ValueError("No LLM instance available")

    llm = LLM.INSTANCES[-1].llm
    llm_id = id(llm)

    if llm_id not in ToolEditor._state:
        raise ValueError("Not in editor mode")

    state = ToolEditor._state[llm_id]
    return llm_id, state.current_line, state.path


EDITOR_TOOLS.append(EditorToolFindNext)
EDITOR_TOOLS.append(EditorToolFindPrev)
