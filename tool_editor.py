"""
Tool Editor - A specialized editing mode for LLM with buffer-based operations.

This module provides an interactive editing mode where LLM works with a buffered view
of a file, with specialized tools for navigation, pattern searching, and editing.
"""

import copy
import json
import re
from dataclasses import dataclass
from typing import Annotated

from config import real_path
from llm import LLM, FinishGeneration
from tool import Tool
from tool_edit_patch import ToolEditDiffPatch
from tool_io import ToolAppend, ToolWriteFile, ToolReadFile

# Constants
LINES = 250  # Size of buffer (number of lines to show)
KEEP_OLD_BUFFERS = 5  # Number of buffer prints to keep in messages


@dataclass
class EditorEntry:
    path: str
    current_line: int  # 1 indexed


class ToolEditor(Tool):
    _state: dict[int, EditorEntry] = {}  # {id(LLM) -> state }

    @staticmethod
    def reset(id=None):
        if id is None:
            ToolEditor._state.clear()
        else:
            if id in ToolEditor._state:
                del ToolEditor._state[id]

    SKIP_PRINTING: bool = False

    def __init__(self):
        super().__init__(
            name="edit_file",
            description="""Start editing a file in special editor mode.

This tool enters an interactive editing session where you work with a buffered view of the file.
Empty or non-existing files can be edited - they will be initialized with empty content.""",
        )

    @staticmethod
    def init_editor_tools(llm: LLM):
        llm.add_tool(EditorToolPrint())
        llm.add_tool(EditorToolFindPrev())
        llm.add_tool(EditorToolFindNext())
        llm.add_tool(EditorToolGoto())
        # llm.add_tool(EditorToolPatchCurrentFile())
        # llm.add_tool(tool_suffix_patch.ToolPatchSuffix())
        llm.add_tool(EditorToolEditFile())
        llm.add_tool(EditorToolSwitchFile())
        llm.add_tool(EditorToolRead())
        llm.add_tool(EditorToolWriteNewContent())
        llm.add_tool(EditorToolAppend())
        llm.add_tool(EditorToolFinishEditing())
        llm.add_tool(EditorToolInsertBefore())
        llm.add_tool(EditorToolInsertAfter())

    def __call__(
        self,
        path: Annotated[str, "Path to the file to edit (must exist)"],
    ):
        p = real_path(path)

        if not p.exists():
            p.write_text("")
        elif not p.is_file():
            return {"error": f"{path} is not a file"}

        # Read file content
        text = p.read_text()

        # Check if file is empty
        if not text.strip():
            # Write empty string to the file to allow editing
            p.write_text("")
            text = ""

        # Clone the current LLM
        original_llm = LLM.INSTANCES[-1].llm
        original_messages = LLM.INSTANCES[-1].messages

        # Create new LLM for editing
        editor_llm = original_llm.clone()

        # Remove all tools from the cloned LLM
        editor_llm.tools.clear()

        # Add editor-specific tools
        ToolEditor.init_editor_tools(editor_llm)

        # Store current line and file path for this LLM instance
        llm_id = id(editor_llm)
        ToolEditor._state[llm_id] = EditorEntry(path, 1)

        # Prepare messages for the editor LLM
        editor_messages = copy.deepcopy(original_messages)

        # Remove the parent's tool call from the clone's memory
        if (
            editor_messages[-1]["role"] == "assistant"
            and "tool_calls" in editor_messages[-1]
        ):
            editor_messages.pop()

        # Add system message about editor mode
        editor_messages.append(
            {
                "role": "user",
                "content": f"""[SYSTEM OVERRIDE: EDITOR MODE ACTIVATED]

You are now in EDITOR MODE for file: {path}

EDITOR MODE RULES:
1. You are working with a buffered view of the file (showing {LINES} lines at a time)
2. Line numbers are 1-indexed and displayed as 5-digit numbers (e.g., 00001, 00123)
3. The buffer shows lines starting from your current line position
4. You can navigate using: goto <line>, find_prev/find_next <regex>
5. You can switch files using: edit_file <path>
6. When done, use finish_editing(report) to quit editing and continue other tasks(e.g. compilation)
7. All tools except read work only on the current buffer view
8. The current file path is: {path}

Current buffer (starting from line 1):"""
                + "\n"
                + ToolEditor._format_buffer(path, 1, text),
            }
        )

        # Generate response from editor LLM
        result = editor_llm.generate(editor_messages)

        # After editing is done, restore current line tracking for original LLM
        # (editor LLM instance will be cleaned up)

        return {"status": "editing_complete", "file": path, "result": result.content}

    @staticmethod
    def _format_buffer(path: str, start_line: int, full_text: str) -> str:
        """Format a buffer view of the file starting from start_line (1-indexed).

        Returns formatted text with line numbers.
        Line numbers are shown only for:
        - First line of buffer
        - Last line of buffer
        - Lines divisible by 10
        """
        lines = full_text.splitlines()
        total_lines = len(lines)

        # Calculate the range to display
        end_line = min(start_line + LINES - 1, total_lines)

        # Format lines with line numbers
        buffer_lines = []
        for i in range(start_line - 1, end_line):
            line_num = i + 1  # Convert to 1-indexed
            line_content = lines[i]

            # Determine if we should show the line number
            is_first_line = i == start_line - 1
            is_last_line = i == end_line - 1
            is_divisible_by_10 = line_num % 10 == 0

            if is_first_line or is_last_line or is_divisible_by_10:
                buffer_lines.append(f"{line_num:05d}|{line_content}")
            else:
                buffer_lines.append(f"     |{line_content}")

        # Add file info at the top
        info = f"[FILE: {path} | LINES: {start_line}-{end_line}/{total_lines}]"

        result = "\n".join([info] + buffer_lines)
        if not ToolEditor.SKIP_PRINTING:
            print(result)
        return result

    @staticmethod
    def _prune_old_buffers(messages: list[dict], keep_count: int = KEEP_OLD_BUFFERS):
        """Prune old buffer prints and tool calls from messages.

        Iterates messages from last backwards and prunes tool calls and outputs
        that are beyond keep_count from the end.

        This is similar to context/suffix.py pruning but for editor buffers.
        """

        # Track buffer prints from the end
        buffer_count = 0
        editor_tools = {
            "print_buffer",
            "find_prev",
            "find_next",
            "goto",
            "patch_current_file",
            "patch_suffix",
            "edit_file",
            "read_file",
            "write_file",
            "finish_editing",
            "insert_before",
            "insert_after",
        }

        # Iterate backwards
        for msg_idx in range(len(messages) - 1, -1, -1):
            msg = messages[msg_idx]

            # Check tool output messages
            if msg.get("role") == "tool":
                tool_name = msg.get("name", "")
                if tool_name in editor_tools:
                    buffer_count += 1
                    if buffer_count > keep_count:
                        # Prune this tool output
                        msg["content"] = f"[PRUNED: Old {tool_name} output]"

            # Check assistant messages with tool calls
            elif msg.get("role") == "assistant":
                tool_calls = msg.get("tool_calls", [])
                for tc in tool_calls:
                    func_name = tc.get("function", {}).get("name", "")
                    if func_name in editor_tools:
                        # This will be pruned when its output is pruned
                        pass


# ============================================================================
# Editor-Specific Tools
# ============================================================================


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


class EditorToolPrint(Tool):
    """Print current buffer view (called implicitly after navigation/edit operations)"""

    def __init__(self):
        super().__init__(
            name="print_buffer",
            description="Print current buffer view (called implicitly after navigation/edit operations)",
        )

    def __call__(self):
        """Print current buffer view.

        This is called implicitly after navigation/edit operations to show the current state.
        """

        # Get current LLM instance
        if not LLM.INSTANCES:
            return {"error": "No LLM instance available"}

        llm = LLM.INSTANCES[-1].llm
        llm_id = id(llm)

        # Get current state
        if llm_id not in ToolEditor._state:
            return {"error": "Not in editor mode"}

        state = ToolEditor._state[id(llm)]
        p = real_path(state.path)
        text = p.read_text()

        # Print buffer from current line
        buffer_output = ToolEditor._format_buffer(state.path, state.current_line, text)

        return buffer_output


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


class EditorToolGoto(Tool):
    """Go to a specific line and update current line position."""

    def __init__(self):
        super().__init__(
            name="goto",
            description="""Jump to a specific line number (1-indexed) and set current line to it.

After jumping, prints the buffer starting from the new current line position.
Line numbers must be within file bounds (1 to total_lines).
After goto, old buffer messages are pruned to save context.""",
        )

    def __call__(
        self, line_number: Annotated[int, "Line number to jump to (1-indexed)"]
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

        p = real_path(path)
        text = p.read_text()
        lines = text.splitlines()
        total_lines = len(lines)

        # Validate line number is within bounds (1-indexed)
        if line_number < 1:
            return {"error": f"Line number must be >= 1, got {line_number}"}

        if line_number > total_lines:
            return {
                "error": f"Line number {line_number} exceeds file length ({total_lines} lines)",
                "suggestion": f"Use a line number between 1 and {total_lines}",
            }

        # Update current line
        ToolEditor._state[llm_id].current_line = line_number

        # Format output with buffer
        output_lines = []
        output_lines.append(f"Goto line {line_number}")
        output_lines.append("")

        # Print buffer from new current line
        buffer_output = ToolEditor._format_buffer(path, line_number, text)
        output_lines.append(buffer_output)

        # Prune old buffers
        messages = LLM.INSTANCES[-1].messages
        ToolEditor._prune_old_buffers(messages)

        return "\n".join(output_lines)


class EditorToolPatchCurrentFile(Tool):
    """Apply a unified diff patch to the current file.

    PREFERRED EDITING METHOD: This is the recommended way to edit files in editor mode.
    More reliable than edit_file for complex changes and supports multiple modifications.
    """

    def __init__(self):
        super().__init__(
            name="patch_current_file",
            description="""Apply a unified diff patch to the current file.

PREFERRED EDITING METHOD: This is the recommended way to edit files in editor mode.
Use this tool for most edits as it's more reliable and allows multiple changes at once.

REQUIREMENTS:
- Patch hunks must overlap with the current buffer view (the {LINES} lines currently visible)
- If patch lacks '---' and '+++' headers, they will be added automatically using current filename
- After successful patching, buffer is printed from current line position
- Old buffer messages are pruned to save context

FORMAT: Standard unified diff format with @@ hunks.
Example:
--- a/filename
+++ b/filename
@@ -start,count +start,count @@
 context line
-removed line
+added line
 context line""",
        )

    def __call__(
        self,
        patch_text: Annotated[
            str,
            "Unified diff patch to apply (must have hunks overlapping current buffer)",
        ],
    ):
        # Get current LLM instance
        if not LLM.INSTANCES:
            return {"error": "No LLM instance available"}

        llm = LLM.INSTANCES[-1].llm
        llm_id = id(llm)

        # Get current state
        if llm_id not in ToolEditor._state:
            return {"error": "Not in editor mode"}

        current_line = ToolEditor._state[llm_id].current_line
        path = ToolEditor._state[llm_id].path

        p = real_path(path)
        text = p.read_text()
        lines = text.splitlines()
        total_lines = len(lines)

        # Ensure patch ends with newline
        if not patch_text.endswith("\n"):
            patch_text += "\n"

        # Check if patch has proper headers
        patch_lines = patch_text.splitlines()
        needs_headers = True

        if (
            len(patch_lines) >= 2
            and patch_lines[0].startswith("--- ")
            and patch_lines[1].startswith("+++ ")
        ):
            needs_headers = False

        # Add headers if missing
        if needs_headers:
            # Format: --- a/filename and +++ b/filename
            patch_text = f"--- a/{path}\n+++ b/{path}\n" + patch_text

        # Validate patch format (basic check)
        patch_lines = patch_text.splitlines()
        if not patch_lines[0].startswith("--- a/"):
            return {
                "error": f"Patch must start with '--- a/', got: {patch_lines[0]}",
                "suggestion": "Use proper unified diff format",
            }

        if len(patch_lines) < 2 or not patch_lines[1].startswith("+++ b/"):
            return {
                "error": f"Patch second line must start with '+++ b/', got: {patch_lines[1] if len(patch_lines) > 1 else 'no second line'}",
                "suggestion": "Use proper unified diff format",
            }

        # Calculate current buffer range
        buffer_start = current_line
        buffer_end = min(current_line + LINES - 1, total_lines)

        # Parse patch to check if hunks overlap with buffer
        # Look for @@ lines (hunk headers)
        hunk_pattern = re.compile(r"^@@ -(\d+),\d+ \+(\d+),\d+ @@")
        hunks_in_patch = []

        for i, line in enumerate(patch_lines):
            match = hunk_pattern.match(line)
            if match:
                old_start = int(match.group(1))
                # Collect hunk body to determine end line
                hunk_body_lines = 0
                j = i + 1
                while j < len(patch_lines):
                    if patch_lines[j].startswith("@@ ") or patch_lines[j].startswith(
                        "--- "
                    ):
                        break
                    if patch_lines[j].startswith("-") or patch_lines[j].startswith(" "):
                        hunk_body_lines += 1
                    j += 1

                hunk_end = old_start + hunk_body_lines - 1
                hunks_in_patch.append((old_start, hunk_end))

        # Check if any hunk overlaps with buffer
        # A hunk overlaps if its range intersects with buffer range
        overlap_found = False
        for hunk_start, hunk_end in hunks_in_patch:
            # Check for overlap: not (hunk_end < buffer_start or hunk_start > buffer_end)
            if not (hunk_end < buffer_start or hunk_start > buffer_end):
                overlap_found = True
                break

        if not overlap_found and hunks_in_patch:
            return {
                "error": f"Patch hunks do not overlap with current buffer (lines {buffer_start}-{buffer_end})",
                "suggestion": f"Navigate to a line that shows the code you want to patch, then try again",
                "buffer_range": f"{buffer_start}-{buffer_end}",
                "patch_hunks": hunks_in_patch,
            }

        # Temporarily disable debug file saving
        old_skip = ToolEditDiffPatch.SKIP_SAVING_INVALID_PATCHES
        ToolEditDiffPatch.SKIP_SAVING_INVALID_PATCHES = True

        try:
            patch_tool = ToolEditDiffPatch()
            result = patch_tool(path, patch_text)

            # Check if patch was successful
            if isinstance(result, dict) and "error" in result:
                return {
                    "error": f"Failed to apply patch: {result['error']}",
                    "status": "patch_failed",
                }

            # Read updated file content
            updated_text = p.read_text()
            updated_lines = updated_text.splitlines()
            new_total_lines = len(updated_lines)

            # Adjust current line if necessary (ensure it's still valid)
            if current_line > new_total_lines:
                current_line = new_total_lines
                ToolEditor._state[llm_id].current_line = current_line

            # Format success output
            output_lines = []
            output_lines.append("Patch applied successfully")
            output_lines.append("")

            # Print buffer from current line
            buffer_output = ToolEditor._format_buffer(path, current_line, updated_text)
            output_lines.append(buffer_output)

            # Prune old buffers
            messages = LLM.INSTANCES[-1].messages
            ToolEditor._prune_old_buffers(messages)

            return "\n".join(output_lines)

        finally:
            # Restore original setting
            ToolEditDiffPatch.SKIP_SAVING_INVALID_PATCHES = old_skip


class EditorToolEditFile(Tool):
    """Edit file by replacing text in the current visible buffer.

    CRITICAL: This tool works ONLY on the visible buffer view, not the entire file.
    PREFERRED ALTERNATIVE: Use patch_current_file for most edits as it's more reliable.
    """

    def __init__(self):
        super().__init__(
            name="search_and_replace",
            description="""Replace text in the current VISIBLE BUFFER only.

⚠️ CRITICAL RESTRICTIONS:
- Works ONLY on the visible buffer (LINES={LINES} lines from current position)
- replace_from must exist EXACTLY ONCE in the current buffer view
- If text exists outside buffer or multiple times in buffer, operation will fail
- After replacement, file is written immediately and buffer is reprinted

🔧 PREFERRED APPROACH: Use patch_current_file instead for:
  - Multiple changes at once
  - More reliable edits
  - Better context control
  - Edits outside current buffer view

Use search_and_replace only for simple, single replacements within visible buffer.""",
        )

    def __call__(
        self,
        replace_from: Annotated[
            str,
            "Text to find (must exist exactly once in CURRENT BUFFER, not whole file)",
        ],
        replace_with: Annotated[str, "Text to replace with"],
    ):
        # Get current LLM instance
        if not LLM.INSTANCES:
            return {"error": "No LLM instance available"}

        llm = LLM.INSTANCES[-1].llm
        llm_id = id(llm)

        # Get current state
        if llm_id not in ToolEditor._state:
            return {"error": "Not in editor mode"}

        current_line = ToolEditor._state[llm_id].current_line
        path = ToolEditor._state[llm_id].path

        p = real_path(path)
        full_text = p.read_text()
        full_lines = full_text.splitlines()
        total_lines = len(full_lines)

        # Calculate buffer range
        buffer_start_idx = current_line - 1  # Convert to 0-indexed
        buffer_end_idx = min(current_line + LINES - 1, total_lines)  # Exclusive

        # Get buffer content (the visible lines)
        buffer_lines = full_lines[buffer_start_idx:buffer_end_idx]
        buffer_content = "\n".join(buffer_lines)

        # Check if replace_from exists in buffer
        buffer_count = buffer_content.count(replace_from)

        if buffer_count == 0:
            # Show where it might be in the buffer for debugging
            return {
                "error": f"Text not found in current buffer (lines {current_line}-{buffer_end_idx})",
                "replace_from": repr(replace_from),
                "buffer_range": f"Lines {current_line} to {buffer_end_idx}",
                "suggestion": "Use goto or find_next/find_prev to navigate to the text you want to edit, or use patch_current_file",
            }

        if buffer_count > 1:
            return {
                "error": f"Text appears {buffer_count} times in current buffer (must be exactly once)",
                "replace_from": repr(replace_from),
                "suggestion": "Be more specific with replace_from to match exactly one occurrence, or use patch_current_file for multiple changes",
            }

        # Now we need to replace in the FULL file, not just buffer
        # Find the position in the full file
        full_count = full_text.count(replace_from)

        if full_count == 0:
            # This shouldn't happen since we found it in buffer, but just in case
            return {"error": "Text not found in file (internal error)"}

        # Replace in full text (replace all occurrences in file)
        # Since we verified it's unique in buffer, and buffer is part of file,
        # we replace it in the full file
        new_full_text = full_text.replace(replace_from, replace_with)

        # Write the updated file using ToolWriteFile

        write_tool = ToolWriteFile()
        write_result = write_tool(path, new_full_text)

        # Check if write was successful
        if isinstance(write_result, dict) and "error" in write_result:
            return write_result

        # Read updated file to get new content
        updated_text = p.read_text()
        updated_lines = updated_text.splitlines()
        new_total_lines = len(updated_lines)

        # Adjust current line if necessary (ensure it's still valid)
        if current_line > new_total_lines:
            current_line = max(1, new_total_lines)
            ToolEditor._state[llm_id].current_line = current_line

        # Format success output
        output_lines = []
        output_lines.append(f"Replaced text in {path}")
        output_lines.append(f"  From: {repr(replace_from)}")
        output_lines.append(f"  To:   {repr(replace_with)}")
        output_lines.append("")

        # Print buffer from current line
        buffer_output = ToolEditor._format_buffer(path, current_line, updated_text)
        output_lines.append(buffer_output)

        # Prune old buffers
        messages = LLM.INSTANCES[-1].messages
        ToolEditor._prune_old_buffers(messages)

        return "\n".join(output_lines)


def _editor_insert_text(
    path: str,
    insert_line: int,
    text_to_insert: str,
    llm_id: int,
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

    # Read updated file to get new content
    updated_text = p.read_text()
    updated_lines = updated_text.splitlines()
    new_total_lines = len(updated_lines)

    # Get current line before updating
    current_line = ToolEditor._state[llm_id].current_line

    # Adjust current line if necessary (shift if we inserted before it)
    lines_added = len(insert_lines)
    if insert_line <= current_line:
        current_line += lines_added

    # Ensure current line is still valid
    if current_line > new_total_lines:
        current_line = new_total_lines
    if current_line < 1:
        current_line = 1

    ToolEditor._state[llm_id].current_line = current_line

    # Format success output
    output_lines = []
    output_lines.append(f"Inserted {len(insert_lines)} line(s) at line {insert_line}")
    output_lines.append("")

    # Print buffer from current line
    buffer_output = ToolEditor._format_buffer(path, current_line, updated_text)
    output_lines.append(buffer_output)

    # Prune old buffers
    messages = LLM.INSTANCES[-1].messages
    ToolEditor._prune_old_buffers(messages)

    return "\n".join(output_lines)


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
        return _editor_insert_text(path, match_start, text_to_insert, llm_id)


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
        return _editor_insert_text(path, match_end + 1, text_to_insert, llm_id)


class EditorToolRead(Tool):
    """Read another file while in editor mode."""

    def __init__(self):
        super().__init__(
            name="read_file_to_view",
            description="""Read another file to view while in editor mode.

This allows you to view other files without leaving the editor.
This does not change the current file being edited.
All edit commands after read_file will apply to the file that was edited before reading""",
        )

    def __call__(self, path: Annotated[str, "Path to the file to read"]):
        read_tool = ToolReadFile()
        result = read_tool(path)
        if isinstance(result, str):
            llm = LLM.INSTANCES[-1].llm
            llm_id = id(llm)
            current_path = ToolEditor._state[llm_id].path
            result += f"\n\nNote: still editing {current_path}"

        return result


class EditorToolSwitchFile(Tool):
    """Switch to editing a different file while in editor mode."""

    def __init__(self):
        super().__init__(
            name="edit_file",
            description="""Switch to editing a different file while in editor mode.

This tool allows you to switch to a new file without exiting editor mode.

REQUIREMENTS:
- The new file must exist
- After switching, the editor continues with the new file starting from line 1
- Empty files will be initialized with empty content

Use this when you need to edit other file from the current one quckly, but it's preferrably to finish_editing and in report state what needs to be edited.""",
        )

    def __call__(self, path: Annotated[str, "Path to the new file to edit"]):
        # Get current LLM instance
        if not LLM.INSTANCES:
            return {"error": "No LLM instance available"}

        llm = LLM.INSTANCES[-1].llm
        llm_id = id(llm)

        # Check if we're in editor mode
        if llm_id not in ToolEditor._state:
            return {"error": "Not in editor mode"}

        # Validate the new file
        p = real_path(path)

        if not p.exists():
            return {
                "error": f"File {path} does not exist",
                "suggestion": "Use write_file to create a new file first",
            }

        if not p.is_file():
            return {"error": f"{path} is not a file"}

        # Read the new file content
        try:
            new_text = p.read_text()
        except Exception as e:
            return {"error": f"Failed to read file {path}: {e}"}

        # Check if file is empty
        if not new_text.strip():
            # Write empty string to the file to allow editing
            p.write_text("")
            new_text = ""

        # Prune old buffers if we exceed KEEP_OLD_BUFFERS
        messages = LLM.INSTANCES[-1].messages
        ToolEditor._prune_old_buffers(messages)

        # Update editing state to the new file
        old_file = ToolEditor._state[llm_id]
        ToolEditor._state[llm_id] = EditorEntry(path, 1)

        # Format output
        output_lines = []
        output_lines.append(f"Switching from '{old_file}' to '{path}'")
        output_lines.append("")

        # Print buffer from line 1 of new file
        buffer_output = ToolEditor._format_buffer(path, 1, new_text)
        output_lines.append(buffer_output)

        return "\n".join(output_lines)


class EditorToolWriteNewContent(Tool):
    """Write new content to the current file and exit editing mode."""

    def __init__(self):
        super().__init__(
            name="write_file",
            description="""Write the complete new content to the current file and exit editing mode.

This tool:
1. Validates that the provided path matches the currently editing file
2. Writes the entire `text` to the file (replacing all existing content)
3. Exits editor mode
4. Returns a result indicating success

REQUIREMENTS:
- path: Must match the file currently being edited in editor mode, as only editing it is allowed.
- text: Complete new content to write (replaces all existing content)

Use this when you want to completely replace the file content with new content.
The editor will exit and return control to the main LLM.""",
        )

    def __call__(
        self,
        path: Annotated[
            str,
            "Path to the file to write to (must match the file currently being edited)",
        ],
        new_content: Annotated[
            str,
            "Complete new content to write to the file (replaces all existing content)",
        ],
    ):
        # Get current LLM instance
        if not LLM.INSTANCES:
            return {"error": "No LLM instance available"}

        llm = LLM.INSTANCES[-1].llm
        llm_id = id(llm)

        # Get current state
        if llm_id not in ToolEditor._state:
            return {"error": "No file being edited"}

        current_editing_path = ToolEditor._state[llm_id].path

        # Validate that the provided path matches the currently editing file
        provided_path = real_path(path)
        editing_path = real_path(current_editing_path)

        # Normalize paths for comparison (resolve to absolute paths)
        if provided_path != editing_path:
            return {
                "error": f"Path mismatch",
                "suggestion": f"You are editing '{current_editing_path}' but tried to write to '{path}'. Use the correct path or use finish_editing() to exit and write_file() for a different file.",
                "editing_file": current_editing_path,
                "requested_path": path,
            }

        write_tool = ToolWriteFile()
        result = write_tool(path, new_content)

        # Check if write was successful
        if isinstance(result, dict) and "error" in result:
            return result

        # Clean up editor state
        ToolEditor.reset(llm_id)

        # Return FinishGeneration to stop the editor LLM and return to main LLM
        result_value = {
            "status": "success",
            "message": f"File {path} written successfully and editor mode exited",
            "file": path,
            "lines_written": len(new_content.splitlines()),
        }
        return FinishGeneration(value=json.dumps(result_value, indent=1))


class EditorToolAppend(Tool):
    """Write new content to the current file and exit editing mode."""

    def __init__(self):
        super().__init__(
            name="append_to_file",
            description="""Append text to the end of the file.

This tool:
1. Validates that the provided path matches the currently editing file
2. Writes the entire `text` at the end of the file (replacing all existing content)

REQUIREMENTS:
- path: Must match the file currently being edited in editor mode, as only editing it is allowed.
- text: Text to write

Use this when you want to append new content to the end of the file.""",
        )

    def __call__(
        self,
        path: Annotated[
            str,
            "Path to the file to write to (must match the file currently being edited)",
        ],
        text: Annotated[
            str,
            "Complete new content to write to the file (replaces all existing content)",
        ],
    ):
        # Get current LLM instance
        if not LLM.INSTANCES:
            return {"error": "No LLM instance available"}

        llm = LLM.INSTANCES[-1].llm
        llm_id = id(llm)

        # Get current state
        if llm_id not in ToolEditor._state:
            return {"error": "No file being edited"}

        current_editing_path = ToolEditor._state[llm_id].path

        # Validate that the provided path matches the currently editing file
        provided_path = real_path(path)
        editing_path = real_path(current_editing_path)

        # Normalize paths for comparison (resolve to absolute paths)
        if provided_path != editing_path:
            return {
                "error": f"Path mismatch",
                "suggestion": f"You are editing '{current_editing_path}' but tried to append to '{path}'. Use the correct path or use finish_editing() to exit and write_file() for a different file.",
                "editing_file": current_editing_path,
                "requested_path": path,
            }

        append_tool = ToolAppend()
        return append_tool(path, text)


class EditorToolFinishEditing(Tool):
    """Exit editing mode with an optional report without making changes."""

    def __init__(self):
        super().__init__(
            name="finish_editing",
            description="""Exit editing mode with a report that will be reported to the main agent.

Use this when you've finished reviewing/editing and want to return to the main LLM.
The report can describe what changes were made or what was observed.""",
        )

    def __call__(
        self,
        report: Annotated[
            str, "Optional report describing what was done during editing session"
        ],
    ):
        # Get current LLM instance
        if not LLM.INSTANCES:
            return {"error": "No LLM instance available"}

        llm = LLM.INSTANCES[-1].llm
        llm_id = id(llm)

        # Get current state
        if llm_id not in ToolEditor._state:
            return {"error": "No file being edited"}

        path = ToolEditor._state[llm_id].path

        # Clean up editor state
        ToolEditor.reset(llm_id)

        # Return FinishGeneration to stop the editor LLM and return to main LLM
        result_value = {
            "status": "editing_finished",
            "file": path,
            "report": report if report else "No report provided",
            "message": f"Editor mode exited for {path}",
        }
        return FinishGeneration(value=json.dumps(result_value, indent=1))
