from datetime import datetime
from typing import Annotated
from pathlib import Path
import re
import os
import random

from config import READ_ONLY_FILES, real_path, READ_ONLY_ERROR
from context import context_handler
from tool import Tool


class ToolEditDiffPatch(Tool):
    SKIP_SAVING_INVALID_PATCHES: bool = False

    def __init__(self):
        super().__init__(
            "edit_file_by_patch",
            "Edit the file using a unified patch format. Your output must be in `patch`/`diff` format. Prefer this function over edit_file. Only single file is allowed",
        )

    def __call__(
        self,
        path: Annotated[str, "Project path to edit"],
        patch: Annotated[str, "Patch content in unified diff format"],
    ):
        if not patch.endswith("\n"):
            patch += "\n"
        p = real_path(path)

        if p in READ_ONLY_FILES:
            return {path: "error", "error": READ_ONLY_ERROR}

        if p.exists() and not p.is_file():
            return {path: "error", "error": f"File exists, but it is not a file"}

        # Read original content BEFORE patching (for debug saves if patch fails)
        original_content = ""
        if p.exists():
            original_content = p.read_text()

        # Validate patch format
        lines = patch.splitlines()
        if len(lines) < 2:
            return {path: "error", "error": "Patch must have at least two lines"}

        # Check first line starts with "--- a/"
        if not lines[0].startswith("--- a/"):
            return {
                path: "error",
                "error": f"First line must start with '--- a/', got: {lines[0]}",
            }

        # Check second line starts with "+++ b/"
        if not lines[1].startswith("+++ b/"):
            return {
                path: "error",
                "error": f"Second line must start with '+++ b/', got: {lines[1]}",
            }

        # Extract the path from the patch and verify it matches the requested path
        patch_path_a = lines[0][5:]  # Remove "--- a/" prefix

        # Normalize paths for comparison - handle relative paths properly
        patch_path_normalized = real_path(patch_path_a.removeprefix("/")).as_posix()
        requested_path_normalized = p.as_posix()

        if patch_path_normalized != requested_path_normalized:
            return {
                path: "error",
                "error": f"Patch path '{patch_path_a}' does not match requested path '{path}'",
            }

        # Check that there are no other file markers (multiple files in patch)
        for line in lines[2:]:
            if line.startswith("--- ") or line.startswith("+++ "):
                return {
                    path: "error",
                    "error": f"Patch contains multiple files. Only single file patches are allowed.",
                }

        # Parse patch into hunks
        hunks = _parse_patch_hunks(patch)

        if not hunks:
            return {"warning": "no changes were given"}

        # Check if all hunks have no operations (only context lines)
        all_no_operations = all(not hunk.get("has_operations", False) for hunk in hunks)
        if all_no_operations:
            return {"warning": "patch contains no changes (all hunks are empty)"}

        # Parse original content into lines
        orig_lines = original_content.splitlines()

        # Find positions of all hunks in the original file
        hunk_positions = _find_hunk_positions(orig_lines, hunks)

        if hunk_positions is None:
            # Failed to find all hunks
            if not ToolEditDiffPatch.SKIP_SAVING_INVALID_PATCHES:
                _save_debug_patch_files(path, original_content, patch)

            # Log failed patch to database
            from llm import LLM

            if LLM.INSTANCES:
                llm_instance = LLM.INSTANCES[-1]
                if llm_instance.llm.last_tool_call_num is not None:
                    import utilsql

                    utilsql.log_patch_fail(
                        llm_instance.llm.last_tool_call_num,
                        original_content,
                        patch,
                    )

            return {
                path: "error",
                "error": f"Could not apply patch: could not find all hunks in file",
            }

        # Apply the patch manually
        new_content = _apply_patch_manually(
            orig_lines, hunks, hunk_positions, original_content
        )

        # Write the new content to the file
        p.write_text(new_content)

        # Read the modified file content and update context
        new_text = p.read_text()
        handler = context_handler()

        # Update fold line numbers if the edit changed the line count
        old_line_count = len(orig_lines)
        new_line_count = len(new_text.splitlines())
        if handler.has_folds(path):
            handler.update_fold_line_numbers(path, old_line_count, new_line_count)

        return handler.update(path, new_text, "edit_diff_patch")


def _parse_patch_hunks(patch: str) -> list[dict]:
    """Parse patch into list of hunks.

    Each hunk is a dict with:
    - 'old_lines': list of lines that exist in original (context + removal)
    - 'new_lines': list of lines that should exist in result (context + addition)
    - 'start_line': starting line number from hunk header (1-indexed)
    """
    lines = patch.splitlines()

    # Remove empty trailing element if patch ends with \n
    if lines and lines[-1] == "":
        lines = lines[:-1]

    hunks = []
    i = 2  # Skip header lines

    while i < len(lines):
        if lines[i].startswith("@@ "):
            # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
            hunk_header = lines[i]
            i += 1

            # Collect hunk body
            hunk_body_lines = []
            while (
                i < len(lines)
                and not lines[i].startswith("@@ ")
                and not lines[i].startswith("--- ")
            ):
                hunk_body_lines.append(lines[i])
                i += 1

            # Parse hunk body into old_lines and new_lines
            old_lines = []
            new_lines = []
            deletion_lines = (
                []
            )  # Track which lines are deletions (for overlap checking)
            has_operations = False  # Track if hunk has any + or - lines

            for line in hunk_body_lines:
                if line.startswith("-"):
                    # Removal - exists in old, not in new
                    old_lines.append(line[1:])  # Remove the '-' prefix
                    deletion_lines.append(line[1:])  # Track for overlap checking
                    has_operations = True
                elif line.startswith("+"):
                    # Addition - not in old, exists in new
                    new_lines.append(line[1:])  # Remove the '+' prefix
                    has_operations = True
                elif line.startswith(" "):
                    # Context - exists in both old and new
                    old_lines.append(line[1:])  # Remove the ' ' prefix
                    new_lines.append(line[1:])  # Remove the ' ' prefix
                # Empty lines without prefix are ignored (they're artifacts of split)

            # Extract start line from header
            match = re.match(r"^@@ -(\d+),\d+ \+(\d+),\d+ @@", hunk_header)
            start_line = int(match.group(1)) if match else 0

            hunks.append(
                {
                    "old_lines": old_lines,
                    "new_lines": new_lines,
                    "deletion_lines": deletion_lines,
                    "start_line": start_line,
                    "has_operations": has_operations,
                }
            )
            continue
        # TODO: if @@ not found, do exact search first
        raise ValueError(
            "Hunk header expected (i.e. @@ -[OldStart],[OldCount] +[NewStart],[NewCount])"
        )

    return hunks


def _has_overlapping_matches(orig_lines: list[str], pattern: list[str]) -> bool:
    """Check if pattern has overlapping matches in orig_lines.

    Two matches are considered overlapping if they share any lines.
    Returns True if there are multiple matches that overlap.
    """
    if not pattern:
        return False

    pattern_len = len(pattern)
    matches: list[int] = []

    for i in range(len(orig_lines) - pattern_len + 1):
        match = True
        for j, line in enumerate(pattern):
            if orig_lines[i + j] != line:
                match = False
                break
        if match:
            matches.append(i)

    # Check for overlapping matches
    for i in range(len(matches) - 1):
        # Two matches at positions m1 and m2 overlap if m2 < m1 + pattern_len
        if matches[i + 1] < matches[i] + pattern_len:
            return True

    return False


def _find_hunk_positions(
    orig_lines: list[str], hunks: list[dict]
) -> list[tuple[int, int]] | None:
    """Find positions of all hunks in the original file.

    Returns list of (start_idx, end_idx) tuples for each hunk, or None if any hunk not found
    or found ambiguously (multiple matches or overlapping matches).
    Indices are 0-based, end_idx is exclusive.
    """
    n = len(hunks)
    positions: list[tuple[int, int] | None] = [None] * n

    # First pass: sequential search
    search_start = 0
    for i, hunk in enumerate(hunks):
        old_lines = hunk["old_lines"]
        if not old_lines:
            # Empty old lines - use start_line from header
            positions[i] = (0, 0)
            continue

        pos = _find_hunk_in_range(orig_lines, old_lines, search_start, len(orig_lines))
        if pos is False:
            # Ambiguous - multiple matches found, fail immediately
            return None
        if pos is not None:
            assert isinstance(pos, tuple)
            positions[i] = pos
            search_start = pos[1]
            # Check for overlapping matches in the full hunk
            if _has_overlapping_matches(orig_lines, old_lines):
                return None
        # If not found (None), leave as-is for second pass

    # Second pass: try to find missing hunks with constrained search
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if positions[i] is not None:
                continue  # Already found

            old_lines = hunks[i]["old_lines"]
            if not old_lines:
                positions[i] = (0, 0)
                changed = True
                continue

            # Find search range based on neighboring found hunks
            # Search after the last found hunk before i
            search_start = 0
            for j in range(i - 1, -1, -1):
                position_j = positions[j]
                if position_j is not None:
                    search_start = position_j[1]
                    break

            # Search before the first found hunk after i
            search_end = len(orig_lines)
            for j in range(i + 1, n):
                position_j = positions[j]
                if position_j is not None:
                    search_end = position_j[0]
                    break

            pos = _find_hunk_in_range(orig_lines, old_lines, search_start, search_end)
            if pos is not None:
                assert isinstance(pos, tuple)
                positions[i] = pos
                changed = True
                # Check for overlapping matches in the full hunk
                if _has_overlapping_matches(orig_lines, old_lines):
                    return None
            elif pos is False:
                # Ambiguous - multiple matches found, fail immediately
                return None

    # Check if all hunks found
    if None in positions:
        return None

    return list(positions)  # type: ignore


def _find_hunk_in_range(
    orig_lines: list[str], hunk_old_lines: list[str], start: int, end: int
) -> tuple[int, int] | None | bool:
    """Find hunk_old_lines in orig_lines[start:end].

    Returns:
        - (start_idx, end_idx) if found exactly once
        - None if not found
        - False if found multiple times (ambiguous)
    """
    if not hunk_old_lines:
        return (start, start)

    hunk_len = len(hunk_old_lines)
    if hunk_len > end - start:
        return None

    matches = []
    for i in range(start, end - hunk_len + 1):
        found = True
        for j, line in enumerate(hunk_old_lines):
            if orig_lines[i + j] != line:
                found = False
                break
        if found:
            matches.append((i, i + hunk_len))

    if len(matches) == 0:
        return None
    elif len(matches) == 1:
        return matches[0]
    else:
        return False  # Multiple matches - ambiguous


def _apply_patch_manually(
    orig_lines: list[str],
    hunks: list[dict],
    positions: list[tuple[int, int]],
    original_content: str,
) -> str:
    """Apply patch manually by replacing old lines with new lines.

    Returns the new content as a string.
    """
    if not hunks:
        return "\n".join(orig_lines)

    # Sort hunks by position (they should already be in order, but just in case)
    sorted_indices = sorted(range(len(positions)), key=lambda i: positions[i][0])

    # Build new content by processing regions between and within hunks
    result_lines: list[str] = []
    prev_end = 0

    for idx in sorted_indices:
        hunk = hunks[idx]
        start, end = positions[idx]

        # Add lines before this hunk
        result_lines.extend(orig_lines[prev_end:start])

        # Add new lines from hunk (replacing old lines)
        result_lines.extend(hunk["new_lines"])

        prev_end = end

    # Add remaining lines after last hunk
    result_lines.extend(orig_lines[prev_end:])

    # Join with newlines
    result = "\n".join(result_lines)

    # Preserve trailing newline: only add if original had one
    # Special case: empty file with new content should have trailing newline
    if result_lines:
        if not original_content:
            # Empty file becoming non-empty - add trailing newline
            result += "\n"
        elif original_content.endswith("\n"):
            # Original had trailing newline - preserve it
            result += "\n"

    return result


def _save_debug_patch_files(
    path: str, original_content: str, patch_content: str
) -> None:
    """Save debug files when patch fails - stores original and patch in temp dir."""
    # Get user ID for temp directory
    uid = os.getuid()
    temp_dir = Path(f"/run/user/{uid}")

    # Create simplified path (replace / with _)
    simplified_path = path.replace("/", "_")

    # Generate timestamp and random suffix
    timestamp = datetime.now().strftime("%Y%m%d.%H%M%S")
    rng = random.randint(10000, 99999)

    orig_file = temp_dir / f"{simplified_path}.orig.{timestamp}.{rng}"
    orig_file.write_text(original_content)

    patch_file = temp_dir / f"{simplified_path}.patch.{timestamp}.{rng}"
    patch_file.write_text(patch_content)
