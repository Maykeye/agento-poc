from datetime import datetime
from typing import Annotated
from pathlib import Path
import re
import os
import random

from config import READ_ONLY_FILES, real_path, READ_ONLY_ERROR
from context import context_handler
from tool import Tool, run_executable


class ToolEditDiffPatch(Tool):
    SKIP_SAVING_INVALID_PATCHES = False

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
        # If patch path starts with /, strip it; if not, prepend project dir
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

        # Fix hunk line counts before applying patch
        fixed_patch = _fix_patch_hunk_counts(patch)

        if not fixed_patch:
            return {"warning": "no changes were given"}

        # Run patch directly on the file (don't use -p, just pass the file)
        result = run_executable(
            ["patch", "--reject-file=-", "--no-backup", "-u", str(p)],
            stdin_text=fixed_patch,
        )

        if result.get("exitcode") != 0:
            if not ToolEditDiffPatch.SKIP_SAVING_INVALID_PATCHES:
                _save_debug_patch_files(path, original_content, patch)
            return {
                path: "error",
                "error": f"Patch failed with exit code {result.get('exitcode')}",
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
            }

        # Read the modified file content and update context
        new_text = p.read_text()
        handler = context_handler()

        # Update fold line numbers if the edit changed the line count
        old_line_count = len(patch.splitlines())  # This is just for context
        new_line_count = len(new_text.splitlines())
        if handler.has_folds(path):
            handler.update_fold_line_numbers(path, old_line_count, new_line_count)

        return handler.update(path, new_text, "edit_diff_patch")


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


def _fix_patch_hunk_counts(patch: str) -> str:
    # Split into lines but preserve the trailing newline
    lines = patch.splitlines()

    # Remove empty trailing element if patch ends with \n
    if lines and lines[-1] == "":
        lines = lines[:-1]

    if len(lines) < 2:
        return patch

    result_lines = []
    i = 0
    total_lines = len(lines)

    # Copy header lines (first two lines: --- a/... and +++ b/...)
    while i < total_lines and not lines[i].startswith("@@ "):
        result_lines.append(lines[i])
        i += 1

    # Process hunks - track if we have any non-empty hunks
    has_non_empty_hunk = False

    while i < total_lines:
        if lines[i].startswith("@@ "):
            # Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
            hunk_header = lines[i]
            i += 1

            has_change = False

            # Collect and count lines in the hunk body
            old_count = 0
            new_count = 0
            hunk_body_lines = []

            # Process hunk body until we hit next hunk or end of file
            while (
                i < total_lines
                and not lines[i].startswith("@@ ")
                and not lines[i].startswith("--- ")
            ):
                line = lines[i]
                if line.startswith("-"):
                    old_count += 1
                    has_change = True
                elif line.startswith("+"):
                    new_count += 1
                    has_change = True
                elif line.startswith(" "):
                    old_count += 1
                    new_count += 1
                # Empty lines or special lines don't count but are kept
                hunk_body_lines.append(line)
                i += 1

            # Only add hunk if it has actual changes (not just context lines)
            if has_change:
                has_non_empty_hunk = True
                # First rebalance prefix and suffix, then fix counts
                hunk_body_lines = _rebalance_hunk_prefix_suffix(
                    hunk_body_lines, hunk_header
                )
                old_count = sum(
                    1 for l in hunk_body_lines if l.startswith("-") or l.startswith(" ")
                )
                new_count = sum(
                    1 for l in hunk_body_lines if l.startswith("+") or l.startswith(" ")
                )
                fixed_header = _fix_hunk_header(hunk_header, old_count, new_count)
                result_lines.append(fixed_header)
                result_lines.extend(hunk_body_lines)

        else:
            # Any remaining lines after last hunk
            result_lines.append(lines[i])
            i += 1

    is_empty = not has_non_empty_hunk
    if is_empty:
        return ""
    return "\n".join(result_lines) + "\n"


def _rebalance_hunk_prefix_suffix(
    hunk_body_lines: list[str], hunk_header: str
) -> list[str]:
    """Rebalance prefix and suffix context lines in a hunk body

    For non-BOF hunks, count unchanged lines in prefix and suffix and remove extras.
    """
    # Parse hunk header to get old_start position
    pattern = r"^@@ -(\d+),\d+ \+(\d+),\d+ @@(.*)$"
    match = re.match(pattern, hunk_header)

    if not match:
        return hunk_body_lines

    # Find the position where changed lines start and end
    # Prefix context comes first, then changes, then suffix context
    change_start = None
    change_end = 0

    for i, line in enumerate(hunk_body_lines):
        if line.startswith("-") or line.startswith("+"):
            if change_start is None:
                change_start = i
            change_end = i

    # If no changes found, return as-is
    if change_start is None:
        return hunk_body_lines

    # Count prefix context lines (before any changes)
    prefix_context_count = change_start

    # Count suffix context lines (after all changes)
    suffix_context_count = len(hunk_body_lines) - 1 - change_end

    # If counts are equal, no rebalancing needed
    if prefix_context_count == suffix_context_count:
        return hunk_body_lines

    # Need to rebalance - remove extras from the longer side
    # Remove from the end of the longer side (keeping more meaningful context)
    if prefix_context_count > suffix_context_count:
        # Remove extra prefix context lines from the beginning
        excess = prefix_context_count - suffix_context_count
        result = hunk_body_lines[excess:]
    else:
        # Remove extra suffix context lines from the end
        excess = suffix_context_count - prefix_context_count
        result = hunk_body_lines[: len(hunk_body_lines) - excess]

    return result


def _fix_hunk_header(header: str, old_count: int, new_count: int) -> str:
    """Replace old and new counts in hunk header while preserving start positions.

    Format: @@ -old_start,old_count +new_start,new_count @@ optional context
    """
    # Pattern to match hunk header: @@ -old_start,old_count +new_start,new_count @@
    pattern = r"^@@ -(\d+),\d+ \+(\d+),\d+ @@(.*)$"
    match = re.match(pattern, header)

    if match:
        old_start = match.group(1)
        new_start = match.group(2)
        rest = match.group(3)  # Optional context after @@
        return f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{rest}"

    # If pattern doesn't match, return header as-is (shouldn't happen for valid patches)
    return header
