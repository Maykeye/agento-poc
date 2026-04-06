from datetime import datetime
from typing import Annotated
from pathlib import Path
import os
import random

from config import READ_ONLY_FILES, real_path, READ_ONLY_ERROR
from context import context_handler
from tool import Tool, run_executable


class ToolEditDiffPatch(Tool):
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

        # Run patch directly on the file (don't use -p, just pass the file)
        result = run_executable(
            ["patch", "--reject-file=-", "--no-backup", "-u", str(p)], stdin_text=patch
        )

        if result.get("exitcode") != 0:
            # Save debug files when patch fails
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
