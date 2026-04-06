from pathlib import Path
from typing import Annotated
import os
from config import real_path, READ_ONLY_FILES, READ_ONLE_ERROR
from tool import Tool, run_executable
import re
from context import ContextMode, context_handler

# TODO: import logging.basicConfig(level=logging.INFO)


class ToolLs(Tool):
    def __init__(self):
        super().__init__(
            name="ls",
            description="List files and subdirectories in a given directories",
        )

    def _dumb_gitignore_filter(self):
        """Very simple .gitignore reader. Cut empty lines and comments"""
        p = real_path(Path(".gitignore"))
        if not p.exists():
            return []
        lines = p.read_text().splitlines()
        lines = [re.sub("^/", "./", x.lstrip()) for x in lines]
        lines = [re.sub(r"#.*", "", x).strip() for x in lines]
        lines = [real_path(x) for x in lines]
        return lines

    def __call__(
        self,
        paths: Annotated[
            list,
            "List of paths to get listing from(use [] or ['.'] for listing base project path)",
        ],
    ):
        if not paths:
            paths = ["."]

        out = []

        for path in paths:
            p = real_path(path)
            if p.exists() and not p.is_dir():
                out.append(f"Listing: {path} : Path exists, but is not a directory")
                continue
            if not p.exists():
                out.append(f"Listing: {path} : Path does not exist")
                continue

            # We'll excldue listing of files included in .gitignore
            exclude = self._dumb_gitignore_filter()
            entries = [x for x in p.glob("*")]

            buf = []
            if p != real_path("."):
                buf.append(" Directory: ..")
            for entry in entries:
                if entry.absolute() in exclude:
                    continue

                desc = ""
                if entry.is_file():
                    desc += " File: "

                if entry.is_dir():
                    desc += " Directory: "

                desc += entry.name
                if entry.is_file():
                    desc += f" (Size: {os.path.getsize(entry)} bytes)"
                buf.append(desc)
            buf.sort(key=lambda v: v.lower())

            out.append(
                f"Listing: {path}\n" + "\n".join(buf) + f"\n({len(buf)} entries)"
            )
        return "\n\n".join(out)


class ToolReadFile(Tool):
    def __init__(self):
        description = "Read a file. If file was read ok, its content will be returned as plain-text between `=== CONTENT START ===` and `=== CONTENT END ===`"
        if context_handler().mode() == ContextMode.RAW:
            description = "Read a file. If file was read OK, it will be placed into the files context."
        super().__init__(name="read_file", description=description)

    def __call__(self, path: Annotated[str, "Project path to read from"]):
        p = real_path(path)

        if not p.exists():
            return {path: "error", "error": f"File {path} doesn't exist"}
        if not p.is_file():
            return {path: "error", "error": f"Is not a file"}

        text = p.read_text()
        return context_handler().update(path, text, "read_file")


class ToolAppend(Tool):
    def __init__(self):
        super().__init__(
            "append_to_file",
            "Append text to the end of the file, leaving old as is",
        )

    def __call__(
        self,
        path: Annotated[str, "Project path to append to"],
        text: Annotated[str, "New content to append"],
    ):
        write = ToolWriteFile()
        old_text = real_path(path).read_text().removesuffix("\n")
        full_text = old_text + "\n" + text.removeprefix("\n")
        return write(path, full_text)


class ToolWriteFile(Tool):
    def __init__(self):
        super().__init__(
            "write_file",
            "Write the file. If file didn't exist, it will be created with the given content. If file existed, its content will be replaced with given content",
        )

    def __call__(
        self,
        path: Annotated[str, "Project path to write to"],
        text: Annotated[str, "Content to write"],
    ):
        p = real_path(path)

        if p in READ_ONLY_FILES:
            return {path: "error", "error": READ_ONLE_ERROR}

        # Check if there are any folds on this file
        handler = context_handler()
        if handler.has_folds(path):
            return {
                path: "error",
                "error": f"Cannot write file with active folds. Use file_unfold or file_unfold_all first.",
            }

        if p.exists() and not p.is_file():
            return {path: "error", "error": f"File exists, but it is not a file"}

        p.write_text(text)
        return context_handler().update(path, text, "write_file")


class ToolDeleteFile(Tool):
    def __init__(self):
        super().__init__("delete_file", "Delete the file")

    def __call__(self, path: Annotated[str, "File to delete"]):
        p = real_path(path)
        if p in READ_ONLY_FILES:
            return {path: "error", "error": READ_ONLE_ERROR}

        if p.exists() and not p.is_file():
            return {path: "error", "error": f"File exists, but it is not a file"}

        if not p.exists():
            return {path: "error", "error": f"File does't exist"}

        ToolUnfoldAll()(path)
        p.unlink()
        return context_handler().update(path, "(file deleted)", "delete_file")


class ToolMkDir(Tool):
    def __init__(self):
        super().__init__(
            "mkdir",
            "Create a new directory with parents(i.e. if creation of `foo/bar/` is asked, but `foo/` doesnt't exist, it will be created so `bar` can be created inside)",
        )

    def __call__(self, path: Annotated[str, "Directory to make"]):
        p = real_path(path)

        if p.exists() and p.is_dir():
            return {path: "directory", "info": f"directory already exists"}

        if p.exists():
            return {path: "error", "error": f"exists but not a directory"}

        p.mkdir(parents=True, exist_ok=True)
        return {path: "created", "info": f"Created directory"}


class ToolRmDir(Tool):
    def __init__(self):
        super().__init__(
            "rmdir",
            "Remove the empty directory. Directory must be empty or error will occur",
        )

    def __call__(self, path: Annotated[str, "File to delete"]):
        p = real_path(path)

        if p.exists() and not p.is_dir():
            return {path: "error", "error": f"Path exists, but it is not a directory"}
        if not p.exists():
            return {path: "error", "error": f"File does't exist"}

        data = [x for x in p.glob("*")]
        if data:
            return {path: "error", "error": f"Directory is not empty"}

        p.rmdir()
        return {path: "ok", "desc": f"Folder deleted"}


class ToolEditFile(Tool):
    def __init__(self):
        super().__init__(
            "edit_file",
            "Edit existing file. Finds `replace_from` and replaces the it with `replace_with`. Exactly one instance of `replace_from` must exist or error will occur",
        )

    def __call__(
        self,
        path: Annotated[str, "Project path to edit"],
        replace_from: Annotated[str, "Text to remove"],
        replace_with: Annotated[str, "Text insert instead"],
    ):
        p = real_path(path)
        if p in READ_ONLY_FILES:
            return {path: "error", "error": READ_ONLE_ERROR}

        if p.exists() and not p.is_file():
            return {path: "error", "error": f"File exists, but it is not a file"}

        old_text = p.read_text()
        handler = context_handler()

        # If file has folds, we need to check uniqueness in visible content only
        if handler.has_folds(path):
            # Validate that the edit target is visible and unique in visible content
            is_valid, error_msg = handler.validate_edit_in_visible_content(
                path, replace_from
            )
            if not is_valid:
                return {path: "error", "error": error_msg}

            # Text exists exactly once in visible content - proceed with edit
            # First, check if text exists in the actual file (already validated above)
            idx1 = old_text.find(replace_from)
            if idx1 == -1:
                return {path: "error", "error": f"Can not find `{repr(replace_from)}`"}

            # Replace only the first occurrence (which is in visible content)
            new_text = old_text.replace(replace_from, replace_with, 1)
        else:
            # No folds - use traditional uniqueness check
            idx1 = old_text.find(replace_from)
            if idx1 == -1:
                return {path: "error", "error": f"Can not find `{repr(replace_from)}`"}
            idx2 = old_text.find(replace_from, idx1 + len(replace_from))
            if idx2 != -1:
                return {
                    path: "error",
                    "error": f"`{repr(replace_from)}` must exists exactly once",
                }
            new_text = old_text.replace(replace_from, replace_with)

        # Write the new text to the file
        p.write_text(new_text)

        # Update fold line numbers if the edit changed the line count
        old_line_count = len(old_text.splitlines())
        new_line_count = len(new_text.splitlines())
        if handler.has_folds(path):
            handler.update_fold_line_numbers(path, old_line_count, new_line_count)

        # Update the context with new content
        return handler.update(
            path, new_text, "edit_file", edit_chunk=(replace_from, replace_with)
        )


class ToolFoldAdd(Tool):
    def __init__(self):
        super().__init__(
            "file_add_fold",
            "Add a fold to hide file content in the LLM context. Specify line numbers to fold a range of lines. The line text arguments help validate correct line counting.",
        )

    def __call__(
        self,
        path: Annotated[str, "Project path to add fold to"],
        fold_from_line_num: Annotated[
            int, "Line number to start fold from (1-indexed)"
        ],
        fold_from_line: Annotated[
            str, "Textual representation of line fold_from_line_num (for validation)"
        ],
        fold_to_line_num: Annotated[int, "Line number to end fold at (1-indexed)"],
        fold_to_line: Annotated[
            str, "Textual representation of line fold_to_line_num (for validation)"
        ],
        name: Annotated[str, "Unique name/description for this fold"],
    ):
        return context_handler().add_fold(
            path,
            fold_from_line_num,
            fold_from_line,
            fold_to_line_num,
            fold_to_line,
            name,
        )


class ToolUnfold(Tool):
    def __init__(self):
        super().__init__(
            "file_unfold",
            "Remove a specific fold by name from a file.",
        )

    def __call__(
        self,
        path: Annotated[str, "Project path to unfold"],
        name: Annotated[str, "Name of the fold to remove"],
    ):
        return context_handler().unfold(path, name)


class ToolUnfoldAll(Tool):
    def __init__(self):
        super().__init__(
            "file_unfold_all",
            "Remove all folds from a file.",
        )

    def __call__(
        self,
        path: Annotated[str, "Project path to unfold all"],
    ):
        handler = context_handler()

        result = handler.unfold_all(path)
        return result


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
            return {path: "error", "error": READ_ONLE_ERROR}

        if p.exists() and not p.is_file():
            return {path: "error", "error": f"File exists, but it is not a file"}

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
        result = run_executable(["patch", "-u", str(p)], stdin_text=patch)

        if result.get("exitcode") != 0:
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
