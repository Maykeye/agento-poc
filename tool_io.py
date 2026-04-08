from pathlib import Path
import glob
from typing import Annotated
import os
from config import real_path, READ_ONLY_FILES, READ_ONLY_ERROR, project_directory
from tool import Tool
import re
from context import ContextMode, context_handler

# TODO: import logging.basicConfig(level=logging.INFO)


class ToolLs(Tool):
    def __init__(self):
        super().__init__(
            name="ls",
            description="List files and subdirectories in a given directories. Supports glob patterns (e.g., *.rs, **/*.py)",
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

    def _is_glob_pattern(self, path: str) -> bool:
        """Check if path contains glob patterns."""
        return any(c in path for c in "*?[]")

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
            # Check if this is a glob pattern
            if self._is_glob_pattern(path):
                # Resolve the path relative to the project directory
                # glob.glob resolves relative paths from the current directory,
                # but we need to resolve from the project directory
                project_dir = project_directory()
                # Convert the pattern to an absolute path relative to project directory
                abs_pattern = project_dir / path

                # Use glob.glob with recursive=True for glob patterns
                matched = glob.glob(str(abs_pattern), recursive=True)

                if not matched:
                    out.append(f"Listing: {path} : No files found matching pattern")
                    continue

                # Build output for matched files
                buf = []
                for m in matched:
                    m_path = Path(m)
                    # Convert absolute path to relative path from project directory
                    rel_path = m_path.relative_to(project_dir)
                    if m_path.is_file():
                        desc = f" File: {rel_path}"
                        desc += f" (Size: {os.path.getsize(m_path)} bytes)"
                    elif m_path.is_dir():
                        desc = f" Directory: {rel_path}"
                    else:
                        desc = f" Path: {rel_path}"
                    buf.append(desc)
                buf.sort(key=lambda v: v.lower())

                out.append(
                    f"Listing: {path}\n" + "\n".join(buf) + f"\n({len(buf)} entries)"
                )
            else:
                # Original behavior for non-glob paths
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
            return {path: "error", "error": READ_ONLY_ERROR}

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
            return {path: "error", "error": READ_ONLY_ERROR}

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
            "search_replace_once",
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
            return {path: "error", "error": READ_ONLY_ERROR}

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


class ToolFoldAddImpl(Tool):
    def __init__(self):
        super().__init__(
            "file_add_fold_impl",
            "INTERNAL: Add a fold using line numbers (implementation class). Use ToolFoldAdd for regex-based folding instead.",
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


class ToolFoldAdd(Tool):
    def __init__(self):
        super().__init__(
            "file_add_fold",
            """Add a fold to hide file content in the LLM context using regex patterns.

SPECIFICATION:
- Use regex patterns to identify the start and end of the region to fold.
- Multiline patterns are supported and encouraged for more reliable matching.
- The start pattern must match exactly once in the VISIBLE (non-folded) content.
- The end pattern must match exactly once AFTER the start pattern match.
- The fold region (start..end) must not overlap with existing folds.

LINE NUMBER RULES:
- If the start pattern matches multiple lines, the fold starts from the FIRST matching line.
- If the end pattern matches multiple lines, the fold ends at the LAST matching line.

EXAMPLES:
1. Single line patterns:
   start_pattern="def my_function\\(", end_pattern="^    pass$"

2. Multiline patterns (recommended for reliability):
   start_pattern="def my_function\\([^)]*\\):\\s*\\n\\s*\"\"\"",
   end_pattern="\"\"\"\\s*\\n\\n"

When lines are folded, `FOLD: lines N..M (description)` will be displayed where N and M are real line numbers. Use them for counting further folding.""",
        )

    def __call__(
        self,
        path: Annotated[str, "Project path to add fold to"],
        start_pattern: Annotated[
            str,
            "Regex pattern to match the START of the region to fold. Must match exactly once in visible content. For multiline patterns, folding starts from the first matching line.",
        ],
        end_pattern: Annotated[
            str,
            "Regex pattern to match the END of the region to fold. Must match exactly once AFTER the start. For multiline patterns, folding ends at the last matching line.",
        ],
        name: Annotated[str, "Unique name/description for this fold"],
    ):

        handler = context_handler()

        # Check if file has been read
        from context.suffix import SUFFIX_CONTEXTS

        if path not in SUFFIX_CONTEXTS:
            return {
                path: "error",
                "error": f"File {path} has not been read. Read it first with read_file.",
            }

        # Get the full file content
        full_text = SUFFIX_CONTEXTS[path].text
        full_lines = full_text.splitlines()

        # Get visible content (with folds applied)
        visible_text = handler.format_folded_content(path, full_text)

        # Compile patterns
        try:
            start_re = re.compile(start_pattern, re.MULTILINE)
        except re.error as e:
            return {path: "error", "error": f"Invalid start_pattern regex: {e}"}

        try:
            end_re = re.compile(end_pattern, re.MULTILINE)
        except re.error as e:
            return {path: "error", "error": f"Invalid end_pattern regex: {e}"}

        # Find start pattern in visible content
        start_matches = list(start_re.finditer(visible_text))
        if len(start_matches) == 0:
            return {
                path: "error",
                "error": f"start_pattern `{start_pattern}` not found in visible content",
            }
        if len(start_matches) > 1:
            return {
                path: "error",
                "error": f"start_pattern `{start_pattern}` matches {len(start_matches)} times in visible content (must match exactly once)",
            }
        start_match = start_matches[0]

        # Find end pattern AFTER start match in visible content
        search_after_start = visible_text[start_match.start() :]
        end_matches = list(end_re.finditer(search_after_start))
        if len(end_matches) == 0:
            return {
                path: "error",
                "error": f"end_pattern `{end_pattern}` not found after start_pattern in visible content",
            }
        if len(end_matches) > 1:
            return {
                path: "error",
                "error": f"end_pattern `{end_pattern}` matches {len(end_matches)} times after start (must match exactly once)",
            }
        end_match = end_matches[0]

        # Calculate visible line numbers for start and end
        # Start: first line of start match
        start_visible_line = visible_text[: start_match.start()].count("\n") + 1

        # End: last line of end match (in the search_after_start context)
        # end_match.end() is relative to search_after_start
        absolute_end_pos = start_match.start() + end_match.end()
        end_visible_line = visible_text[:absolute_end_pos].count("\n") + 1

        # Map visible line numbers to actual file line numbers using context handler
        start_actual_line = handler.visible_to_actual(
            path, start_visible_line, full_text
        )
        end_actual_line = handler.visible_to_actual(path, end_visible_line, full_text)

        # Get the actual line content for validation
        start_actual_line_idx = start_actual_line - 1
        end_actual_line_idx = end_actual_line - 1

        if start_actual_line_idx < 0 or start_actual_line_idx >= len(full_lines):
            return {
                path: "error",
                "error": f"Calculated start line {start_actual_line} is out of bounds",
            }
        if end_actual_line_idx < 0 or end_actual_line_idx >= len(full_lines):
            return {
                path: "error",
                "error": f"Calculated end line {end_actual_line} is out of bounds",
            }

        start_line_content = full_lines[start_actual_line_idx]
        end_line_content = full_lines[end_actual_line_idx]

        # Call the implementation class with computed line numbers
        impl = ToolFoldAddImpl()
        return impl(
            path=path,
            fold_from_line_num=start_actual_line,
            fold_from_line=start_line_content,
            fold_to_line_num=end_actual_line,
            fold_to_line=end_line_content,
            name=name,
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
