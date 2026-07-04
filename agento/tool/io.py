from pathlib import Path
from typing import Annotated
import glob
import os
import re

from agento.config import real_path, READ_ONLY_ERROR, CONFIG
from agento.context import ContextMode, context_handler
from agento.tool import Tool

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
                # Convert the pattern to an absolute path relative to project directory
                abs_pattern = CONFIG.project_directory / path

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
                    rel_path = m_path.relative_to(CONFIG.project_directory)
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
        p = real_path(path)
        old_text = p.read_text().removesuffix("\n") if p.exists() else ""
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

        if p in CONFIG.read_only_files:
            return {path: "error", "error": READ_ONLY_ERROR}

        if p.exists() and not p.is_file():
            return {path: "error", "error": f"File exists, but it is not a file"}

        p.write_text(text)
        return context_handler().update(path, text, "write_file")


class ToolDeleteFile(Tool):
    def __init__(self):
        super().__init__("delete_file", "Delete the file")

    def __call__(self, path: Annotated[str, "File to delete"]):
        p = real_path(path)
        if p in CONFIG.read_only_files:
            return {path: "error", "error": READ_ONLY_ERROR}

        if p.exists() and not p.is_file():
            return {path: "error", "error": f"File exists, but it is not a file"}

        if not p.exists():
            return {path: "error", "error": f"File does't exist"}

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


class ToolSearchReplaceOnce(Tool):
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
        if p in CONFIG.read_only_files:
            return {path: "error", "error": READ_ONLY_ERROR}

        if p.exists() and not p.is_file():
            return {path: "error", "error": f"File exists, but it is not a file"}

        old_text = p.read_text()
        handler = context_handler()

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

        # Update the context with new content
        return handler.update(
            path, new_text, "edit_file", edit_chunk=(replace_from, replace_with)
        )


class ToolRename(Tool):
    def __init__(self):
        super().__init__(
            "rename_file",
            "Rename(or move) a file from path_src to path_dst. Checks that path_src exists and is a file (not directory), and that path_dst does not exist.",
        )

    def __call__(
        self,
        path_src: Annotated[
            str, "Source file path (must exist and be a file, not directory)"
        ],
        path_dst: Annotated[str, "Destination file path (must not exist)"],
    ):

        p_src = real_path(path_src)
        p_dst = real_path(path_dst)

        # Check that path_src exists and is a file (not directory)
        if not p_src.exists():
            return {path_src: "error", "error": f"Source file {path_src} doesn't exist"}
        if not p_src.is_file():
            return {path_src: "error", "error": f"Source path {path_src} is not a file"}

        # Check that path_dst does not exist
        if p_dst.exists():
            return {
                path_dst: "error",
                "error": f"Destination path {path_dst} already exists",
            }

        # Perform the rename
        p_src.rename(p_dst)

        # Update context with LLM for suffix mode message updates
        return context_handler().update(path_src, path_dst, "rename")
