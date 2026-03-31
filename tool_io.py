from pathlib import Path
from typing import Annotated
from pathlib import Path
import os
from config import real_path, READ_ONLY_FILES, READ_ONLE_ERROR
from tool import Tool
import re
from context import CONTEXTS, ContextMode, context_handler

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
        p.write_text(new_text)

        return context_handler().update(
            path, new_text, "edit_file", edit_chunk=(replace_from, replace_with)
        )
