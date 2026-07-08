from typing import Annotated

from agento.config import real_path, READ_ONLY_ERROR, CONFIG
from agento.context import context_handler
from agento.tool import Tool


class ToolReadFile(Tool):
    def __init__(self):
        description = "Read a file and print its content into the buffer."
        super().__init__(name="read_file", description=description)

    def __call__(self, path: Annotated[str, "Project path to read from"]):
        # Todo: range
        p = real_path(path)

        if not p.exists():
            return {path: "error", "error": f"File {path} doesn't exist"}
        if not p.is_file():
            return {path: "error", "error": f"Is not a file"}

        text = p.read_text()
        return context_handler().update(path, text, "read_file")


class ToolAppend(Tool):
    # Todo: prepend
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


class ToolSearchReplaceOnce(Tool):
    def __init__(self):
        super().__init__(
            "edit_search_replace_once",
            "Edit existing file. Finds `replace_from` argument and replaces the it with `replace_with` argument. Exactly one instance of `replace_from` must exist or error will occur",
        )

    def __call__(
        self,
        path: Annotated[str, "Project path to file"],
        replace_from: Annotated[str, "Text to replace from"],
        replace_with: Annotated[str, "Text to replace with"],
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
