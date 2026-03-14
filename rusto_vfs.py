from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated
from pathlib import Path
import logging
import os

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

PROJECT_DIRECTORY = Path(".")
""" Home project directory, a "root" of the project """


def set_project_directory(project: str | Path):
    """Setup project directory (will be resolved to absolute, expanduser() yaself)"""
    global PROJECT_DIRECTORY
    PROJECT_DIRECTORY = Path(project).absolute()
    logging.info(f"New project dir: {PROJECT_DIRECTORY}")


set_project_directory("./.project/")


def real_path(project_path: str | Path):
    path = PROJECT_DIRECTORY.joinpath(project_path).resolve()
    if path.is_relative_to(PROJECT_DIRECTORY):
        return path
    else:
        raise ValueError(f"{project_path} is out of bounds of project directory")


def read_file(path: Annotated[str, "Project path to read from"]):
    """Read a file. If file read ok, its content will be returned as plain-text between `=== CONTENT START ===` and `=== CONTENT END ===`"""
    p = real_path(path)

    if not p.exists():
        return {path: "error", "error": f"File {path} doesn't exist"}
    if not p.is_file():
        return {path: "error", "error": f"Is not a file"}

    text = p.read_text()
    return f">>> OK: READ {path}\n>>> === CONTENT START ===\n{text}\n>>> === CONTENT END ==="


def write_file(
    path: Annotated[str, "Project path to write to"],
    content: Annotated[str, "Content to write"],
):
    """Write the file. If file didn't exist, it will be created with the given content. If file existed, its content will be replaced with given content"""
    p = real_path(path)

    if p.exists() and not p.is_file():
        return {path: "error", "error": f"File exists, but it is not a file"}

    p.write_text(content)
    sz = os.path.getsize(p)
    first_line = content.splitlines()[0] if content else ""
    return f">>> OK: WRITTEN {path} ({sz} bytes) \n>>> FIRST LINE WRITTEN: {first_line}"


def edit_file(
    path: Annotated[str, "Project path to edit"],
    replace_from: Annotated[str, "Text to remove"],
    replace_with: Annotated[str, "Text insert instead"],
):
    """Edit existing file. Finds `replace_from` and replaces the it with `replace_with`. Exactly one instance of `replace_from` must exist or error will occur"""
    p = real_path(path)

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
    return {
        path: "ok",
        "desc": f"Chunk replaced from `{repr(replace_from[:32])}`... to `{repr(replace_with[:32])}`...",
    }


# print(split_file("src/main.rs"))
