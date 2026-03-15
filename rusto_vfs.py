from pathlib import Path
from typing import Annotated
from pathlib import Path
import logging
import os
import subprocess

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


def empty_stdin():
    """Helper function to disable stdin in subprocesses"""
    return open("/dev/null")


def ls(path: Annotated[str, "Path to get listing(use '.' for listing base path)"]):
    """Read a file. If file read ok, its content will be returned as plain-text between `=== CONTENT START ===` and `=== CONTENT END ===`"""
    p = real_path(path)
    if p.exists() and not p.is_dir():
        return {path: "error", "error": f"Path exists, but it is not a directory"}
    if not p.exists():
        return {path: "error", "error": f"File does't exist"}

    entries = [x for x in p.glob("*")]
    entries.sort()

    out = []
    if p != real_path("."):
        out.append(" Directory: ..")
    for entry in entries:
        desc = ""
        if entry.is_file():
            desc += " File: "

        if entry.is_dir():
            desc += " Directory: "

        desc += entry.name
        if entry.is_file():
            desc += f" (Size: {os.path.getsize(entry)} bytes)"
        out.append(desc)

    return f"LISTING: {path}\n" + "\n".join(out) + f"\n({len(out)} entries)"


def read_file(path: Annotated[str, "Project path to read from"]):
    """Read a file. If file read ok, its content will be returned as plain-text between `=== CONTENT START ===` and `=== CONTENT END ===`"""
    p = real_path(path)

    if not p.exists():
        return {path: "error", "error": f"File {path} doesn't exist"}
    if not p.is_file():
        return {path: "error", "error": f"Is not a file"}

    text = p.read_text()
    return f">>> OK: READ {path}\n>>> === CONTENT START ===\n{text}\n>>> === CONTENT END ==="


def delete_file(path: Annotated[str, "File to delete"]):
    """Delete the file"""
    p = real_path(path)

    if p.exists() and not p.is_file():
        return {path: "error", "error": f"File exists, but it is not a file"}

    if not p.exists():
        return {path: "error", "error": f"File does't exist"}

    p.unlink()
    return {path: "ok", "desc": f"File deleted"}


def mkdir(path: Annotated[str, "Directory to make"]):
    """Create a new directory with parents(i.e. if creation of `foo/bar/` is asked, but `foo/` doesnt't exist, it will be created so `bar` can be created inside)"""
    p = real_path(path)

    if p.exists() and p.is_dir():
        return {path: "directory", "info": f"directory already exists"}

    if p.exists():
        return {path: "error", "error": f"exists but not a directory"}

    p.mkdir(parents=True, exist_ok=True)
    return {path: "created", "info": f"Created directory"}


def rmdir(path: Annotated[str, "File to delete"]):
    """Remove the empty directory. Directory must be empty or error will occur"""
    p = real_path(path)

    if p.exists() and not p.is_dir():
        return {path: "error", "error": f"Path exists, but it is not a directory"}
    if not p.exists():
        return {path: "error", "error": f"File does't exist"}

    data = [x for x in p.glob("*")]
    if data:
        return {path: "error", "error": f"Directory is not empty"}

    p.rmdir()
    return {path: "ok", "desc": f"File deleted"}


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
    lines = content.splitlines()
    first_line = lines[0] if content else ""
    return f">>> OK: WRITTEN {path} ({sz} bytes, {len(lines)} lines) \n>>> FIRST WRITTEN LINE: {first_line}"


def run_executable(args: list[str]):
    try:
        # TODO: tee me
        print(">>> RUN:", args)
        with empty_stdin() as stdin:
            p = subprocess.run(args, capture_output=True, text=True, stdin=stdin)
        return {"exitcode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except Exception as ex:
        print(ex)
        return {"error": str(ex)}


def run_cargo(cmd: str, args: list[str]):
    all_args = [
        "cargo",
        "--color",
        "never",
        cmd,
        "--manifest-path",
        str(PROJECT_DIRECTORY.joinpath("Cargo.toml")),
    ] + args
    return run_executable(all_args)


def run_git(args):
    return run_executable(["git", "-C", str(PROJECT_DIRECTORY)] + args)


def git_status():
    """Execute `git status` with no arguments"""
    return run_git(["status"])


def git_diff(args: Annotated[list, "arguments to pass AFTER git diff"]):
    """Execute `git diff with additional provided arguments (maybe an empty array)`"""
    return run_git(["diff"] + args)


def cargo_add_crate(args: Annotated[list, "Arguments that go after `cargo add`"]):
    """Execute `cargo add (args)` with additionaly provided args to add packages."""
    return run_cargo("add", args)


def cargo_check():
    """Execute `cargo check` to check if project(and its tests) compiles fine"""
    return run_cargo("check", ["--tests"])


def cargo_test(args: Annotated[list, "Arguments that go after `cargo test`"]):
    """Execute `cargo test (args)` with additionals provided args to test."""
    return run_cargo("test", args)


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
