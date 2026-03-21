from pathlib import Path
from typing import Annotated
from pathlib import Path
import os
import subprocess
import shlex
from tool import Tool
import re
import json

# TODO: import logging.basicConfig(level=logging.INFO)


def read_config(path: str):
    cfg_path = Path(path).expanduser()
    if not cfg_path.exists():
        raise ValueError(f"{path} doesn't exist, create json file there")
    text = cfg_path.read_text()
    try:
        config = json.loads(text)

        if not (project_directory := config.get("project_directory")):
            raise ValueError(f"{path}: project_directory is not defined")

        p = Path(project_directory)
        if not p.exists() or not p.resolve().is_dir():
            raise ValueError("not a directory")
        set_project_directory(project_directory)

    except Exception:
        raise ValueError(
            f'create {path} with {{"project_directory": "/path/to/existing/project_directory"}}'
        )


def rustfmt():
    """Run rust fmt"""
    p = shlex.quote(str(PROJECT_DIRECTORY))
    subprocess.run(["bash", "-c", f"cd {p}; rustfmt **/*.rs"])


PROJECT_DIRECTORY = Path(".")
""" Home project directory, a "root" of the project """


def set_project_directory(project: str | Path):
    """Setup project directory (will be resolved to absolute, expanduser() yaself)"""
    global PROJECT_DIRECTORY
    PROJECT_DIRECTORY = Path(project).absolute()
    print(f"New project dir: {PROJECT_DIRECTORY}")


READ_ONLY_FILES = []
READ_ONLE_ERROR = """FATAL ERRROR. 
You are NOT allowed to edit this file.
>>> ABORT EVERYTHING WHAT YOU ARE DOING AT ONCE! 
>>> INSTEAD EXPLAIN WHAT PART OF REQUEST MADE YOU TO TRY TO EDIT IT
"""


def make_file_readonly(path: str):
    READ_ONLY_FILES.append(real_path(path))


def reset_readonly_files():
    READ_ONLY_FILES.clear()


def real_path(project_path: str | Path):
    path = PROJECT_DIRECTORY.joinpath(project_path).resolve()
    if path.is_relative_to(PROJECT_DIRECTORY):
        return path
    else:
        raise ValueError(f"{project_path} is out of bounds of project directory")


def empty_stdin():
    """Helper function to disable stdin in subprocesses"""
    return open("/dev/null")


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


#################
class ToolReadFile(Tool):
    def __init__(self):
        super().__init__(
            name="read_file",
            description="Read a file. If file read ok, its content will be returned as plain-text between `=== CONTENT START ===` and `=== CONTENT END ===`",
        )

    def __call__(self, path: Annotated[str, "Project path to read from"]):
        p = real_path(path)

        if not p.exists():
            return {path: "error", "error": f"File {path} doesn't exist"}
        if not p.is_file():
            return {path: "error", "error": f"Is not a file"}

        text = p.read_text()
        return f">>> OK: READ {path}\n>>> === CONTENT START ===\n{text}\n>>> === CONTENT END ==="


class ToolWriteFile(Tool):
    def __init__(self):
        super().__init__(
            "write_file",
            "Write the file. If file didn't exist, it will be created with the given content. If file existed, its content will be replaced with given content",
        )

    def __call__(
        self,
        path: Annotated[str, "Project path to write to"],
        content: Annotated[str, "Content to write"],
    ):
        p = real_path(path)

        if p in READ_ONLY_FILES:
            return {path: "error", "error": READ_ONLE_ERROR}

        if p.exists() and not p.is_file():
            return {path: "error", "error": f"File exists, but it is not a file"}

        p.write_text(content)
        sz = os.path.getsize(p)
        lines = content.splitlines()
        first_line = lines[0] if content else ""
        return f">>> OK: WRITTEN {path} ({sz} bytes, {len(lines)} lines) \n>>> FIRST WRITTEN LINE: {first_line}"


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
        return {path: "ok", "desc": f"File deleted"}


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
        return {path: "ok", "desc": f"File deleted"}


class ToolRustApiInfo(Tool):
    def __init__(self):
        super().__init__(
            "brief_rust_api_info",
            "Returns information about API of the project: scans all .rs files and return info of used modules and what functions and structs there are with their docstrings, implementations of functions are omitted. Use this tool to know what struct or function where defined",
        )

    def __call__(self):
        result = run_executable(["rust-api-helper.sh", str(PROJECT_DIRECTORY)])
        if result.get("exitcode") != 0 or "stdout" not in result:
            return result
        stdout = result["stdout"].strip()
        return f"=== vvv API vvv ===\n{stdout}\n=== ^^^ API ^^^ ==="


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


class ToolGitStatus(Tool):
    def __init__(self):
        super().__init__("git_status", "Execute `git status` with no arguments")

    def __call__(self):
        return run_git(["status"])


class ToolGitAdd(Tool):
    def __init__(self):
        super().__init__(
            "git_add",
            "Execute `git add` with additional provided arguments",
        )

    def __call__(self, args: Annotated[list, "arguments tht go after `git diff`"]):
        return run_git(["add"] + args)


class ToolGitDiff(Tool):
    def __init__(self):
        super().__init__(
            "git_diff",
            "Execute `git diff` with additional provided arguments (maybe an empty array)",
        )

    def __call__(self, args: Annotated[list, "arguments tht go after `git diff`"]):
        return run_git(["diff"] + args)


class ToolCargoAdd(Tool):
    def __init__(self):
        super().__init__(
            "cargo_add",
            "Execute `cargo add (args)` with additionaly provided args to add packages.",
        )

    def __call__(self, args: Annotated[list, "arguments that go after `cargo add`"]):
        return run_cargo("add", args)


class ToolCargoCheck(Tool):
    def __init__(self):
        super().__init__(
            "cargo_check",
            "Execute `cargo check` to check if project(and its tests) compiles fine",
        )

    def __call__(self):
        return run_cargo("check", ["--tests"])


class ToolCargoTest(Tool):
    def __init__(self):
        super().__init__(
            "cargo_test",
            "Execute `cargo test (args)` with additionals provided args to test.",
        )

    def __call__(self, args: Annotated[list, "arguments that go after `cargo test`"]):
        return run_cargo("test", args)


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
        return {
            path: "ok",
            "desc": f"Chunk replaced from `{repr(replace_from[:32])}`... to `{repr(replace_with[:32])}`...",
        }
