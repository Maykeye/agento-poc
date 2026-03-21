# File with tools that run executables
from tool import run_executable, Tool
from typing import Annotated
from config import PROJECT_DIRECTORY
import subprocess
import shlex


def run_git(args: list[str]):
    return run_executable(["git", "-C", str(PROJECT_DIRECTORY)] + args)


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


class ToolGitAdd(Tool):
    def __init__(self):
        super().__init__(
            "git_add",
            "Execute `git add` with additional provided arguments",
        )

    def __call__(self, args: Annotated[list, "arguments tht go after `git diff`"]):
        return run_git(["add"] + args)


class ToolGitStatus(Tool):
    def __init__(self):
        super().__init__("git_status", "Execute `git status` with no arguments")

    def __call__(self):
        return run_git(["status"])


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


# TODO: promoto to the tool one day
def rustfmt():
    """Run rust fmt"""
    p = shlex.quote(str(PROJECT_DIRECTORY))
    subprocess.run(["bash", "-c", f"cd {p}; rustfmt **/*.rs"])
