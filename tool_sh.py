# File with tools that run executables
from pathlib import Path
from tool import run_executable, Tool
from typing import Annotated
import config
import subprocess
import shlex


def run_git(args: list[str]):
    return run_executable(["git", "-C", str(config.project_directory())] + args)


def run_cargo(cmd: str, args: list[str]):
    all_args = [
        "cargo",
        "--color",
        "never",
        cmd,
        "--manifest-path",
        str(config.project_directory().joinpath("Cargo.toml")),
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


class ToolCargoClippy(Tool):
    def __init__(self):
        super().__init__(
            "cargo_clippy",
            "Execute `cargo clippy` to check if project has code smells",
        )

    def __call__(self, args: Annotated[list, "arguments that go after `cargo clippy`"]):
        return run_cargo("clippy", args)


class ToolCargoTest(Tool):
    def __init__(self):
        super().__init__(
            "cargo_test",
            "Execute `cargo nextest run (args)` with additionals provided args to run tests using `nextest`.",
        )

    def __call__(
        self, args: Annotated[list, "arguments that go after `cargo nextest run`"]
    ):
        return run_cargo("nextest", ["run"] + args)


class ToolPythonUnittest(Tool):
    def __init__(self):
        super().__init__(
            "python_unittest",
            "Execute `python -m unittest (args)` with additionals provided args to run tests using `unittest`.",
        )

    def __call__(
        self, args: Annotated[list, "arguments that go after `python -m unittest`"]
    ):
        all_args = [
            "python",
            "-m",
            "unittest",
        ] + args
        return run_executable(all_args)


class ToolRustApiInfo(Tool):
    def __init__(self):
        super().__init__(
            "brief_rust_api_info",
            "Returns information about API of the project: scans all .rs files and return info of used modules and what functions and structs there are with their docstrings, implementations of functions are omitted. Use this tool to know what struct or function where defined",
        )

    def __call__(self):
        result = run_executable(["rust-api-helper.sh", str(config.project_directory())])
        if result.get("exitcode") != 0 or "stdout" not in result:
            return result
        stdout = result["stdout"].strip()
        return f"=== vvv API vvv ===\n{stdout}\n=== ^^^ API ^^^ ==="


class ToolPupeeter(Tool):
    def __init__(self) -> None:
        return super().__init__(
            "puppeteer",
            "Run pupeeter against html file and returns all JS console output.",
        )

    def __call__(
        self,
        html_path: Annotated[
            str,
            "A project path to .html file to run in puppeteer to get the testing info",
        ],
    ):
        full_path = Path(__file__).parent.resolve()
        puppeteer_path = full_path / "tool_puppeteer_run.js"
        all_args = [str(puppeteer_path), config.real_path(html_path)]
        return run_executable(all_args)


# TODO: promoto to the tool one day
def rustfmt():
    """Run rust fmt"""
    p = shlex.quote(str(config.project_directory()))
    subprocess.run(["bash", "-c", f"cd {p}; rustfmt --edition 2024 **/*.rs"])
