# Tools that run executables
from pathlib import Path
from typing import Annotated
import subprocess
import shlex

from agento import config
from agento.config import CONFIG
from agento.tool import run_executable, Tool


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


class ToolBash(Tool):
    def __init__(self):
        super().__init__(
            "bash",
            "Run `bash -c` command. Use this to execute shell commands. "
            "For large output, pipe through `tail` to limit lines (e.g., `abc | tail -n20` "
            "to show only last 20 lines). Accepts a single string command.",
        )

    def __call__(
        self,
        command: Annotated[str, "bash command as a single string"],
    ):
        return run_executable(["bash", "-c", command])


# TODO: promoto to the tool one day
def rustfmt():
    """Run rust fmt"""
    p = shlex.quote(str(CONFIG.project_directory))
    subprocess.run(["bash", "-c", f"cd {p}; rustfmt --edition 2024 **/*.rs"])
