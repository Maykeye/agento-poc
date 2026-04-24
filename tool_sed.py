# File with sed tool that runs sed on a file
from typing import Annotated
import difflib
from tool import Tool, run_executable
from config import real_path

# Stub: this may become configurable in the future
accept = True


class ToolSed(Tool):
    def __init__(self):
        super().__init__(
            name="sed",
            description="Run sed on a file with given script. Captures diff -u between original and sed-processed output. If accept is True, writes result to the file.",
        )

    def __call__(
        self,
        path: Annotated[str, "Project path to the file to process"],
        script: Annotated[str, "sed script/expression to apply"],
    ):
        p = real_path(path)

        if not p.exists():
            return {path: "error", "error": f"File {path} doesn't exist"}
        if not p.is_file():
            return {path: "error", "error": f"Path {path} is not a file"}

        original_text = p.read_text()

        # Run sed with the file content as stdin
        result = run_executable(["sed", script], stdin_text=original_text)

        if result.get("exitcode", -1) != 0:
            return {
                path: "error",
                "error": f"sed failed with exitcode {result.get('exitcode')}",
                "stderr": result.get("stderr", ""),
            }

        sed_output = result.get("stdout", "")

        # Generate diff -u between original and sed output
        original_lines = original_text.splitlines(keepends=True)
        sed_lines = sed_output.splitlines(keepends=True)

        diff = list(
            difflib.unified_diff(
                original_lines,
                sed_lines,
                fromfile=f"a/{path}",
                tofile=f"b/{path}",
            )
        )
        diff_text = "".join(diff) if diff else ""

        if accept:
            p.write_text(sed_output)
            return {
                path: "accepted",
                "info": f"File written after sed",
                "diff": diff_text if diff_text else "(no changes)",
            }
        else:
            return {
                path: "rejected",
                "info": f"accept=False, file not written",
                "diff": diff_text if diff_text else "(no changes)",
            }
