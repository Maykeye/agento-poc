import difflib
import subprocess
import os
from typing import Iterable, Optional
from pathlib import Path

from agento.config import CONFIG
import time

TEMP_DIR = Path(os.getenv("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}") + "/.agento")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

"""Default ininitial external tools"""


def import_tools(path: str):
    globals = {}
    txt = Path(f"{CONFIG.project_directory}/{path.strip()}").read_text()
    exec(txt, globals)
    globals["import_tools"](CONFIG.external_tools)


def delta_time_str(_cache: list = [time.perf_counter()]):
    elapsed = (time.perf_counter() - _cache[0]) / 60.0
    return f"{elapsed:.02f}m"


def expand_file(prompt_file: str, used_files: Optional[set[str]] = None, done=False):
    def cmd(s: str, pfx: str):
        if s.startswith(pfx):
            return s[len(pfx) :].lstrip()
        return None

    used_files = used_files or set()
    if prompt_file in used_files:
        raise ValueError(f"looping {prompt_file}")

    used_files.add(prompt_file)
    prompt = read_text(prompt_file).strip()
    lines = prompt.splitlines()
    new = []

    for line in lines:
        if done:
            new += [line]
            continue
        if cmd(line, "@done") is not None:
            done = True
            continue
        elif cmd(line, "@eof") is not None:
            break
        elif lang := cmd(line, "@lang "):
            lang = lang.strip()
            print(f"Forced langauge: {lang}")
            CONFIG.forced_language = True
            CONFIG.language = lang
            continue
        elif include := cmd(line, "@read "):
            out = expand_file(include, used_files, done)
            new += [out]
            continue
        elif project := cmd(line, "@project_dir "):
            CONFIG.project_directory = Path(project).resolve()
            print(f"Project directory: {CONFIG.project_directory}")
            continue
        elif tool := cmd(line, "@import_tools "):
            import_tools(tool)
            continue
        elif line.startswith("@#"):  # commentary
            continue
        elif line.startswith("@"):
            raise ValueError(f"Unknown command {line}")
        else:
            new += [line]
    return "\n".join(new)


def read_text(path, default: Optional[str] = None):
    if default is not None and not Path(path).exists():
        print(f"{path} doesn't exist, default is used")
        return default

    return Path(path).read_text()


def commit_files(desc: str, files: dict[str, str]):
    """Commit files to the git repo"""
    for path, text in files.items():
        Path(path).write_text(text)
        subprocess.run(["git", "add", path])
    return subprocess.run(["git", "commit", "-m", desc])


def data_tag(tag: str, value: str):
    return f"<{tag}>\n{value}\n</{tag}>"


def diff_gen(old_file: str, new_file: str, path: str) -> Iterable[str]:
    return difflib.unified_diff(
        old_file.splitlines(),
        new_file.splitlines(),
        f"a/{path}",
        f"b/{path}",
        lineterm="",
    )


def extract_tag(text: str, tag: str, strip=True) -> str:
    """Helper to safely extract content between XML tags.
    Matching <tag></tag>, then  <tag>.*$, then just text
    """

    if (idx := text.rfind(f"<{tag}>\n")) >= 0:
        text = text[idx + len(f"<{tag}>\n") :]
    if (idx := text.rfind(f"</{tag}>\n")) >= 0:
        text = text[:idx]
    return text.strip() if strip else text


def debug_print(*args, **kwargs):
    """Debug print function. Passes all args to print.
    Can be easily disabled later by replacing with a no-op function."""
    print(*args, **kwargs)


def error(*args, **kwargs):
    BR_RED = "\x1b[91m"
    RESET = "\x1b[0m"
    print(BR_RED, end="")
    print(*args, **kwargs, end=f"{RESET}\n")


def format_duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"
