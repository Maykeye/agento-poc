import subprocess
import os
from typing import Optional
from pathlib import Path

from config import CONFIG

TEMP_DIR = Path(os.getenv("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}") + "/.agento")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


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
        elif cmd(line, "@eof") is not None:
            break
        elif include := cmd(line, "@read "):
            out = expand_file(include, used_files, done)
            new += [out]
        elif project := cmd(line, "@project_dir "):
            CONFIG.project_directory = Path(project).resolve()
            print(f"Project directory: {CONFIG.project_directory}")
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


def extract_tag(text: str, tag: str, strip=True) -> str:
    """Helper to safely extract content between XML tags.
    Matching <tag></tag>, then  <tag>.*$, then just text
    """

    if (idx := text.rfind(f"<{tag}>\n")) >= 0:
        text = text[idx + len(f"<{tag}>\n") :]
    if (idx := text.rfind(f"</{tag}>\n")) >= 0:
        text = text[:idx]
    return text.strip() if strip else text


def name_tag(id, _cache={}):
    if id not in _cache:
        _cache[id] = f"q{len(_cache)+1:x}"
    return _cache[id]


def debug_print(*args, **kwargs):
    """Debug print function. Passes all args to print.
    Can be easily disabled later by replacing with a no-op function."""
    print(*args, **kwargs)


def format_duration(seconds: float) -> str:
    minutes, seconds = divmod(int(seconds), 60)
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"
