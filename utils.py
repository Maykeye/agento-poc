from pathlib import Path
import subprocess
from typing import Optional

import config


def expand_file(prompt_file: str, used_files: Optional[set[str]] = None):
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
        if include := cmd(line, "@read "):
            out = expand_file(include, used_files)
            new += [out]
        elif project := cmd(line, "@project_dir "):
            config.set_project_directory(Path(project).resolve())
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


def log_prompt(project, prompt):
    """Log used prompt for the project"""
    import sqlite3
    import datetime

    p = Path("~/.local/state/agento.log").expanduser()
    now = datetime.datetime.now(datetime.UTC).isoformat()
    with sqlite3.connect(p) as sql:
        sql.execute(
            "CREATE TABLE IF NOT EXISTS log(project TEXT NOT NULL, log TEXT NOT NULL, created_at TEXT NOT NULL)"
        )
        sql.execute("CREATE INDEX IF NOT EXISTS log_proj_idx ON log(project)")
        sql.execute(
            "INSERT INTO log(project, log, created_at) VALUES(?,?,?)",
            (project, prompt, now),
        )


def name_tag(id, _cache={}):
    if id not in _cache:
        _cache[id] = f"q{len(_cache)+1:x}"
    return _cache[id]
