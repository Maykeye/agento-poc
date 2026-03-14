from pathlib import Path
import subprocess


def read_text(path):
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
