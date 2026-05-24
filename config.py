from pathlib import Path
from dataclasses import dataclass


@dataclass
class Config:
    language: str
    forced_language: bool
    project_directory: Path
    logging_sqlite_path: Path
    read_only_files: list

    def guess_project_language(self) -> str:
        files = list(self.project_directory.glob("*"))
        if any(file.name == "Cargo.toml" for file in files):
            return "rust"
        if any(file.name.endswith(".py") for file in files):
            return "py"
        if any(file.name.endswith(".js") for file in files):
            return "js"
        if CONFIG.project_directory / "static" / "index.html":
            return "js"
        print(f"WARNING! NO LANGUAGE DETECTED IN {CONFIG.project_directory}")
        return "none"

    def make_file_readonly(self, path: str):
        self.read_only_files.append(real_path(path))

    def reset_readonly_files(self):
        self.read_only_files.clear()


CONFIG = Config(
    language="none",
    project_directory=Path("<THE PROJECT_DIRECTORY IS NOT SET>"),
    logging_sqlite_path=Path("~/.local/state/agento/agento.log").expanduser(),
    read_only_files=[],
    forced_language=False,
)


READ_ONLY_ERROR = """FATAL ERRROR. 
You are NOT allowed to edit this file.
>>> ABORT EVERYTHING WHAT YOU ARE DOING AT ONCE! 
>>> INSTEAD EXPLAIN WHAT PART OF REQUEST MADE YOU TO TRY TO EDIT IT
"""


def real_path(in_project_path: str | Path):
    path = CONFIG.project_directory.joinpath(in_project_path).resolve()
    s = str(in_project_path)
    if path.is_relative_to(CONFIG.project_directory) or s.startswith("@") or "/@" in s:
        return path
    else:
        raise ValueError(
            f"{in_project_path} is out of bounds of project directory {CONFIG.project_directory}"
        )
