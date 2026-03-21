from pathlib import Path
import json


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
