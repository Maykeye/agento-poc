from pathlib import Path

PROJECT_DIRECTORY_ = Path("<THE PROJECT DIRECTORY IS NOT SET>")
""" Home project directory, a "root" of the project """


def set_project_directory(project: str | Path, silent=False):
    """Setup project directory (will be resolved to absolute, expanduser() yaself)"""
    global PROJECT_DIRECTORY_
    PROJECT_DIRECTORY_ = Path(project).absolute()
    if not silent:
        print(f"New project dir: {PROJECT_DIRECTORY_}")


def project_directory():
    return PROJECT_DIRECTORY_


def guess_project_language():
    files = list(project_directory().glob("*"))
    if any(file.name == "Cargo.toml" for file in files):
        return "rust"
    if any(file.name.endswith(".py") for file in files):
        return "py"
    if any(file.name.endswith(".js") for file in files):
        return "js"
    if project_directory() / "static" / "index.html":
        return "js"
    print(f"WARNING! NO LANGUAGE DETECTED IN {project_directory()}")
    return "nul"


READ_ONLY_FILES = []
READ_ONLY_ERROR = """FATAL ERRROR. 
You are NOT allowed to edit this file.
>>> ABORT EVERYTHING WHAT YOU ARE DOING AT ONCE! 
>>> INSTEAD EXPLAIN WHAT PART OF REQUEST MADE YOU TO TRY TO EDIT IT
"""


def make_file_readonly(path: str):
    READ_ONLY_FILES.append(real_path(path))


def reset_readonly_files():
    READ_ONLY_FILES.clear()


def real_path(in_project_path: str | Path):
    path = project_directory().joinpath(in_project_path).resolve()
    if path.is_relative_to(project_directory()):
        return path
    else:
        raise ValueError(
            f"{in_project_path} is out of bounds of project directory {project_directory()}"
        )
