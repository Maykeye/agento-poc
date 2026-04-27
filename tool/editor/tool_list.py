from typing import Callable
from tool import Tool

EDITOR_TOOLS: list[Callable[[], Tool]] = []
""" List of editor-oriented tool constructor. 
Example: `EDITOR_TOOLS.append(ToolGoto)` """
