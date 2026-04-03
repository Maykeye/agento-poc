from dataclasses import dataclass


@dataclass
class Fold:
    """Represents a fold in a file."""

    name: str
    position: str  # "head" or "tail"
    pattern: str
    start_line: int  # 1-indexed
    end_line: int  # 1-indexed
