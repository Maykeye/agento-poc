from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional, Protocol, ClassVar

from context.fold import Fold


@dataclass
class ContextEntry:
    path: str
    text: str
    id: str
    operation: str

    last_id: ClassVar[int] = 0


class ContextMode(StrEnum):
    RAW = "raw"
    PREFIX = "prefix"
    SUFFIX = "suffix"


class LlmProto(Protocol):
    def msg_user(self, txt: str) -> dict[str, str]: ...
    def messages(self) -> list[dict]: ...


class ContextHandler(ABC):
    @abstractmethod
    def update(
        self,
        path: str,
        text: str,
        oper: str,
        edit_chunk: Optional[tuple[str, str]] = None,
    ) -> str | dict: ...

    def prepare_current_llm(self, llm: LlmProto):
        del llm

    @abstractmethod
    def mode(self) -> ContextMode: ...

    def update_fold_line_numbers(
        self, path: str, old_line_count: int, new_line_count: int
    ) -> None:
        del (path, old_line_count, new_line_count)

    def validate_edit_in_visible_content(
        self, path: str, replace_from: str
    ) -> tuple[bool, str]:
        del (path, replace_from)
        return (True, "")

    # Fold operations - base implementation that throws exceptions or returns dummy results
    def add_fold(
        self,
        path: str,
        fold_from_line_num: int,
        fold_from_line: str,
        fold_to_line_num: int,
        fold_to_line: str,
        name: str,
    ) -> dict | str:
        """Add a fold to hide file content.

        Base implementation raises NotImplementedError.
        Only SuffixHandler supports folding.
        """
        del (path, fold_from_line_num, fold_from_line)
        del (fold_to_line_num, fold_to_line, name)
        raise NotImplementedError(
            f"Fold operations not supported in {self.mode().value} context mode. "
            "Use SUFFIX context mode for folding support."
        )

    def unfold(self, path: str, name: str) -> dict | str:
        """Remove a fold by name.

        Base implementation raises NotImplementedError.
        Only SuffixHandler supports folding.
        """
        del (path, name)
        return (
            f"Fold operations not supported in {self.mode().value} context mode. "
            "Use SUFFIX context mode for folding support."
        )

    def unfold_all(self, path: str) -> dict | str:
        """Remove all folds from a file.

        Base implementation raises NotImplementedError.
        Only SuffixHandler supports folding.
        """
        del path
        return (
            f"Fold operations not supported in {self.mode().value} context mode. "
            "Use SUFFIX context mode for folding support."
        )

    def get_folds(self, path: str) -> list[Fold]:
        """Get all folds for a file.

        Base implementation returns empty list (no folds).
        """
        del path
        return []

    def has_folds(self, path: str) -> bool:
        """Check if a file has any folds.

        Base implementation returns False (no folds).
        """
        del path
        return False

    def format_folded_content(self, path: str, text: str) -> str:
        """Format file content with folds applied.

        Base implementation returns original content (no folds to apply).
        """
        del path
        return text.rstrip()

    def visible_to_actual(self, path: str, visible_line: int, text: str) -> int:
        """Convert visible line number to actual file line number.

        Base implementation returns visible line number unchanged (no folds).
        Override in SuffixHandler to handle fold markers.

        Args:
            path: File path
            visible_line: Line number in visible (folded) content (1-indexed)
            text: Full original file content

        Returns:
            Actual line number in the original file (1-indexed)
        """
        del (path, text)
        return visible_line

    def rename_file(
        self, path_src: str, path_dst: str, llm: Optional[LlmProto] = None
    ) -> str | dict:
        """Handle file rename in context.

        Base implementation raises NotImplementedError.
        Override in each context handler to implement rename logic.

        Args:
            path_src: Source file path
            path_dst: Destination file path
            llm: Optional LLM instance for updating messages (suffix mode)

        Returns:
            Success message or error dict
        """
        del (path_src, path_dst, llm)
        raise NotImplementedError(
            f"Rename operations not supported in {self.mode().value} context mode"
        )
