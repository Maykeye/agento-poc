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

    # Fold operations - base implementation that throws exceptions or returns dummy results
    def add_fold(self, path: str, position: str, pattern: str, name: str) -> dict | str:
        """Add a fold to hide file content.

        Base implementation raises NotImplementedError.
        Only SuffixHandler supports folding.
        """
        del (path, position, pattern, name)
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
