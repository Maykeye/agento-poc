from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import Optional, Protocol, ClassVar


@dataclass
class ContextEntry:
    path: str
    text: str
    id: str
    operation: str

    last_id: ClassVar[int] = 0


class ContextMode(StrEnum):
    RAW = "raw"


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
