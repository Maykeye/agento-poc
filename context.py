from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional, override
from enum import StrEnum
from typing import Protocol
import re


class LlmProto(Protocol):
    def msg_user(self, txt: str) -> dict[str, str]: ...
    def messages(self) -> list[dict]: ...


class ContextMode(StrEnum):
    RAW = "raw"
    PREFIX = "prefix"
    SUFFIX = "suffix"


def set_context_mode(value: ContextMode, reset_ctx_id=False):
    global _CONTEXT_HANDLER
    if reset_ctx_id:
        ContextEntry.last_id = -1
    match value:
        case ContextMode.RAW:
            _CONTEXT_HANDLER = RawHandler()
        case ContextMode.PREFIX:
            _CONTEXT_HANDLER = PrefixHandler()
        case ContextMode.SUFFIX:
            _CONTEXT_HANDLER = SuffixHandler()


@dataclass
class ContextEntry:
    path: str
    text: str
    id: str
    operation: str

    last_id: ClassVar[int] = 0


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


class RawHandler(ContextHandler):
    @override
    def update(self, path: str, text: str, oper: str, edit_chunk=None):
        if oper == "read_file":
            return f">>> OK: {oper} {path}\n>>> === CONTENT START ===\n{text}\n>>> === CONTENT END ==="
        elif oper == "write_file":
            sz = Path(path).stat().st_size
            lines = text.splitlines()
            return f">>> OK: {oper} {path} ({sz} bytes, {len(lines)} lines)\n>>> FIRST WRITTEN LINE: {lines[0]}"
        elif oper == "delete_file":
            return {path: "ok", "desc": f"File deleted"}
        elif oper == "edit_file":
            assert edit_chunk
            replace_from, replace_with = edit_chunk
            return {
                path: "ok",
                "desc": f"Chunk replaced from `{repr(replace_from[:32])}`... to `{repr(replace_with[:32])}`...",
            }

        assert False, "not yet implemented"

    @override
    def mode(self):
        return ContextMode.RAW


_CONTEXT_HANDLER: ContextHandler = RawHandler()
SUFFIX_CONTEXTS = {}


def context_handler():
    return _CONTEXT_HANDLER


class PrefixHandler(ContextHandler):
    @override
    def mode(self):
        return ContextMode.PREFIX

    @override
    def update(self, path: str, text: str, oper: str, edit_chunk=None):
        del edit_chunk
        ContextEntry.last_id += 1
        id = f"CTX({ContextEntry.last_id})"

        if path in CONTEXTS:
            del CONTEXTS[path]
        CONTEXTS[path] = ContextEntry(path, text, id, oper)
        return {"context-change": {"path": path, "operation": oper, "id": id}}

    def prepare_current_llm(self, llm: LlmProto):
        messages = llm.messages()

        # allow 1st message to be the system
        idx = -1
        if messages[0]["role"] == "user":
            idx = 0
        elif len(messages) > 1 and messages[1]["role"] == "user":
            idx = 1

        if idx >= 0:
            if not messages[idx]["content"].startswith(self.context_header()):
                messages.insert(idx, {})
        else:
            raise ValueError(
                f"Can't find index for content message#{idx} is: {messages[idx:idx+1]}"
            )
        messages[idx] = llm.msg_user(self.format_context())

    def context_header(self):
        return "THIS IS AUTOMATICALLY GENERATED MESSAGE RESERVED FOR FILES CONTEXTS."

    def format_context(self):
        res = [
            self.context_header(),
            "Result of file context operations will be logged here. Eg if you read/edit/write file, it will be placed in this message.",
            "This message will be edited automatically, so even if you reply goes after it, the message content is up to date.",
            "Each file operation will be identified by `id`entifier.",
            "When you perform operation with any file, tool will give you the identifier and the identifier will be in the context together with path",
            "so you can focus on the fresh full file content",
            "",
        ]

        for ctx in CONTEXTS.values():
            res += [
                f"=== Path: {ctx.path} (last id: {ctx.id}; last operation: {ctx.operation}) CONTEXT START ==="
            ]
            res += ctx.text.splitlines()
            res += [
                f"=== Path: {ctx.path} (last id: {ctx.id}; last operation: {ctx.operation}) CONTEXT END ===\n"
            ]
        return "\n".join(res).rstrip()


CONTEXTS = {}


class SuffixHandler(ContextHandler):
    """Suffix handler that replaces old file content in messages with references to new content."""

    def __init__(self):
        self._prefix: str = ">>>"

    @property
    def prefix(self) -> str:
        """Get the prefix used for context markers."""
        return self._prefix

    @prefix.setter
    def prefix(self, value: str) -> None:
        """Set the prefix used for context markers."""
        self._prefix = value

    @override
    def mode(self):
        return ContextMode.SUFFIX

    @override
    def update(self, path: str, text: str, oper: str, edit_chunk=None):  # type: ignore
        # Increment ID and create new context entry
        ContextEntry.last_id += 1
        new_id = f"CTX({ContextEntry.last_id})"

        # Store the new context entry
        SUFFIX_CONTEXTS[path] = ContextEntry(path, text, new_id, oper)

        # Return the new content with ID
        pfx = self.prefix
        if oper == "read_file":
            return f"{pfx} ID: {new_id} OPERATION: {oper} CTX-IO-FILE:  {path}\n{pfx} === CONTENT START ===\n{text}\n{pfx} === CONTENT END ==="
        elif oper == "write_file":
            sz = Path(path).stat().st_size
            lines = text.splitlines()
            return f"{pfx} ID: {new_id} OPERATION: {oper} CTX-IO-FILE:  {path}\n{pfx} OK: {oper} {path} ({sz} bytes, {len(lines)} lines)\n{pfx} === CONTENT START ===\n{text}\n{pfx} === CONTENT END ==="
        elif oper == "delete_file":
            return f"{pfx} ID: {new_id} OPERATION: {oper} CTX-IO-FILE:  {path}\n{pfx} OK: {oper} {path}\n{pfx} === CONTENT START ===\n{text}\n{pfx} === CONTENT END ==="
        elif oper == "edit_file":
            assert edit_chunk
            replace_from, replace_with = edit_chunk
            return f"{pfx} ID: {new_id} OPERATION: {oper} CTX-IO-FILE:  {path}\n{pfx} OK: edit {path} (replaced `{repr(replace_from)}` with `{repr(replace_with)}`)\n{pfx} === CONTENT START ===\n{text}\n{pfx} === CONTENT END ==="

        raise ValueError("Unknown oper")

    def prepare_current_llm(self, llm: LlmProto):
        """Update all messages that reference old file content to point to new content."""
        messages = llm.messages()

        # For each tracked path, update all messages that have old CTX-IO-FILE references
        for path, ctx in SUFFIX_CONTEXTS.items():
            self._update_messages_for_path(messages, path, ctx)

    def _update_messages_for_path(
        self, messages: list[dict], path: str, ctx: ContextEntry
    ):
        """Update all messages for a given path to reference the current context.

        Only updates tool messages (role="tool"), not assistant messages.
        Assistant messages that contain file content replicas should remain unchanged.
        """

        for msg in messages:
            # Skip non-tool messages (e.g., assistant messages)
            # Assistant messages may contain replicas of tool results, but they
            # should not be modified during epilogue
            if msg.get("role") != "tool":
                continue

            if "content" not in msg:
                continue
            content = msg["content"]
            if not isinstance(content, str):
                continue

            if "CTX-IO-FILE" not in content or path not in content:
                continue

            if not content.startswith(self.prefix):
                continue

            msg["content"] = self._replace_old_content(content, path, ctx.id)

    def _replace_old_content(self, content: str, path: str, current_id: str) -> str:
        """Replace old file content blocks with reference to current content."""
        pfx = re.escape(self.prefix)

        # Pattern 1: Match full content blocks (with CONTENT START/END)
        # Allow optional extra lines (like ">>> OK: write_file ...") between header and content start
        block_pattern = rf"({pfx} ID: CTX\(\d+\) OPERATION: \w+ CTX-IO-FILE:\s*{path})\s*\n(?:.*\s*\n)*?{pfx} === CONTENT START ===\s*\n(.*?)\s*\n{pfx} === CONTENT END ==="

        def replace_block_impl(match, n):
            header = match.group(n)
            id_match = re.search(r"CTX\((\d+)\)", header)
            current_id_match = re.search(r"CTX\((\d+)\)", current_id)
            if id_match and current_id_match:
                old_num = int(id_match.group(1))
                current_num = int(current_id_match.group(1))
                if old_num < current_num:
                    return f"{header}\n{self.prefix} === CURRENT CONTENT IN {current_id} ==="
            return match.group(0)

        new_content = re.sub(
            block_pattern, lambda m: replace_block_impl(m, 1), content, flags=re.DOTALL
        )

        # Pattern 2: Also update existing "CURRENT CONTENT" references to point to new current_id
        # Example: >>> ID: CTX(0) OPERATION: read_file CTX-IO-FILE:  .agento.demo.foo
        #          >>> === CURRENT CONTENT IN CTX(1) ===
        modern_ref_pattern = rf"(({pfx} ID: CTX\(\d+\) OPERATION: \w+ CTX-IO-FILE:\s*{path})\s*\n{pfx} === CURRENT CONTENT IN CTX\(\d+\) ===)"
        new_content = re.sub(
            modern_ref_pattern, lambda m: replace_block_impl(m, 2), new_content
        )

        return new_content.rstrip()
