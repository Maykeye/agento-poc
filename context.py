from dataclasses import dataclass
from typing import ClassVar

_CONTEXT_MODE: bool = False


def set_context_mode(value: bool):
    global _CONTEXT_MODE
    _CONTEXT_MODE = value


def context_mode():
    return _CONTEXT_MODE


@dataclass
class ContextEntry:
    path: str
    text: str
    id: str
    operation: str

    last_id: ClassVar[int] = 0


CONTEXTS = {}


def update_context(path: str, text: str, operation: str):
    ContextEntry.last_id += 1
    id = f"CTX({ContextEntry.last_id})"

    if path in CONTEXTS:
        del CONTEXTS[path]
    CONTEXTS[path] = ContextEntry(path, text, id, operation)
    return {"context-change": {"path": path, "operation": operation, "id": id}}


def context_header():
    return "THIS IS AUTOMATICALLY GENERATED MESSAGE RESERVED FOR FILES CONTEXTS."


def format_context():
    res = [
        context_header(),
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
