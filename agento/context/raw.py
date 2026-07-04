from agento.context.context_handler import ContextHandler, ContextMode, LlmProto
from typing import override


class RawHandler(ContextHandler):
    """Raw handler is as raw as it can be"""

    @override
    def update(self, path: str, text: str, oper: str, edit_chunk=None):
        text = f">>> OK: {oper}: {path}\n>>> === CONTENT START ===\n{text}\n>>> === CONTENT END ==="
        if edit_chunk:
            text += "\n>>> === DATA CHUNK ===\n"
            text += str(edit_chunk)
        return text

    @override
    def mode(self):
        return ContextMode.RAW
