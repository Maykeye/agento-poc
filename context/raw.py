from context.context_handler import ContextHandler
from typing import override
from pathlib import Path

import config
from context.context_handler import ContextMode


class RawHandler(ContextHandler):
    @override
    def update(self, path: str, text: str, oper: str, edit_chunk=None):
        if oper == "read_file":
            return f">>> OK: {oper} {path}\n>>> === CONTENT START ===\n{text}\n>>> === CONTENT END ==="
        elif oper == "write_file":
            sz = Path(config.real_path(path)).stat().st_size
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
        elif oper == "edit_diff_patch":
            sz = Path(config.real_path(path)).stat().st_size
            lines = text.splitlines()
            return f">>> OK: {oper} {path} ({sz} bytes, {len(lines)} lines)\n>>> PATCH APPLIED"

        assert False, "not yet implemented"

    @override
    def mode(self):
        return ContextMode.RAW
