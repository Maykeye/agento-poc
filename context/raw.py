from context.context_handler import ContextHandler
from typing import override, Optional
from pathlib import Path

import config
from context.context_handler import ContextMode, LlmProto


class RawHandler(ContextHandler):
    @override
    def update(self, path: str, text: str, oper: str, edit_chunk=None):
        if oper == "read_file":
            return f">>> OK: {oper} {path}\n>>> === CONTENT START ===\n{text}\n>>> === CONTENT END ==="
        elif oper == "write_file":
            sz = Path(config.real_path(path)).stat().st_size
            lines = text.splitlines()
            first_line = f"\n>>> FIRST WRITTEN LINE: {lines[0]}" if lines else ""
            return f">>> OK: {oper} {path} ({sz} bytes, {len(lines)} lines){first_line}"
        elif oper == "delete_file":
            return {path: "ok", "desc": f"File deleted"}
        elif oper == "edit_file":
            assert edit_chunk
            replace_from, replace_with = edit_chunk
            return {
                path: "ok",
                "desc": f"Chunk replaced from `{repr(replace_from[:32])}`... to `{repr(replace_with[:32])}`...",
            }
        elif oper == "patch_suffix":
            return {path: "ok", "desc": f"Suffix patch applied"}

        assert False, "not yet implemented"

    @override
    def mode(self):
        return ContextMode.RAW

    @override
    def rename_file(
        self, path_src: str, path_dst: str, llm: Optional[LlmProto] = None
    ) -> str | dict:
        """Handle file rename in raw context mode.

        Raw mode doesn't track context, so rename returns a simple success message.

        Args:
            path_src: Source file path
            path_dst: Destination file path
            llm: Optional LLM instance (not used in raw mode)

        Returns:
            Success message
        """
        del llm  # Not used in raw mode
        return f">>> OK: rename_file from {path_src} to {path_dst}"

    @override
    def close_file(
        self, path: str, reason: str, llm: Optional[LlmProto] = None
    ) -> str | dict:
        """Handle file close in raw context mode.

        Raw mode doesn't track context, so close_file is a NOP.

        Args:
            path: File path to close
            reason: Reason for closing the file
            llm: Optional LLM instance (not used in raw mode)

        Returns:
            Success message
        """
        del (reason, llm)  # Not used in raw mode
        return f">>> OK: close_file {path}"
