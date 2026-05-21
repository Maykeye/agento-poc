from context.context_handler import ContextHandler
from typing import override, Optional

import config
from context.context_handler import ContextMode, ContextEntry, LlmProto
from context.fold import Fold
from pathlib import Path
import json
import copy


class SuffixHandler(ContextHandler):
    """Suffix handler that replaces old file content in messages with references to new content."""

    keep_old_edits = 5

    def __init__(self):
        self._prefix: str = ">>>"
        # Track folds per file: {path: [Fold, ...]}
        self._folds: dict[str, list[Fold]] = {}

        self.llm_to_file_entries: dict[int, dict[str, ContextEntry]] = {}

    @property
    def prefix(self) -> str:
        """Get the prefix used for context markers."""
        return self._prefix

    @prefix.setter
    def prefix(self, value: str) -> None:
        """Set the prefix used for context markers."""
        self._prefix = value or ">>>"

    @override
    def mode(self):
        return ContextMode.SUFFIX

    def file_entries(self):
        from llm import LLM

        key = id(LLM.INSTANCES[-1].llm) if LLM.INSTANCES else 0
        if key not in self.llm_to_file_entries:
            self.llm_to_file_entries[key] = {}
            # Copy entries from previous LLM
            if len(LLM.INSTANCES) > 1:
                penultimate_key = id(LLM.INSTANCES[-2].llm)
                if penultimate_key in self.llm_to_file_entries:
                    self.llm_to_file_entries[key] = copy.deepcopy(
                        self.llm_to_file_entries[penultimate_key]
                    )

        # TODO: cleanup
        return self.llm_to_file_entries[key]

    @override
    def update(self, path: str, text: str, oper: str, edit_chunk=None):  # type: ignore
        # Increment ID and create new context entry
        ContextEntry.last_id += 1
        new_id = f"CTX({ContextEntry.last_id})"

        # Store the new context entry
        self.file_entries()[path] = ContextEntry(path, text, new_id, oper)

        # Return the new content with ID
        pfx = self.prefix
        if oper == "read_file":
            # Apply existing folds when reading a file
            display_text = (
                self.format_folded_content(path, text) if self.has_folds(path) else text
            )
            return f"{pfx} ID: {new_id} OPERATION: {oper} CTX-IO-FILE: {path}\n{pfx} === CONTENT START ===\n{display_text}\n{pfx} === CONTENT END ==="
        elif oper == "write_file":
            sz = Path(config.real_path(path)).stat().st_size
            lines = text.splitlines()
            return f"{pfx} ID: {new_id} OPERATION: {oper} CTX-IO-FILE: {path}\n{pfx} OK: {oper} {path} ({sz} bytes, {len(lines)} lines)\n{pfx} === CONTENT START ===\n{text}\n{pfx} === CONTENT END ==="
        elif oper == "delete_file":
            return f"{pfx} ID: {new_id} OPERATION: {oper} CTX-IO-FILE: {path}\n{pfx} OK: {oper} {path}\n{pfx} === CONTENT START ===\n{text}\n{pfx} === CONTENT END ==="
        elif oper == "edit_file":
            assert edit_chunk
            replace_from, replace_with = edit_chunk
            return f"{pfx} ID: {new_id} OPERATION: {oper} CTX-IO-FILE: {path}\n{pfx} OK: edit {path} (replaced `{repr(replace_from)}` with `{repr(replace_with)}`)\n{pfx} === CONTENT START ===\n{text}\n{pfx} === CONTENT END ==="
        elif oper == "file_add_fold":
            assert edit_chunk
            fold_name, _ = edit_chunk
            return f"{pfx} ID: {new_id} OPERATION: {oper} CTX-IO-FILE: {path}\n{pfx} OK: fold added '{fold_name}'\n{pfx} === CONTENT START ===\n{text}\n{pfx} === CONTENT END ==="
        elif oper == "file_unfold":
            assert edit_chunk
            fold_name, _ = edit_chunk
            return f"{pfx} ID: {new_id} OPERATION: {oper} {path}\n{pfx} OK: fold removed '{fold_name}'\n{pfx} === CONTENT START ===\n{text}\n{pfx} === CONTENT END ==="
        elif oper == "file_unfold_all":
            assert edit_chunk
            num_folds, _ = edit_chunk
            return f"{pfx} ID: {new_id} OPERATION: {oper} {path}\n{pfx} OK: {num_folds} removed\n{pfx} === CONTENT START ===\n{text}\n{pfx} === CONTENT END ==="

        raise ValueError(f"Unknown oper {oper}")

    def prepare_current_llm(self, llm: LlmProto):
        """Update all messages that reference old file content to point to new content.

        For each file, find the latest context ID in SUFFIX_CONTEXTS and update
        all earlier tool messages to reference the latest content instead of showing
        outdated full content.
        """
        import re

        messages = llm.messages()

        # Build a map of file path -> latest context ID from SUFFIX_CONTEXTS
        latest_contexts: dict[str, str] = {}
        for path, entry in self.file_entries().items():
            latest_contexts[path] = entry.id

        # Process each message to prune old messages
        for msg in messages:
            if msg.get("role") != "tool":
                continue

            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            # Find all context entries in this message
            # Pattern: {prefix} ID: CTX(N) OPERATION: XXX path
            pfx = re.escape(self.prefix)
            pattern = (
                rf"({pfx}\s*ID: CTX\((\d+)\) OPERATION: (\w+) CTX-IO-FILE:\s+(\S+))"
            )

            if not (matches := list(re.finditer(pattern, content))):
                continue

            # Process each context entry found in this message to mark outdated one
            for match in reversed(matches):
                ctx_id_num = int(match.group(2))
                path = match.group(4)

                ctx_id = f"CTX({ctx_id_num})"

                if path not in latest_contexts:
                    continue
                latest_id = latest_contexts[path]
                if ctx_id != latest_id:
                    new_content = self._replace_outdated_context(content, ctx_id, path)
                    msg["content"] = new_content

        # Aggressive pruning of tool calls for editing operations
        self._prune_old_tool_calls(messages, list(latest_contexts.keys()))

    def _replace_outdated_context(
        self, content: str, old_ctx_id: str, path: str
    ) -> str:
        """Replace an outdated context block with a reference to the latest context.

        Args:
            content: Original message content
            old_ctx_id: The outdated context ID (e.g., "CTX(0)")
            path: File path

        Returns:
            Updated content with outdated context replaced by reference
        """
        import re

        pfx = self.prefix

        # Find the full context block for this context ID
        # Pattern: from ">>> ID: CTX(N)..." to ">>> === CONTENT END ==="

        # Match the header line - need to handle both regular and "outdated" operations
        header_pattern = rf"{re.escape(pfx)}\s*ID: {re.escape(old_ctx_id)}\s+OPERATION:\s+\S+\s+CTX-IO-FILE:\s+\S+"
        header_match = re.search(header_pattern, content)

        if not header_match:
            return content

        header_start = header_match.start()
        header_end = header_match.end()

        # Find the CONTENT START marker after the header
        content_start_marker = f"{pfx} === CONTENT START ==="
        content_start_idx = content.find(content_start_marker, header_end)

        if content_start_idx == -1:
            return content

        # Find the CONTENT END marker
        content_end_marker = f"{pfx} === CONTENT END ==="
        content_end_idx = content.find(content_end_marker, content_start_idx)

        if content_end_idx == -1:
            return content

        # Find the end of the CONTENT END line
        content_end_line_end = content.find("\n", content_end_idx)
        if content_end_line_end == -1:
            content_end_line_end = len(content)

        # Replace the entire block with a reference
        old_block = content[header_start:content_end_line_end]
        new_block = (
            f"{pfx} ID: {old_ctx_id} OPERATION: outdated {path}\n"
            f"{pfx} === Content is out of date (updated version below) ==\n"
        )

        return content.replace(old_block, new_block, 1)

    def _prune_old_tool_calls(
        self,
        messages: list[dict],
        our_paths: list[str],
        keep_old_edits: Optional[int] = None,
        prune_all_for_paths: Optional[list[str]] = None,
        include_read_file: bool = False,
    ):
        """Aggressively prune old tool calls that edit files."""
        if keep_old_edits is None:
            keep_old_edits = self.keep_old_edits
        if prune_all_for_paths is None:
            prune_all_for_paths = []

        editing_tools = {
            "search_replace_once",
            "edit_file",
            "write_file",
            "delete_file",
            "file_add_fold",
            "file_unfold",
        }
        # Include read_file if requested (for close_file operation)
        if include_read_file:
            editing_tools.add("read_file")

        # Track edits per path
        edit_counts: dict[str, int] = {}

        # Iterate messages backwards
        for msg_idx in range(len(messages) - 1, -1, -1):
            msg = messages[msg_idx]

            # Skip non-assistant messages
            if msg.get("role") != "assistant":
                continue

            # Skip messages without tool calls
            tool_calls = msg.get("tool_calls", [])
            if not tool_calls:
                continue

            # Iterate tool calls in reverse order
            for tc_idx in range(len(tool_calls) - 1, -1, -1):
                tool_call = tool_calls[tc_idx]

                # Get function info
                func_info = tool_call.get("function", {})
                func_name = func_info.get("name", "")

                # Skip non-editing tools
                if func_name not in editing_tools:
                    continue

                # Decode arguments
                try:
                    args = json.loads(func_info.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    continue

                # Skip if not our path
                if (call_path := args.get("path", "")) not in our_paths:
                    continue

                # Track this edit
                if call_path not in edit_counts:
                    edit_counts[call_path] = 0
                edit_counts[call_path] += 1

                # Check if we should prune this edit
                # For paths in prune_all_for_paths, prune ALL occurrences (not just exceeding keep_old_edits)
                should_prune = (
                    call_path in prune_all_for_paths
                    or edit_counts[call_path] > keep_old_edits
                )

                if should_prune:
                    # Prune this tool call by replacing arguments
                    new_args = {"path": call_path, "cleanup": "the call is removed"}
                    func_info["arguments"] = json.dumps(new_args)

    def _format_fold_result(
        self, path: str, folded_text: str, oper: str, fold_info: str
    ) -> str:
        """Format fold result message, updating SUFFIX_CONTEXTS to track latest context.

        This method updates the context entry's ID to mark this as the current state
        for outdated detection, while preserving the original file content for validation.

        Args:
            path: File path
            folded_text: Formatted content with folds applied (for display only)
            oper: Operation type
            fold_info: Name of fold (or number of folds for unfold_all)

        Returns:
            Formatted result message
        """
        ContextEntry.last_id += 1
        new_id = f"CTX({ContextEntry.last_id})"
        pfx = self.prefix

        # Update the context entry's ID but keep original text for validation
        # This allows prepare_current_llm to mark previous reads as outdated
        # while still preserving original content for fold validation
        if path in self.file_entries():
            old_entry = self.file_entries()[path]
            # Keep the original text, update ID and operation
            self.file_entries()[path] = ContextEntry(
                old_entry.path, old_entry.text, new_id, oper
            )

        if oper == "file_add_fold":
            return f"{pfx} ID: {new_id} OPERATION: {oper} {path}\n{pfx} OK: fold added '{fold_info}'\n{pfx} === CONTENT START ===\n{folded_text}\n{pfx} === CONTENT END ==="
        elif oper == "file_unfold":
            return f"{pfx} ID: {new_id} OPERATION: {oper} {path}\n{pfx} OK: fold removed '{fold_info}'\n{pfx} === CONTENT START ===\n{folded_text}\n{pfx} === CONTENT END ==="
        elif oper == "file_unfold_all":
            return f"{pfx} ID: {new_id} OPERATION: {oper} {path}\n{pfx} OK: {fold_info} removed\n{pfx} === CONTENT START ===\n{folded_text}\n{pfx} === CONTENT END ==="

        raise ValueError(f"Unknown oper {oper}")

    # Fold operations
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

        Args:
            path: File path
            fold_from_line_num: Line number to start fold from (1-indexed)
            fold_from_line: Textual representation of fold_from_line_num (for validation)
            fold_to_line_num: Line number to end fold at (1-indexed)
            fold_to_line: Textual representation of fold_to_line_num (for validation)
            name: Unique name for this fold

        Returns:
            Success message with fold info or error dict
        """
        # Check if file has been read
        if path not in self.file_entries():
            return {
                path: "error",
                "error": f"File {path} has not been read. Read it first with read_file.",
            }

        # Get file content and lines
        text = self.file_entries()[path].text
        lines = text.splitlines()

        # Validate line numbers
        if fold_from_line_num < 1:
            return {
                path: "error",
                "error": f"fold_from_line_num must be >= 1, got {fold_from_line_num}",
            }
        if fold_to_line_num < 1:
            return {
                path: "error",
                "error": f"fold_to_line_num must be >= 1, got {fold_to_line_num}",
            }
        if fold_from_line_num > fold_to_line_num:
            return {
                path: "error",
                "error": f"fold_from_line_num ({fold_from_line_num}) must be <= fold_to_line_num ({fold_to_line_num})",
            }
        if fold_to_line_num > len(lines):
            return {
                path: "error",
                "error": f"fold_to_line_num ({fold_to_line_num}) exceeds file line count ({len(lines)})",
            }

        # Validate line content matches
        # Convert to 0-indexed
        from_idx = fold_from_line_num - 1
        to_idx = fold_to_line_num - 1

        actual_from_line = lines[from_idx]
        actual_to_line = lines[to_idx]

        # Strip trailing whitespace for comparison
        if actual_from_line.rstrip() != fold_from_line.rstrip():
            return {
                path: "error",
                "error": f"Line {fold_from_line_num} content mismatch. Expected `{repr(fold_from_line)}`, got `{repr(actual_from_line)}`",
            }
        if actual_to_line.rstrip() != fold_to_line.rstrip():
            return {
                path: "error",
                "error": f"Line {fold_to_line_num} content mismatch. Expected `{repr(fold_to_line)}`, got `{repr(actual_to_line)}`",
            }

        # Check if fold with same name already exists
        if path in self._folds:
            for existing_fold in self._folds[path]:
                if existing_fold.name == name:
                    return {
                        path: "error",
                        "error": f"Fold with name '{name}' already exists for {path}",
                    }

        # Check for overlapping folds
        if path in self._folds:
            new_fold = Fold(name, fold_from_line_num, fold_to_line_num)
            for existing_fold in self._folds[path]:
                # Check if ranges overlap
                if not (
                    new_fold.end_line < existing_fold.start_line
                    or new_fold.start_line > existing_fold.end_line
                ):
                    return {
                        path: "error",
                        "error": f"Fold '{name}' overlaps with existing fold '{existing_fold.name}' (lines {existing_fold.start_line}..{existing_fold.end_line})",
                    }

        # Create and store the fold
        fold = Fold(name, fold_from_line_num, fold_to_line_num)
        if path not in self._folds:
            self._folds[path] = []
        self._folds[path].append(fold)

        # Format content with folds
        folded_text = self.format_folded_content(path, text)
        return self._format_fold_result(path, folded_text, "file_add_fold", fold.name)

    def unfold(self, path: str, name: str) -> dict | str:
        """Remove a fold by name.

        Args:
            path: File path
            name: Name of fold to remove

        Returns:
            Success message or error dict
        """
        # Check if file has folds
        if path not in self._folds:
            return {
                path: "error",
                "error": f"No folds exist for file {path}",
            }

        # Find the fold
        fold_to_remove = None
        for fold in self._folds[path]:
            if fold.name == name:
                fold_to_remove = fold
                break

        if fold_to_remove is None:
            return {
                path: "error",
                "error": f"Fold '{name}' not found in {path}",
            }

        # Remove the fold
        self._folds[path].remove(fold_to_remove)

        # If no more folds, clean up
        if not self._folds[path]:
            del self._folds[path]

        # Format with remaining folds (or no folds if this was the last one)
        text = self.file_entries()[path].text
        folded_text = self.format_folded_content(path, text)
        return self._format_fold_result(path, folded_text, "file_unfold", name)

    def unfold_all(self, path: str) -> dict | str:
        """Remove all folds from a file.

        Args:
            path: File path

        Returns:
            Success message or error dict
        """
        # Check if file has folds
        if path not in self._folds:
            return {
                path: "error",
                "error": f"No folds exist for file {path}",
            }

        # Get fold names for reporting
        fold_names = [fold.name for fold in self._folds[path]]
        num_folds = len(fold_names)

        # Remove all folds
        del self._folds[path]

        # Format with no folds (full content)
        text = self.file_entries()[path].text
        folded_text = self.format_folded_content(path, text)
        return self._format_fold_result(
            path, folded_text, "file_unfold_all", f"{num_folds} folds"
        )

    def get_folds(self, path: str) -> list[Fold]:
        """Get all folds for a file."""
        return self._folds.get(path, [])

    def has_folds(self, path: str) -> bool:
        """Check if a file has any folds."""
        return path in self._folds and len(self._folds[path]) > 0

    def format_folded_content(self, path: str, text: str) -> str:
        """Format file content with folds applied.

        Returns visible content with fold markers replacing folded lines.
        """
        if path not in self._folds or not self._folds[path]:
            return text.rstrip()

        lines = text.splitlines()
        result_lines = []

        # Sort folds by start line
        sorted_folds = sorted(self._folds[path], key=lambda f: f.start_line)

        fold_idx = 0
        current_line = 1  # 1-indexed line number

        while current_line <= len(lines):
            # Check if we're at the start of a fold
            if (
                fold_idx < len(sorted_folds)
                and current_line == sorted_folds[fold_idx].start_line
            ):
                fold = sorted_folds[fold_idx]
                # Add fold marker
                pfx = self.prefix
                fold_marker = f"{pfx} >> FOLD: lines {fold.start_line}..{fold.end_line} ({fold.name})"
                result_lines.append(fold_marker)
                # Skip to after the fold
                current_line = fold.end_line + 1
                fold_idx += 1
            else:
                # Add the current line
                result_lines.append(lines[current_line - 1])
                current_line += 1

        return "\n".join(result_lines)

    def update_fold_line_numbers(
        self, path: str, old_line_count: int, new_line_count: int
    ) -> None:
        """Update fold line numbers when file content changes."""
        if path not in self._folds:
            return

        line_delta = new_line_count - old_line_count

        for fold in self._folds[path]:
            fold.start_line += line_delta
            fold.end_line += line_delta

            # Ensure folds stay within bounds
            if fold.start_line < 1:
                fold.start_line = 1
            if fold.end_line > new_line_count:
                fold.end_line = new_line_count

    def validate_edit_in_visible_content(
        self, path: str, replace_from: str
    ) -> tuple[bool, str]:
        """Check if text exists exactly once in visible (non-folded) content.

        Returns:
            Tuple of (is_valid, error_message)
        """
        if path not in self.file_entries():
            return (False, f"File {path} has not been read")

        text = self.file_entries()[path].text
        visible_content = self.format_folded_content(path, text)

        # Count occurrences in visible content
        count = visible_content.count(replace_from)

        if count == 0:
            return (
                False,
                f"`{repr(replace_from)}` not found in visible content (may be in a folded region)",
            )
        if count > 1:
            return (
                False,
                f"`{repr(replace_from)}` appears {count} times in visible content (must be exactly once)",
            )

        return (True, "")

    def visible_to_actual(self, path: str, visible_line: int, text: str) -> int:
        """Convert visible line number to actual file line number.

        This method handles fold markers by jumping over folded regions.
        Uses the current prefix setting to detect fold markers.

        Args:
            path: File path
            visible_line: Line number in visible (folded) content (1-indexed)
            text: Full original file content

        Returns:
            Actual line number in the original file (1-indexed)
        """
        import re

        # Get visible content (with folds applied)
        visible_text = self.format_folded_content(path, text)
        visible_lines = visible_text.splitlines()

        actual_line = 0
        pfx = self.prefix

        for i, vline in enumerate(visible_lines):
            actual_line += 1
            # Check if this is a fold marker (using current prefix)
            fold_marker = f"{pfx} >> FOLD:"
            if vline.startswith(fold_marker):
                if match := re.search(r"FOLD: lines \d+\.\.(\d+)", vline):
                    # This visible line represents a fold, jump to end line
                    actual_line = int(match.group(1))

            if i + 1 == visible_line:
                return actual_line

        return actual_line

    @override
    def rename_file(
        self, path_src: str, path_dst: str, llm: Optional[LlmProto] = None
    ) -> str | dict:
        """Handle file rename in suffix context mode.

        This method:
        1. Moves context entry from path_src to path_dst
        2. Updates all messages referencing path_src to reference path_dst

        Args:
            path_src: Source file path
            path_dst: Destination file path
            llm: Optional LLM instance for updating messages

        Returns:
            Success message
        """
        # Move context entry from path_src to path_dst
        # TODO: do it in every suffx context of every LLM
        if path_src in self.file_entries():
            entry = self.file_entries()[path_src]
            del self.file_entries()[path_src]
            # Create new entry with new path but same content
            ContextEntry.last_id += 1
            new_id = f"CTX({ContextEntry.last_id})"
            self.file_entries()[path_dst] = ContextEntry(
                path_dst, entry.text, new_id, "rename_file"
            )

        # Update LLM messages if llm is provided
        if llm is not None:
            self._update_rename_messages(llm.messages(), path_src, path_dst)

        # Return success message
        return f">>> OK: rename_file from {path_src} to {path_dst}"

    def _update_rename_messages(
        self, messages: list[dict], path_src: str, path_dst: str
    ):
        """Update all messages (inputs and outputs) to reference path_dst instead of path_src after rename.

        Args:
            messages: List of message dictionaries
            path_src: Source file path (old name)
            path_dst: Destination file path (new name)
        """
        import re
        import json

        # Get the new context ID for path_dst (created in rename_file)
        new_ctx_id = None
        if path_dst in self.file_entries():
            new_ctx_id = self.file_entries()[path_dst].id

        # Iterate through all messages
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            # Update tool output messages - replace path references in CTX-IO-FILE
            pfx = re.escape(self.prefix)

            # Pattern to match the full header with path_src
            pattern = rf"({pfx}\s*ID:\s*CTX\((\d+)\)\s+OPERATION:\s+(\w+)\s+)CTX-IO-FILE:\s+{re.escape(path_src)}"

            def replace_with_new_path_and_id(match):
                """Replace path_src with path_dst and update context ID if needed."""
                _ = match.group(1)
                ctx_num = match.group(2)
                operation = match.group(3)
                new_id = new_ctx_id if new_ctx_id else f"CTX({ctx_num})"
                return (
                    f"{pfx} ID: {new_id} OPERATION: {operation} CTX-IO-FILE: {path_dst}"
                )

            new_content = re.sub(pattern, replace_with_new_path_and_id, content)
            if new_content != content:
                msg["content"] = new_content

            # Update tool call arguments (assistant messages with tool_calls)
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tool_call in tool_calls:
                    func_info = tool_call.get("function", {})
                    _ = func_info.get("name", "")
                    try:
                        args = json.loads(func_info.get("arguments", "{}"))
                        # Check if path argument references path_src
                        if args.get("path", "") == path_src:
                            args["path"] = path_dst
                            func_info["arguments"] = json.dumps(args)
                        # Check if src argument references path_src (for rename_file)
                        if args.get("src", "") == path_src:
                            args["src"] = path_dst
                            func_info["arguments"] = json.dumps(args)
                        # Check if dst argument references path_src (for rename_file second rename)
                        if args.get("dst", "") == path_src:
                            args["dst"] = path_dst
                            func_info["arguments"] = json.dumps(args)

                    except (json.JSONDecodeError, TypeError):
                        pass

    @override
    def close_file(
        self, path: str, reason: str, llm: Optional[LlmProto] = None
    ) -> str | dict:
        """Handle file close in suffix context mode.

        This method:
        1. Prunes all old tool calls for this file (including read_file)
        2. Replaces all content blocks with a "File closed" message
        3. Updates the context entry to show the file is closed

        Args:
            path: File path to close
            reason: Reason for closing the file
            llm: Optional LLM instance for updating messages

        Returns:
            Success message
        """

        # Get messages to update
        messages = llm.messages() if llm is not None else []

        # Prune all tool calls for this file (including read_file)
        if messages:
            self._prune_old_tool_calls(
                messages,
                [path],
                keep_old_edits=0,  # Prune all
                prune_all_for_paths=[path],  # Prune ALL occurrences
                include_read_file=True,  # Include read_file in pruning
            )

        # Replace all content blocks for this file with "File closed" message
        if messages:
            self._replace_content_blocks(messages, path, reason)

        # Update context entry to show file is closed
        if path in self.file_entries():
            ContextEntry.last_id += 1
            new_id = f"CTX({ContextEntry.last_id})"
            closed_text = f"((File closed, reason: {reason}))"
            self.file_entries()[path] = ContextEntry(
                path, closed_text, new_id, "close_file"
            )

        return f">>> OK: close_file {path}"

    def _replace_content_blocks(self, messages: list[dict], path: str, reason: str):
        """Replace all content blocks for a file with a "File closed" message.

        Args:
            messages: List of message dictionaries
            path: File path to close
            reason: Reason for closing the file
        """
        import re

        pfx = re.escape(self.prefix)
        closed_text = f"{self.prefix} ((File closed, reason: {reason}))"

        # Iterate through all messages
        for msg in messages:
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            # Pattern to match header line for this file
            header_pattern = rf"({pfx}\s*ID:\s*CTX\(\d+\)\s+OPERATION:\s+\w+\s+CTX-IO-FILE:\s+{re.escape(path)})"

            # Find all header matches
            for header_match in re.finditer(header_pattern, content):
                header_start = header_match.start()
                header_end = header_match.end()

                # Find CONTENT START after this header
                content_start_marker = f"{pfx} === CONTENT START ==="
                content_start_idx = content.find(content_start_marker, header_end)
                if content_start_idx == -1:
                    continue

                # Find CONTENT END
                content_end_marker = f"{pfx} === CONTENT END ==="
                content_end_idx = content.find(content_end_marker, content_start_idx)
                if content_end_idx == -1:
                    continue

                # Find end of CONTENT END line
                content_end_line_end = content.find("\n", content_end_idx)
                if content_end_line_end == -1:
                    content_end_line_end = len(content)

                # Replace the entire block
                old_block = content[header_start:content_end_line_end]
                new_block = f"{header_match.group(1)}\n{closed_text}"
                content = content.replace(old_block, new_block, 1)

            # Update message content
            msg["content"] = content
