from context.context_handler import ContextHandler
from typing import override

import config
from context.context_handler import ContextMode, ContextEntry, LlmProto
from context.fold import Fold
from pathlib import Path

SUFFIX_CONTEXTS = {}


class SuffixHandler(ContextHandler):
    """Suffix handler that replaces old file content in messages with references to new content."""

    def __init__(self):
        self._prefix: str = ">>>"
        # Track folds per file: {path: [Fold, ...]}
        self._folds: dict[str, list[Fold]] = {}

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
            sz = Path(config.real_path(path)).stat().st_size
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

            # Check if there are folds for this path
            has_folds = self.has_folds(path)
            # Get the folded content (or original if no folds)
            folded_content = self.format_folded_content(path, ctx.text)
            msg["content"] = self._replace_old_content(
                content, path, ctx.id, folded_content, has_folds
            )

    def _replace_old_content(
        self,
        content: str,
        path: str,
        current_id: str,
        folded_content: str = "",
        has_folds: bool = False,
    ) -> str:
        """Replace file content blocks with current (folded) content.

        Args:
            content: Original message content
            path: File path to replace
            current_id: Current context ID for this file
            folded_content: Folded content to insert
            has_folds: If True, always replace content (to apply folds)
        """
        lines = content.splitlines()
        result_lines: list[str] = []
        i = 0
        pfx = self.prefix
        current_num = self._extract_id(current_id)

        while i < len(lines):
            line = lines[i]

            # Check if this line starts a content block (has the prefix and CTX-IO-FILE)
            if line.startswith(pfx) and "CTX-IO-FILE" in line and path in line:
                old_num = self._extract_id(line)

                # Determine if we should replace:
                # 1. If has_folds is True, always replace (to apply folds)
                # 2. If content has fold markers but has_folds is False, always replace (to remove folds)
                # 3. If old ID < current ID, replace (to update to new content)
                should_replace = False
                if has_folds:
                    should_replace = True
                elif f"{self.prefix} FOLD:" in content:
                    # Content has fold markers but currently has no folds
                    # This means folds were removed, so we need to update
                    should_replace = True
                elif (
                    old_num is not None
                    and current_num is not None
                    and old_num < current_num
                ):
                    should_replace = True

                if should_replace:
                    # Found content to replace, skip everything until CONTENT END
                    header = line
                    i += 1

                    # Skip lines until we find CONTENT END (must start with prefix)
                    while i < len(lines):
                        if (
                            lines[i].startswith(pfx)
                            and "=== CONTENT END ===" in lines[i]
                        ):
                            i += 1
                            break
                        i += 1

                    # Add the header and new reference
                    result_lines.append(header)
                    result_lines.append(
                        f"{pfx} === CURRENT CONTENT IN {current_id} ==="
                    )
                    # Insert the folded (or original) content
                    if folded_content:
                        result_lines.append(folded_content)
                    result_lines.append(f"{pfx} === CONTENT END ===")
                    continue

            result_lines.append(line)
            i += 1

        return "\n".join(result_lines).rstrip()

    def _extract_id(self, text: str) -> int | None:
        """Extract CTX number from text like 'CTX(123)'."""
        import re

        match = re.search(r"CTX\((\d+)\)", text)
        if match:
            return int(match.group(1))
        return None

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
        # Check if file exists in context first
        if path not in SUFFIX_CONTEXTS:
            return {"error": f"File {path} not loaded into context"}

        ctx = SUFFIX_CONTEXTS[path]
        text = ctx.text
        lines = text.splitlines()

        # Validate line numbers are within bounds
        if fold_from_line_num < 1 or fold_from_line_num > len(lines):
            return {
                "error": f"fold_from_line_num {fold_from_line_num} out of range (1..{len(lines)})"
            }
        if fold_to_line_num < 1 or fold_to_line_num > len(lines):
            return {
                "error": f"fold_to_line_num {fold_to_line_num} out of range (1..{len(lines)})"
            }

        # Validate that fold_to_line_num is after fold_from_line_num
        if fold_to_line_num < fold_from_line_num:
            return {
                "error": f"fold_to_line_num ({fold_to_line_num}) must be >= fold_from_line_num ({fold_from_line_num})"
            }

        # Validate that fold_from_line matches the actual line content
        actual_from_line = lines[fold_from_line_num - 1]
        if fold_from_line not in actual_from_line:
            return {
                "error": f"fold_from_line '{fold_from_line}' does not match actual line {fold_from_line_num}: '{actual_from_line}'"
            }

        # Validate that fold_to_line matches the actual line content
        actual_to_line = lines[fold_to_line_num - 1]
        if fold_to_line not in actual_to_line:
            return {
                "error": f"fold_to_line '{fold_to_line}' does not match actual line {fold_to_line_num}: '{actual_to_line}'"
            }

        # Check for duplicate fold name in this file
        if path not in self._folds:
            self._folds[path] = []

        for fold in self._folds[path]:
            if fold.name == name:
                return {"error": f"Fold with name '{name}' already exists in {path}"}

        # Check for overlap with existing folds
        overlap_result = self._check_fold_overlap(
            path, fold_from_line_num, fold_to_line_num
        )
        if overlap_result is not None:
            return overlap_result

        # Create the fold
        fold = Fold(
            name=name,
            start_line=fold_from_line_num,
            end_line=fold_to_line_num,
        )
        self._folds[path].append(fold)

        # Return success message with fold info
        pfx = self.prefix
        ContextEntry.last_id += 1
        new_id = f"CTX({ContextEntry.last_id})"
        return (
            f"{pfx} ID: {new_id} OPERATION: add_fold CTX-IO-FILE:  {path}\n"
            f"{pfx} OK: Added fold '{name}' at lines {fold_from_line_num}..{fold_to_line_num}\n"
            f"{pfx} === FOLD: {name} (lines {fold_from_line_num}..{fold_to_line_num}) ===\n"
        )

    def _check_fold_overlap(
        self, path: str, start_line: int, end_line: int
    ) -> dict | None:
        """Check if a new fold overlaps with existing folds.

        Args:
            path: File path
            start_line: Proposed start line of new fold
            end_line: Proposed end line of new fold

        Returns:
            Error dict if overlap detected, None if valid
        """
        existing_folds = self._folds.get(path, [])

        for fold in existing_folds:
            # Check for overlap: folds must have at least one line buffer between them
            # New fold: [start_line, end_line]
            # Existing fold: [fold.start_line, fold.end_line]
            # Buffer required: at least one line between folds

            # Check if new fold overlaps with existing fold
            # Overlap occurs if:
            # - New fold starts before existing fold ends + 1 (buffer)
            # - New fold ends after existing fold starts - 1 (buffer)

            # New fold would overlap if it touches or intersects existing fold
            if not (end_line < fold.start_line - 1 or start_line > fold.end_line + 1):
                return {
                    "error": f"New fold (lines {start_line}..{end_line}) would overlap with existing fold '{fold.name}' (lines {fold.start_line}..{fold.end_line}). At least one line buffer required between folds."
                }

        return None

    def unfold(self, path: str, name: str) -> dict | str:
        """Remove a fold by name.

        Args:
            path: File path
            name: Name of fold to remove

        Returns:
            Success message or error dict
        """
        if path not in self._folds:
            return {"error": f"No folds found for file {path}"}

        # Find and remove the fold
        folds = self._folds[path]
        for i, fold in enumerate(folds):
            if fold.name == name:
                del folds[i]
                pfx = self.prefix
                new_id = f"CTX({ContextEntry.last_id})"
                return (
                    f"{pfx} ID: {new_id} OPERATION: unfold CTX-IO-FILE:  {path}\n"
                    f"{pfx} OK: Removed fold '{name}'\n"
                )

        return {"error": f"Fold '{name}' not found in {path}"}

    def unfold_all(self, path: str) -> dict | str:
        """Remove all folds from a file.

        Args:
            path: File path

        Returns:
            Success message or error dict
        """
        if path not in self._folds:
            return {"error": f"No folds found for file {path}"}

        del self._folds[path]
        pfx = self.prefix
        new_id = f"CTX({ContextEntry.last_id})"
        return (
            f"{pfx} ID: {new_id} OPERATION: unfold_all CTX-IO-FILE:  {path}\n"
            f"{pfx} OK: Removed all folds from {path}\n"
        )

    def get_folds(self, path: str) -> list[Fold]:
        """Get all folds for a file."""
        return self._folds.get(path, [])

    def has_folds(self, path: str) -> bool:
        """Check if a file has any folds."""
        return path in self._folds and len(self._folds[path]) > 0

    def format_folded_content(self, path: str, text: str) -> str:
        """Format file content with folds applied.

        Args:
            path: File path
            text: Original file content

        Returns:
            Formatted content with fold markers replacing hidden sections
        """
        lines = text.splitlines()
        result_lines: list[str] = []
        folds = self._folds.get(path, [])

        # Sort folds by start_line for proper ordering
        sorted_folds = sorted(folds, key=lambda f: f.start_line)

        line_idx = 0
        for fold in sorted_folds:
            # Add lines before this fold
            while line_idx < fold.start_line - 1 and line_idx < len(lines):
                result_lines.append(lines[line_idx])
                line_idx += 1

            # Add fold marker instead of hidden content
            result_lines.append(
                f"{self.prefix} FOLD: {fold.name} (lines {fold.start_line}..{fold.end_line})"
            )

            # Move to line after the fold ends
            line_idx = fold.end_line

        # Add remaining lines after all folds
        while line_idx < len(lines):
            result_lines.append(lines[line_idx])
            line_idx += 1

        return "\n".join(result_lines)

    def get_visible_lines(self, path: str) -> list[int]:
        """Get all visible (non-folded) line numbers for a file.

        Args:
            path: File path

        Returns:
            List of 1-indexed line numbers that are visible (not folded)
        """
        folds = self._folds.get(path, [])
        if not folds:
            return list(range(1, len(SUFFIX_CONTEXTS[path].text.splitlines()) + 1))

        visible_lines: list[int] = []
        sorted_folds = sorted(folds, key=lambda f: f.start_line)

        prev_end = 0
        for fold in sorted_folds:
            # Add lines before this fold
            for line_num in range(prev_end + 1, fold.start_line):
                visible_lines.append(line_num)
            prev_end = fold.end_line

        # Add lines after all folds
        if path in SUFFIX_CONTEXTS:
            total_lines = len(SUFFIX_CONTEXTS[path].text.splitlines())
            for line_num in range(prev_end + 1, total_lines + 1):
                visible_lines.append(line_num)

        return visible_lines

    def is_line_visible(self, path: str, line_num: int) -> bool:
        """Check if a specific line is visible (not folded).

        Args:
            path: File path
            line_num: 1-indexed line number

        Returns:
            True if the line is visible, False if it's folded
        """
        return line_num in self.get_visible_lines(path)

    def count_occurrences_in_visible(
        self, path: str, text: str
    ) -> tuple[int, list[int]]:
        """Count occurrences of text in visible (non-folded) content.

        Args:
            path: File path
            text: Text to search for (can span multiple lines)

        Returns:
            Tuple of (count, list of 1-indexed line numbers where text starts in visible content)
        """
        if path not in SUFFIX_CONTEXTS:
            return (0, [])

        text_content = SUFFIX_CONTEXTS[path].text
        lines = text_content.splitlines()
        visible_lines = self.get_visible_lines(path)
        visible_line_set = set(visible_lines)

        # Build visible content (concatenating visible lines with newlines)
        visible_content = ""
        for i, line in enumerate(lines):
            line_num = i + 1  # 1-indexed
            if line_num in visible_line_set:
                if visible_content:
                    visible_content += "\n"
                visible_content += line

        # Count occurrences in visible content
        count = 0
        occurrences: list[int] = []
        start = 0
        while True:
            idx = visible_content.find(text, start)
            if idx == -1:
                break
            count += 1
            # Calculate which line this occurrence starts on
            # Count newlines before the match
            newlines_before = visible_content[:idx].count("\n")
            line_num = newlines_before + 1  # 1-indexed
            occurrences.append(line_num)
            start = idx + 1

        return (count, occurrences)

    def is_text_visible(self, path: str, text: str) -> bool:
        """Check if text exists in visible (non-folded) content.

        Args:
            path: File path
            text: Text to search for

        Returns:
            True if text exists in visible content, False otherwise
        """
        count, _ = self.count_occurrences_in_visible(path, text)
        return count > 0

    def update_fold_line_numbers(
        self, path: str, old_line_count: int, new_line_count: int
    ) -> None:
        """Update fold line numbers after file content changes.

        Args:
            path: File path
            old_line_count: Number of lines before edit
            new_line_count: Number of lines after edit
        """
        if path not in self._folds:
            return

        line_diff = new_line_count - old_line_count
        if line_diff == 0:
            return

        folds = self._folds[path]

        for fold in folds:
            # Determine if fold is affected by the change
            # For simplicity, we'll update all folds that start after the edit point
            # This is a simplified approach - in practice, we'd need to know where the edit occurred

            fold.start_line += line_diff
            fold.end_line += line_diff

            # Ensure line numbers stay valid
            fold.start_line = max(1, fold.start_line)
            fold.end_line = max(fold.start_line, fold.end_line)
            fold.end_line = min(new_line_count, fold.end_line)

    def validate_edit_in_visible_content(
        self, path: str, replace_from: str
    ) -> tuple[bool, str]:
        """Validate that an edit target is visible and unique.

        Args:
            path: File path
            replace_from: Text to find

        Returns:
            Tuple of (is_valid, error_message)
        """
        if path not in SUFFIX_CONTEXTS:
            return (False, f"File {path} not in context")

        text_content = SUFFIX_CONTEXTS[path].text

        # First check if text exists at all in the file
        if replace_from not in text_content:
            return (False, f"Text `{repr(replace_from)}` not found in file")

        # Check if text exists in visible content
        count, line_nums = self.count_occurrences_in_visible(path, replace_from)

        if count == 0:
            # Text exists but only in folded content
            return (
                False,
                f"Text `{repr(replace_from)}` exists only in folded (hidden) content",
            )

        if count > 1:
            # Text appears multiple times in visible content
            return (
                False,
                f"Text `{repr(replace_from)}` appears {count} times in visible content (lines {line_nums}), must be unique",
            )

        # Text exists exactly once in visible content - valid edit
        return (True, "")
