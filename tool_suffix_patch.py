from typing import Annotated
from config import real_path, READ_ONLY_FILES, READ_ONLY_ERROR
from tool import Tool
from tool_io import ToolWriteFile


class ToolPatchSuffix(Tool):
    def __init__(self, sfx_remove: str = "<<remove>>", sfx_add: str = "<<add>>"):
        self.sfx_remove = sfx_remove
        self.sfx_add = sfx_add
        super().__init__(
            name="patch_suffix",
            description=f"""Apply a patch to a file using special suffix syntax.

SYNTAX:
  - Context section (<<context>>): Lines prefixed with |, stripped of | before matching
  - Add sections (<<add>>): Lines prefixed with | to insert at marked positions
  - Suffix {sfx_remove}: Remove the line from output
  - Suffix {sfx_add}: Keep line, insert next <<add>> section after it
  - Combined {sfx_remove}{sfx_add}: Remove line, insert next <<add>> section in its place
  - No suffix: Keep line as-is

EXAMPLE:
  Original file (example.txt):
    line1
    line2
    line3
    line4
    line5

  Patch to remove line2, add after line1, replace line4:
    <<context>>
    |line1{self.sfx_add}
    |line2{self.sfx_remove}
    |line3
    |line4{self.sfx_remove}{self.sfx_add}
    |line5
    <<add>>
    |new_line_after_1
    <<add>>
    |new_line_a
    |new_line_b

  Result:
    line1
    new_line_after_1
    line3
    new_line_a
    new_line_b
    line5

SAFETY: File must not have {sfx_remove} or {sfx_add} as line suffixes.
ALL <<add>> sections must be consumed (triggered by {sfx_add} in context).""",
        )

    def __call__(
        self,
        path: Annotated[str, "Project path to patch"],
        patch: Annotated[str, "Patch content with <<context>> and <<add>> sections"],
    ):
        p = real_path(path)

        if p in READ_ONLY_FILES:
            return {path: "error", "error": READ_ONLY_ERROR}

        if p.exists() and not p.is_file():
            return {path: "error", "error": f"File exists, but it is not a file"}

        # Read original content
        original_content = ""
        if p.exists():
            original_content = p.read_text()

        # Safety check: file should not have suffixes at end of any line
        for line in original_content.splitlines():
            if line.endswith(self.sfx_remove) or line.endswith(self.sfx_add):
                return {
                    path: "error",
                    "error": f"File contains forbidden suffix {self.sfx_remove} or {self.sfx_add} at end of line",
                }

        # Apply patch
        try:
            new_content = self._apply_patch(original_content, patch)
        except ValueError as e:
            return {path: "error", "error": str(e)}

        # TODO: check folds
        write = ToolWriteFile()
        return write(path, new_content)

    def _apply_patch(self, original_text: str, patch: str) -> str:
        """Apply patch to original text.

        Args:
            original_text: Original file content
            patch: Patch content with <<context>> and <<add>> sections

        Returns:
            New file content after applying patch

        Raises:
            ValueError: If patch is invalid or cannot be applied
        """
        # Parse patch into sections
        lines = patch.splitlines()

        # Find all section markers and their content
        context_lines = []
        add_sections = []  # List of lists, each inner list is lines in an add section
        current_section = None
        current_add_section = []

        i = 0
        while i < len(lines):
            line = lines[i]

            if line == "<<context>>":
                current_section = "context"
                i += 1
                continue
            elif line == "<<add>>":
                # Save any current add section and start new one
                if current_add_section:
                    add_sections.append(current_add_section)
                    current_add_section = []
                current_section = "add"
                i += 1
                continue

            if current_section == "context":
                if line.startswith("|"):
                    context_lines.append(line[1:])  # Strip pipe
                i += 1
            elif current_section == "add":
                if line.startswith("|"):
                    current_add_section.append(line[1:])  # Strip pipe
                else:
                    # Empty line in add section - add empty string
                    current_add_section.append("")
                i += 1
            else:
                # Lines before any section marker - skip
                i += 1

        # Save the last add section if any
        if current_add_section:
            add_sections.append(current_add_section)

        # Process context lines to extract operations
        # Each context line can have:
        # - No suffix: use as is
        # - {sfx_remove}: mark for removal
        # - {sfx_add}: trigger add section insertion after this line
        # - {sfx_remove}{sfx_add}: remove and insert add section

        # Parse context lines
        parsed_context = []
        for line in context_lines:
            if line.endswith(self.sfx_remove + self.sfx_add):
                # Remove and add
                content = line[: -len(self.sfx_remove) - len(self.sfx_add)]
                parsed_context.append((content, "remove_add"))
            elif line.endswith(self.sfx_remove):
                # Just remove
                content = line[: -len(self.sfx_remove)]
                parsed_context.append((content, "remove"))
            elif line.endswith(self.sfx_add):
                # Just add (keep line, insert add section after)
                content = line[: -len(self.sfx_add)]
                parsed_context.append((content, "add"))
            else:
                # No suffix, use as is
                parsed_context.append((line, "keep"))

        # Find context in original text
        orig_lines = original_text.splitlines()

        # Special case: no context but only add sections
        if not parsed_context:
            # Just append all add sections to the file
            start_idx, end_idx = len(orig_lines), len(orig_lines)
        # Special case: empty file
        elif not original_text:
            # Context must be empty or all empty lines
            if all(c[0] == "" for c in parsed_context):
                start_idx, end_idx = 0, 0
            else:
                raise ValueError(f"Context not found in empty file")
        else:
            # Try to find context in original
            search_results = self._find_context_in_file(
                orig_lines, [c[0] for c in parsed_context]
            )

            if search_results is None:
                raise ValueError(f"Could not find context in file")

            start_idx, end_idx = search_results

        # Build new content
        result_lines = []

        # Add lines before context
        result_lines.extend(orig_lines[:start_idx])

        # Process context lines and apply operations
        add_section_idx = 0

        # Special case: no context lines, just append all add sections at the end
        if not parsed_context:
            for add_section in add_sections:
                result_lines.extend(add_section)
            add_section_idx = len(add_sections)
        else:
            # Process context lines
            for i, (content, operation) in enumerate(parsed_context):
                if operation == "remove":
                    # Skip this line (don't add it)
                    pass
                elif operation == "remove_add":
                    # Skip this line and insert add section
                    if add_section_idx < len(add_sections):
                        result_lines.extend(add_sections[add_section_idx])
                        add_section_idx += 1
                    else:
                        raise ValueError(
                            f"Not enough <<add>> sections for {self.sfx_add} suffix"
                        )
                elif operation == "add":
                    # Keep this line and insert add section after it
                    result_lines.append(content)
                    if add_section_idx < len(add_sections):
                        result_lines.extend(add_sections[add_section_idx])
                        add_section_idx += 1
                    else:
                        raise ValueError(
                            f"Not enough <<add>> sections for {self.sfx_add} suffix"
                        )
                else:  # keep
                    # Just add the line as is
                    result_lines.append(content)

            # Add lines after context
            result_lines.extend(orig_lines[end_idx:])

        # Check all add sections were consumed
        if add_section_idx != len(add_sections):
            raise ValueError(
                f"Not all <<add>> sections were consumed. {len(add_sections) - add_section_idx} remaining."
            )

        # Join with newlines
        result = "\n".join(result_lines)

        # Preserve trailing newline
        if result_lines:
            # If original had trailing newline, preserve it
            # If original was empty but we added content, add trailing newline
            if original_text.endswith("\n") or (not original_text and result_lines):
                result += "\n"

        return result

    def _find_context_in_file(
        self, orig_lines: list[str], context_lines: list[str]
    ) -> tuple[int, int] | None:
        """Find context_lines in orig_lines.

        Returns:
            (start_idx, end_idx) if found exactly once, where end_idx is exclusive
            None if not found or found multiple times
        """
        if not context_lines:
            return (0, 0)

        context_len = len(context_lines)
        if context_len > len(orig_lines):
            return None

        matches = []
        for i in range(len(orig_lines) - context_len + 1):
            found = True
            for j, line in enumerate(context_lines):
                if orig_lines[i + j] != line:
                    found = False
                    break
            if found:
                matches.append((i, i + context_len))

        if len(matches) == 0:
            return None
        elif len(matches) == 1:
            return matches[0]
        else:
            return None  # Multiple matches - ambiguous
