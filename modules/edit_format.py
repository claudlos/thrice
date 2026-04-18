"""
Edit Format Optimization — Selecting the best edit format for LLM-generated code changes.

Different edit formats work better for different situations:
  - WHOLE_FILE for small files (<80 lines)
  - UNIFIED_DIFF for large files with multi-hunk changes
  - XML_TAGGED for complex multi-region edits
  - SEARCH_REPLACE as the reliable default

This module provides:
  1. EditFormat enum and dataclasses (EditOperation, EditRegion)
  2. EditFormatSelector — picks the best format based on file/change characteristics
  3. FormatGenerator — produces edit instructions in each format
  4. FormatParser — parses LLM-generated edit text back into operations
  5. FormatTracker — tracks success/failure rates per format per model

Integration point: Use EditFormatSelector.select_format() before sending code
edit requests to the LLM, and FormatParser to process responses.

Usage:
    from new_files.edit_format import (
        EditFormat, EditFormatSelector, FormatGenerator, FormatParser,
        FormatTracker, EditOperation, EditRegion,
    )

    selector = EditFormatSelector()
    fmt = selector.select_format("app.py", "add logging", 45, "single_function")
    # => EditFormat.WHOLE_FILE (small file)
"""

import difflib
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

# ─── Enums ──────────────────────────────────────────────────────────────────


class EditFormat(Enum):
    """Supported edit formats for LLM code changes."""
    SEARCH_REPLACE = "search_replace"
    WHOLE_FILE = "whole_file"
    UNIFIED_DIFF = "unified_diff"
    XML_TAGGED = "xml_tagged"


# ─── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class EditOperation:
    """A single edit operation parsed from LLM output.

    Attributes:
        file_path: Path to the file being edited.
        old_content: Original content to be replaced.
        new_content: New content to replace with.
        start_line: Optional 1-indexed start line.
        end_line: Optional 1-indexed end line.
    """
    file_path: str
    old_content: str
    new_content: str
    start_line: Optional[int] = None
    end_line: Optional[int] = None


@dataclass
class EditRegion:
    """A labeled region for XML-tagged edits.

    Attributes:
        label: Descriptive label for the region.
        start_line: 1-indexed start line.
        end_line: 1-indexed end line.
        old_content: Original content in the region.
        new_content: Replacement content.
    """
    label: str
    start_line: int
    end_line: int
    old_content: str
    new_content: str


# ─── Selector ───────────────────────────────────────────────────────────────


@dataclass
class EditFormatSelector:
    """Selects the best edit format based on file and change characteristics.

    Configurable thresholds control when each format is preferred.

    Attributes:
        whole_file_threshold: Files with fewer lines than this use WHOLE_FILE.
        diff_hunk_keywords: Keywords in change_description that suggest multi-hunk.
        multi_region_keywords: Keywords that suggest complex multi-region edits.
        large_file_threshold: Files above this line count are considered large.
    """
    whole_file_threshold: int = 80
    large_file_threshold: int = 300
    diff_hunk_keywords: Tuple[str, ...] = (
        "multiple", "several", "many", "various", "across", "scattered",
        "multi-hunk", "hunks", "throughout",
    )
    multi_region_keywords: Tuple[str, ...] = (
        "complex", "multi-region", "restructure", "refactor", "reorganize",
        "move and modify", "rearrange", "interleaved",
    )

    def select_format(
        self,
        file_path: str,
        change_description: str,
        file_size_lines: int,
        change_scope: str,
    ) -> EditFormat:
        """Select the best edit format for a given change.

        Args:
            file_path: Path to the file being edited.
            change_description: Natural language description of the change.
            file_size_lines: Number of lines in the file.
            change_scope: One of "single_line", "single_function", "multi_function",
                          "multi_region", "whole_file".

        Returns:
            The recommended EditFormat.
        """
        # Rule 1: Small files → WHOLE_FILE
        if file_size_lines < self.whole_file_threshold:
            return EditFormat.WHOLE_FILE

        # Rule 2: Explicit whole-file scope
        if change_scope == "whole_file":
            return EditFormat.WHOLE_FILE

        desc_lower = change_description.lower()

        # Rule 3: Complex multi-region edits → XML_TAGGED
        if change_scope == "multi_region":
            return EditFormat.XML_TAGGED
        if any(kw in desc_lower for kw in self.multi_region_keywords):
            return EditFormat.XML_TAGGED

        # Rule 4: Large files with multi-hunk changes → UNIFIED_DIFF
        if file_size_lines >= self.large_file_threshold:
            if change_scope == "multi_function":
                return EditFormat.UNIFIED_DIFF
            if any(kw in desc_lower for kw in self.diff_hunk_keywords):
                return EditFormat.UNIFIED_DIFF

        # Default: SEARCH_REPLACE
        return EditFormat.SEARCH_REPLACE


# ─── Generator ──────────────────────────────────────────────────────────────


class FormatGenerator:
    """Generates edit instructions in each supported format."""

    @staticmethod
    def generate_search_replace(
        old: str,
        new: str,
        context_lines: int = 3,
    ) -> str:
        """Generate a SEARCH/REPLACE block.

        Args:
            old: The original text to search for.
            new: The replacement text.
            context_lines: Number of context lines to include (for display hint).

        Returns:
            Formatted SEARCH/REPLACE block string.
        """
        lines = []
        lines.append("<<<<<<< SEARCH")
        lines.append(old)
        lines.append("=======")
        lines.append(new)
        lines.append(">>>>>>> REPLACE")
        return "\n".join(lines)

    @staticmethod
    def generate_whole_file(content: str, file_path: str) -> str:
        """Generate a whole-file replacement block.

        Args:
            content: The complete new file content.
            file_path: Path to the file.

        Returns:
            Formatted whole-file block string.
        """
        lines = []
        lines.append(f"--- {file_path}")
        lines.append("```")
        lines.append(content)
        lines.append("```")
        return "\n".join(lines)

    @staticmethod
    def generate_unified_diff(
        original: str,
        modified: str,
        file_path: str,
    ) -> str:
        """Generate a unified diff between original and modified content.

        Args:
            original: The original file content.
            modified: The modified file content.
            file_path: Path to the file (used in diff header).

        Returns:
            Unified diff string.
        """
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        )
        return "\n".join(line.rstrip("\n") for line in diff)

    @staticmethod
    def generate_xml_tagged(edits: List[EditRegion]) -> str:
        """Generate XML-tagged edit instructions.

        Args:
            edits: List of EditRegion objects describing each edit.

        Returns:
            XML-tagged edit string.
        """
        lines = ["<edits>"]
        for edit in edits:
            lines.append("  <edit>")
            lines.append(f"    <label>{edit.label}</label>")
            lines.append(f"    <lines start=\"{edit.start_line}\" end=\"{edit.end_line}\" />")
            lines.append("    <old>")
            lines.append(f"      {edit.old_content}")
            lines.append("    </old>")
            lines.append("    <new>")
            lines.append(f"      {edit.new_content}")
            lines.append("    </new>")
            lines.append("  </edit>")
        lines.append("</edits>")
        return "\n".join(lines)


# ─── Parser ─────────────────────────────────────────────────────────────────


# Regex patterns for parsing
_SEARCH_REPLACE_RE = re.compile(
    r"<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE",
    re.DOTALL,
)

_UNIFIED_DIFF_HEADER_RE = re.compile(
    r"^---\s+a/(.+?)$",
    re.MULTILINE,
)

_UNIFIED_DIFF_HUNK_RE = re.compile(
    r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@",
    re.MULTILINE,
)

_XML_EDIT_RE = re.compile(
    r"<edit>\s*"
    r"<label>(.*?)</label>\s*"
    r"<lines\s+start=\"(\d+)\"\s+end=\"(\d+)\"\s*/>\s*"
    r"<old>\s*(.*?)\s*</old>\s*"
    r"<new>\s*(.*?)\s*</new>\s*"
    r"</edit>",
    re.DOTALL,
)

_WHOLE_FILE_RE = re.compile(
    r"^---\s+(.+?)$\n```\n(.*?)\n```",
    re.MULTILINE | re.DOTALL,
)


class FormatParser:
    """Parses LLM-generated edit text back into EditOperation objects."""

    @staticmethod
    def parse_search_replace(text: str) -> List[EditOperation]:
        """Parse SEARCH/REPLACE blocks from text.

        Args:
            text: Text containing one or more SEARCH/REPLACE blocks.

        Returns:
            List of EditOperation objects.
        """
        operations = []
        for match in _SEARCH_REPLACE_RE.finditer(text):
            old_content = match.group(1)
            new_content = match.group(2)
            operations.append(EditOperation(
                file_path="",
                old_content=old_content,
                new_content=new_content,
            ))
        return operations

    @staticmethod
    def parse_unified_diff(text: str) -> List[EditOperation]:
        """Parse a unified diff into EditOperation objects.

        Each hunk becomes a separate EditOperation with line numbers.

        Args:
            text: Text containing a unified diff.

        Returns:
            List of EditOperation objects.
        """
        operations = []

        # Find the file path from the --- line
        file_match = _UNIFIED_DIFF_HEADER_RE.search(text)
        file_path = file_match.group(1) if file_match else ""

        lines = text.splitlines()
        i = 0
        while i < len(lines):
            hunk_match = _UNIFIED_DIFF_HUNK_RE.match(lines[i])
            if hunk_match:
                start_line = int(hunk_match.group(1))
                old_count = int(hunk_match.group(2) or 1)
                i += 1

                old_lines = []
                new_lines = []

                while i < len(lines):
                    line = lines[i]
                    if line.startswith("@@") or line.startswith("---") or line.startswith("+++"):
                        break
                    if line.startswith("-"):
                        old_lines.append(line[1:])
                    elif line.startswith("+"):
                        new_lines.append(line[1:])
                    elif line.startswith(" "):
                        old_lines.append(line[1:])
                        new_lines.append(line[1:])
                    elif line == "\\ No newline at end of file":
                        pass
                    else:
                        # Context line without prefix
                        old_lines.append(line)
                        new_lines.append(line)
                    i += 1

                operations.append(EditOperation(
                    file_path=file_path,
                    old_content="\n".join(old_lines),
                    new_content="\n".join(new_lines),
                    start_line=start_line,
                    end_line=start_line + old_count - 1,
                ))
            else:
                i += 1

        return operations

    @staticmethod
    def parse_xml_tagged(text: str) -> List[EditOperation]:
        """Parse XML-tagged edit blocks.

        Args:
            text: Text containing <edits>...</edits> XML.

        Returns:
            List of EditOperation objects.
        """
        operations = []
        for match in _XML_EDIT_RE.finditer(text):
            # match.group(1) is the block label; kept in the regex for symmetry
            # with other formats but not used here.
            start_line = int(match.group(2))
            end_line = int(match.group(3))
            old_content = match.group(4).strip()
            new_content = match.group(5).strip()
            operations.append(EditOperation(
                file_path="",
                old_content=old_content,
                new_content=new_content,
                start_line=start_line,
                end_line=end_line,
            ))
        return operations

    @staticmethod
    def parse_whole_file(text: str) -> List[EditOperation]:
        """Parse whole-file replacement blocks.

        Args:
            text: Text containing --- path + ``` content ``` blocks.

        Returns:
            List of EditOperation objects.
        """
        operations = []
        for match in _WHOLE_FILE_RE.finditer(text):
            file_path = match.group(1).strip()
            new_content = match.group(2)
            operations.append(EditOperation(
                file_path=file_path,
                old_content="",
                new_content=new_content,
            ))
        return operations

    @staticmethod
    def auto_detect_format(text: str) -> EditFormat:
        """Auto-detect which edit format is present in the text.

        Uses characteristic markers to identify the format.

        Args:
            text: LLM-generated text containing edit instructions.

        Returns:
            The detected EditFormat.
        """
        # Check for SEARCH/REPLACE markers
        if "<<<<<<< SEARCH" in text and ">>>>>>> REPLACE" in text:
            return EditFormat.SEARCH_REPLACE

        # Check for XML tags
        if "<edits>" in text and "<edit>" in text:
            return EditFormat.XML_TAGGED

        # Check for unified diff markers
        if text.lstrip().startswith("---") and "+++" in text and "@@" in text:
            return EditFormat.UNIFIED_DIFF

        # Check for whole-file markers
        if _WHOLE_FILE_RE.search(text):
            return EditFormat.WHOLE_FILE

        # Fallback: if there are diff-like lines
        lines = text.splitlines()
        diff_indicators = sum(
            1 for line in lines
            if line.startswith("+") or line.startswith("-")
        )
        if diff_indicators > len(lines) * 0.3:
            return EditFormat.UNIFIED_DIFF

        # Default
        return EditFormat.SEARCH_REPLACE


# ─── Tracker ────────────────────────────────────────────────────────────────


@dataclass
class _FormatStats:
    """Tracks success/failure counts for a format."""
    successes: int = 0
    failures: int = 0

    @property
    def total(self) -> int:
        return self.successes + self.failures

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.successes / self.total


@dataclass
class FormatTracker:
    """Tracks success/failure rates per format per model.

    Useful for learning which formats work best for each LLM model
    and recommending the most reliable one.

    Attributes:
        stats: Nested dict of model -> format -> stats.
        min_samples: Minimum number of samples before making recommendations.
    """
    stats: Dict[str, Dict[EditFormat, _FormatStats]] = field(default_factory=dict)
    min_samples: int = 5

    def _ensure_model(self, model: str) -> None:
        """Ensure the model entry exists in stats."""
        if model not in self.stats:
            self.stats[model] = {}

    def _ensure_format(self, model: str, fmt: EditFormat) -> None:
        """Ensure the format entry exists for a model."""
        self._ensure_model(model)
        if fmt not in self.stats[model]:
            self.stats[model][fmt] = _FormatStats()

    def record_success(self, model: str, fmt: EditFormat) -> None:
        """Record a successful edit using the given format and model.

        Args:
            model: LLM model identifier (e.g., "gpt-4", "claude-3").
            fmt: The edit format that was used.
        """
        self._ensure_format(model, fmt)
        self.stats[model][fmt].successes += 1

    def record_failure(self, model: str, fmt: EditFormat) -> None:
        """Record a failed edit using the given format and model.

        Args:
            model: LLM model identifier.
            fmt: The edit format that was used.
        """
        self._ensure_format(model, fmt)
        self.stats[model][fmt].failures += 1

    def get_success_rate(self, model: str, fmt: EditFormat) -> float:
        """Get the success rate for a model/format combination.

        Args:
            model: LLM model identifier.
            fmt: The edit format.

        Returns:
            Success rate as a float between 0.0 and 1.0.
            Returns 0.0 if no data is available.
        """
        if model not in self.stats or fmt not in self.stats[model]:
            return 0.0
        return self.stats[model][fmt].success_rate

    def get_total_attempts(self, model: str, fmt: EditFormat) -> int:
        """Get total attempts for a model/format combination.

        Args:
            model: LLM model identifier.
            fmt: The edit format.

        Returns:
            Total number of attempts.
        """
        if model not in self.stats or fmt not in self.stats[model]:
            return 0
        return self.stats[model][fmt].total

    def recommend_format(self, model: str) -> EditFormat:
        """Recommend the best edit format for a model based on historical data.

        Only considers formats with at least min_samples attempts.
        Falls back to SEARCH_REPLACE if insufficient data.

        Args:
            model: LLM model identifier.

        Returns:
            The recommended EditFormat.
        """
        if model not in self.stats:
            return EditFormat.SEARCH_REPLACE

        best_format = EditFormat.SEARCH_REPLACE
        best_rate = -1.0

        for fmt, fmt_stats in self.stats[model].items():
            if fmt_stats.total >= self.min_samples:
                rate = fmt_stats.success_rate
                if rate > best_rate:
                    best_rate = rate
                    best_format = fmt

        return best_format

    def get_summary(self, model: str) -> Dict[str, Dict[str, float]]:
        """Get a summary of all format stats for a model.

        Args:
            model: LLM model identifier.

        Returns:
            Dict mapping format name -> {"success_rate": float, "total": int}.
        """
        if model not in self.stats:
            return {}

        summary = {}
        for fmt, fmt_stats in self.stats[model].items():
            summary[fmt.value] = {
                "success_rate": fmt_stats.success_rate,
                "total": float(fmt_stats.total),
            }
        return summary
