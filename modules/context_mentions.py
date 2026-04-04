"""@-mention Context System for Hermes Agent.

Parses @-mentions in user messages and resolves them to contextual data
that gets injected into the conversation before sending to the API.

Supported mentions:
  @diff          -> git diff output (staged + unstaged)
  @git-log, @log -> recent git log (last 10 commits, oneline)
  @tree          -> directory tree (2 levels deep)
  @problems, @errors -> recent test/lint errors from session
  @file:path     -> inject file contents
  @search:pattern -> search results (ripgrep or grep)
  @branch        -> current git branch + status

Integration point: process mentions in the user message before sending
to the API. Resolved context is injected as additional context blocks.
"""

import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Mention:
    """A parsed @-mention from user text."""
    kind: str          # e.g. "diff", "log", "tree", "file", "search", ...
    arg: Optional[str] # e.g. file path for @file:path, pattern for @search:
    start: int         # character offset in original text
    end: int           # character offset end
    raw: str           # the raw matched text

@dataclass
class ResolvedContext:
    """A resolved mention with its output."""
    mention: Mention
    content: str       # the resolved context text
    truncated: bool = False  # whether output was truncated

@dataclass
class MentionResult:
    """Result of resolving all mentions in a message."""
    cleaned_text: str                    # original text with mentions intact
    contexts: List[ResolvedContext] = field(default_factory=list)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum output size per mention (characters)
MAX_MENTION_OUTPUT = 50_000

# Pattern for @-mentions
# Matches: @diff, @git-log, @log, @tree, @problems, @errors, @branch
# Also:    @file:some/path.py  @search:pattern
MENTION_RE = re.compile(
    r"@(diff|git-log|log|tree|problems|errors|branch"
    r"|file:([^\s]+)"
    r"|search:([^\s]+))",
    re.IGNORECASE,
)

# Recent errors storage (module-level, updated by test/lint runners)
_recent_errors: List[str] = []

def set_recent_errors(errors: List[str]) -> None:
    """Store recent test/lint errors for @problems/@errors resolution."""
    global _recent_errors
    _recent_errors = list(errors)

def get_recent_errors() -> List[str]:
    """Get stored recent errors."""
    return list(_recent_errors)

def clear_recent_errors() -> None:
    """Clear stored recent errors."""
    global _recent_errors
    _recent_errors = []

# ---------------------------------------------------------------------------
# MentionResolver
# ---------------------------------------------------------------------------

class MentionResolver:
    """Parses and resolves @-mentions in user messages."""

    def __init__(self, max_output: int = MAX_MENTION_OUTPUT):
        self.max_output = max_output

    def parse_mentions(self, text: str) -> List[Mention]:
        """Parse all @-mentions from text, returning them in order."""
        mentions = []
        for m in MENTION_RE.finditer(text):
            full = m.group(0)
            kind_raw = m.group(1).lower()

            # Determine kind and arg
            if kind_raw.startswith("file:"):
                kind = "file"
                arg = m.group(2)
            elif kind_raw.startswith("search:"):
                kind = "search"
                arg = m.group(3)
            elif kind_raw in ("git-log", "log"):
                kind = "log"
                arg = None
            elif kind_raw in ("problems", "errors"):
                kind = "problems"
                arg = None
            else:
                kind = kind_raw
                arg = None

            mentions.append(Mention(
                kind=kind,
                arg=arg,
                start=m.start(),
                end=m.end(),
                raw=full,
            ))
        return mentions

    def resolve_mention(self, mention: Mention, cwd: str) -> str:
        """Resolve a single mention to its context string."""
        resolvers = {
            "diff": self._resolve_diff,
            "log": self._resolve_log,
            "tree": self._resolve_tree,
            "problems": self._resolve_problems,
            "branch": self._resolve_branch,
            "file": self._resolve_file,
            "search": self._resolve_search,
        }
        resolver = resolvers.get(mention.kind)
        if resolver is None:
            return f"[Unknown mention type: @{mention.kind}]"
        try:
            return resolver(mention, cwd)
        except Exception as e:
            return f"[Error resolving {mention.raw}: {e}]"

    def resolve_all(self, text: str, cwd: str) -> MentionResult:
        """Parse and resolve all mentions in text.

        Returns a MentionResult with the original text and resolved contexts.
        """
        mentions = self.parse_mentions(text)
        contexts = []
        for mention in mentions:
            raw_content = self.resolve_mention(mention, cwd)
            truncated = len(raw_content) > self.max_output
            if truncated:
                raw_content = raw_content[:self.max_output] + "\n... [truncated]"
            contexts.append(ResolvedContext(
                mention=mention,
                content=raw_content,
                truncated=truncated,
            ))
        return MentionResult(cleaned_text=text, contexts=contexts)

    # -- Individual resolvers -----------------------------------------------

    def _run_git(self, args: List[str], cwd: str) -> str:
        """Run a git command and return output."""
        result = subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    def _resolve_diff(self, mention: Mention, cwd: str) -> str:
        """Resolve @diff -> git diff (unstaged + staged)."""
        unstaged = self._run_git(["diff"], cwd)
        staged = self._run_git(["diff", "--cached"], cwd)
        parts = []
        if staged:
            parts.append(f"=== Staged changes ===\n{staged}")
        if unstaged:
            parts.append(f"=== Unstaged changes ===\n{unstaged}")
        if not parts:
            return "[No changes detected]"
        return "\n\n".join(parts)

    def _resolve_log(self, mention: Mention, cwd: str) -> str:
        """Resolve @log/@git-log -> recent git log."""
        return self._run_git(
            ["log", "--oneline", "--no-decorate", "-n", "10"],
            cwd,
        )

    def _resolve_tree(self, mention: Mention, cwd: str) -> str:
        """Resolve @tree -> directory tree (2 levels deep)."""
        # Try 'tree' command first, fall back to find
        try:
            result = subprocess.run(
                ["tree", "-L", "2", "--noreport", "-I",
                 "__pycache__|node_modules|.git|.venv|venv"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except FileNotFoundError:
            pass

        # Fallback: use find
        result = subprocess.run(
            ["find", ".", "-maxdepth", "2", "-not", "-path", "./.git/*",
             "-not", "-path", "./__pycache__/*",
             "-not", "-path", "./node_modules/*"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or "[Empty directory]"

    def _resolve_problems(self, mention: Mention, cwd: str) -> str:
        """Resolve @problems/@errors -> recent test/lint errors."""
        errors = get_recent_errors()
        if not errors:
            return "[No recent errors recorded]"
        return "\n".join(errors[-20:])  # last 20 errors

    def _resolve_branch(self, mention: Mention, cwd: str) -> str:
        """Resolve @branch -> current branch + status."""
        branch = self._run_git(["branch", "--show-current"], cwd)
        status = self._run_git(["status", "--short"], cwd)
        parts = [f"Branch: {branch}"]
        if status:
            parts.append(f"Status:\n{status}")
        else:
            parts.append("Status: clean")
        return "\n".join(parts)

    def _resolve_file(self, mention: Mention, cwd: str) -> str:
        """Resolve @file:path -> file contents."""
        if mention.arg is None:
            return "[No file path specified]"
        filepath = os.path.join(cwd, mention.arg)
        filepath = os.path.expanduser(filepath)
        if not os.path.isfile(filepath):
            return f"[File not found: {mention.arg}]"
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            return f"[Error reading {mention.arg}: {e}]"

    def _resolve_search(self, mention: Mention, cwd: str) -> str:
        """Resolve @search:pattern -> search results."""
        if mention.arg is None:
            return "[No search pattern specified]"
        # Try ripgrep first, fall back to grep
        for cmd in [
            ["rg", "--no-heading", "-n", "--max-count", "20", "--", mention.arg],
            ["grep", "-rn", "--max-count=20", "--", mention.arg, "."],
        ]:
            try:
                result = subprocess.run(
                    cmd,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if result.returncode <= 1:  # 0=found, 1=not found
                    output = result.stdout.strip()
                    return output if output else f"[No results for: {mention.arg}]"
            except FileNotFoundError:
                continue
        return f"[Search unavailable for: {mention.arg}]"


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------

def format_context_for_prompt(result: MentionResult) -> Optional[str]:
    """Format resolved contexts into a string suitable for prompt injection.

    Returns None if no contexts were resolved.
    """
    if not result.contexts:
        return None

    blocks = []
    for ctx in result.contexts:
        header = f"[Context: {ctx.mention.raw}]"
        blocks.append(f"{header}\n{ctx.content}")

    return "\n\n---\n\n".join(blocks)


def process_message_mentions(
    message: str,
    cwd: Optional[str] = None,
) -> Tuple[str, Optional[str]]:
    """High-level API: process a user message for @-mentions.

    Args:
        message: The user's message text
        cwd: Working directory for resolving mentions (defaults to os.getcwd())

    Returns:
        Tuple of (original_message, context_block_or_None)
    """
    if cwd is None:
        cwd = os.getcwd()

    resolver = MentionResolver()
    result = resolver.resolve_all(message, cwd)
    context = format_context_for_prompt(result)
    return message, context
