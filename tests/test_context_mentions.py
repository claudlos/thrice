"""Tests for context_mentions.py - @-mention Context System."""

import os
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new-files"))

from context_mentions import (
    Mention,
    MentionResolver,
    MentionResult,
    ResolvedContext,
    clear_recent_errors,
    format_context_for_prompt,
    get_recent_errors,
    process_message_mentions,
    set_recent_errors,
)


# ---------------------------------------------------------------------------
# parse_mentions tests
# ---------------------------------------------------------------------------

class TestParseMentions:
    """Test mention parsing from text."""

    def setup_method(self):
        self.resolver = MentionResolver()

    def test_parse_diff(self):
        mentions = self.resolver.parse_mentions("Show me @diff please")
        assert len(mentions) == 1
        assert mentions[0].kind == "diff"
        assert mentions[0].arg is None
        assert mentions[0].raw == "@diff"

    def test_parse_log_variants(self):
        mentions = self.resolver.parse_mentions("Check @log and @git-log")
        assert len(mentions) == 2
        assert all(m.kind == "log" for m in mentions)

    def test_parse_tree(self):
        mentions = self.resolver.parse_mentions("Show @tree")
        assert len(mentions) == 1
        assert mentions[0].kind == "tree"

    def test_parse_problems_variants(self):
        mentions = self.resolver.parse_mentions("Fix @problems and @errors")
        assert len(mentions) == 2
        assert all(m.kind == "problems" for m in mentions)

    def test_parse_branch(self):
        mentions = self.resolver.parse_mentions("What's on @branch?")
        assert len(mentions) == 1
        assert mentions[0].kind == "branch"

    def test_parse_file_mention(self):
        mentions = self.resolver.parse_mentions("Read @file:src/main.py")
        assert len(mentions) == 1
        assert mentions[0].kind == "file"
        assert mentions[0].arg == "src/main.py"

    def test_parse_search_mention(self):
        mentions = self.resolver.parse_mentions("Find @search:TODO")
        assert len(mentions) == 1
        assert mentions[0].kind == "search"
        assert mentions[0].arg == "TODO"

    def test_parse_multiple_mentions(self):
        text = "Check @diff then @log and fix @file:test.py"
        mentions = self.resolver.parse_mentions(text)
        assert len(mentions) == 3
        assert mentions[0].kind == "diff"
        assert mentions[1].kind == "log"
        assert mentions[2].kind == "file"

    def test_parse_no_mentions(self):
        mentions = self.resolver.parse_mentions("Just a regular message")
        assert len(mentions) == 0

    def test_parse_case_insensitive(self):
        mentions = self.resolver.parse_mentions("Show @DIFF and @Tree")
        assert len(mentions) == 2

    def test_positions_recorded(self):
        text = "Hello @diff world"
        mentions = self.resolver.parse_mentions(text)
        assert mentions[0].start == 6
        assert mentions[0].end == 11


# ---------------------------------------------------------------------------
# resolve_mention tests
# ---------------------------------------------------------------------------

class TestResolveMention:
    """Test individual mention resolution."""

    def setup_method(self):
        self.resolver = MentionResolver()
        self.tmpdir = tempfile.mkdtemp()

    def test_resolve_file_existing(self):
        # Create a temp file
        filepath = os.path.join(self.tmpdir, "test.txt")
        with open(filepath, "w") as f:
            f.write("hello world")

        mention = Mention(kind="file", arg="test.txt", start=0, end=0, raw="@file:test.txt")
        result = self.resolver.resolve_mention(mention, self.tmpdir)
        assert result == "hello world"

    def test_resolve_file_not_found(self):
        mention = Mention(kind="file", arg="nonexistent.py", start=0, end=0, raw="@file:nonexistent.py")
        result = self.resolver.resolve_mention(mention, self.tmpdir)
        assert "[File not found" in result

    def test_resolve_file_no_arg(self):
        mention = Mention(kind="file", arg=None, start=0, end=0, raw="@file:")
        result = self.resolver.resolve_mention(mention, self.tmpdir)
        assert "[No file path" in result

    def test_resolve_problems_empty(self):
        clear_recent_errors()
        mention = Mention(kind="problems", arg=None, start=0, end=0, raw="@problems")
        result = self.resolver.resolve_mention(mention, self.tmpdir)
        assert "[No recent errors" in result

    def test_resolve_problems_with_errors(self):
        set_recent_errors(["Error: test failed", "Error: lint warning"])
        mention = Mention(kind="problems", arg=None, start=0, end=0, raw="@problems")
        result = self.resolver.resolve_mention(mention, self.tmpdir)
        assert "test failed" in result
        assert "lint warning" in result
        clear_recent_errors()

    @patch("context_mentions.subprocess.run")
    def test_resolve_diff_no_changes(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr=""
        )
        mention = Mention(kind="diff", arg=None, start=0, end=0, raw="@diff")
        result = self.resolver.resolve_mention(mention, self.tmpdir)
        assert "[No changes" in result

    @patch("context_mentions.subprocess.run")
    def test_resolve_diff_with_changes(self, mock_run):
        def side_effect(cmd, **kwargs):
            if "--cached" in cmd:
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="staged diff", stderr="")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="unstaged diff", stderr="")
        mock_run.side_effect = side_effect
        mention = Mention(kind="diff", arg=None, start=0, end=0, raw="@diff")
        result = self.resolver.resolve_mention(mention, self.tmpdir)
        assert "staged diff" in result
        assert "unstaged diff" in result

    @patch("context_mentions.subprocess.run")
    def test_resolve_log(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="abc1234 first commit\ndef5678 second commit", stderr=""
        )
        mention = Mention(kind="log", arg=None, start=0, end=0, raw="@log")
        result = self.resolver.resolve_mention(mention, self.tmpdir)
        assert "first commit" in result
        assert "second commit" in result

    @patch("context_mentions.subprocess.run")
    def test_resolve_branch(self, mock_run):
        call_count = [0]
        def side_effect(cmd, **kwargs):
            call_count[0] += 1
            if "branch" in cmd:
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="main", stderr="")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="M file.py", stderr="")
        mock_run.side_effect = side_effect
        mention = Mention(kind="branch", arg=None, start=0, end=0, raw="@branch")
        result = self.resolver.resolve_mention(mention, self.tmpdir)
        assert "main" in result
        assert "file.py" in result

    def test_resolve_unknown_kind(self):
        mention = Mention(kind="unknown", arg=None, start=0, end=0, raw="@unknown")
        result = self.resolver.resolve_mention(mention, self.tmpdir)
        assert "[Unknown mention" in result


# ---------------------------------------------------------------------------
# resolve_all tests
# ---------------------------------------------------------------------------

class TestResolveAll:
    """Test full mention resolution pipeline."""

    def setup_method(self):
        self.resolver = MentionResolver()
        self.tmpdir = tempfile.mkdtemp()

    def test_resolve_all_no_mentions(self):
        result = self.resolver.resolve_all("Just a message", self.tmpdir)
        assert isinstance(result, MentionResult)
        assert result.cleaned_text == "Just a message"
        assert result.contexts == []

    def test_resolve_all_with_file(self):
        filepath = os.path.join(self.tmpdir, "hello.txt")
        with open(filepath, "w") as f:
            f.write("content here")

        result = self.resolver.resolve_all("Read @file:hello.txt", self.tmpdir)
        assert len(result.contexts) == 1
        assert result.contexts[0].content == "content here"
        assert result.contexts[0].mention.kind == "file"

    def test_resolve_all_truncation(self):
        resolver = MentionResolver(max_output=20)
        filepath = os.path.join(self.tmpdir, "big.txt")
        with open(filepath, "w") as f:
            f.write("x" * 100)

        result = resolver.resolve_all("Read @file:big.txt", self.tmpdir)
        assert result.contexts[0].truncated is True
        assert "[truncated]" in result.contexts[0].content


# ---------------------------------------------------------------------------
# Integration helper tests
# ---------------------------------------------------------------------------

class TestFormatContext:
    """Test context formatting for prompt injection."""

    def test_format_no_contexts(self):
        result = MentionResult(cleaned_text="hello", contexts=[])
        assert format_context_for_prompt(result) is None

    def test_format_single_context(self):
        ctx = ResolvedContext(
            mention=Mention(kind="diff", arg=None, start=0, end=5, raw="@diff"),
            content="some diff output",
        )
        result = MentionResult(cleaned_text="Show @diff", contexts=[ctx])
        formatted = format_context_for_prompt(result)
        assert formatted is not None
        assert "[Context: @diff]" in formatted
        assert "some diff output" in formatted

    def test_format_multiple_contexts(self):
        ctx1 = ResolvedContext(
            mention=Mention(kind="diff", arg=None, start=0, end=5, raw="@diff"),
            content="diff output",
        )
        ctx2 = ResolvedContext(
            mention=Mention(kind="log", arg=None, start=10, end=14, raw="@log"),
            content="log output",
        )
        result = MentionResult(cleaned_text="text", contexts=[ctx1, ctx2])
        formatted = format_context_for_prompt(result)
        assert "diff output" in formatted
        assert "log output" in formatted
        assert "---" in formatted  # separator


class TestProcessMessageMentions:
    """Test the high-level API."""

    def test_no_mentions(self):
        msg, ctx = process_message_mentions("Hello world", "/tmp")
        assert msg == "Hello world"
        assert ctx is None

    def test_with_problems(self):
        set_recent_errors(["test error 1"])
        msg, ctx = process_message_mentions("Fix @problems", "/tmp")
        assert msg == "Fix @problems"
        assert ctx is not None
        assert "test error 1" in ctx
        clear_recent_errors()


# ---------------------------------------------------------------------------
# Error storage tests
# ---------------------------------------------------------------------------

class TestErrorStorage:
    """Test recent error storage."""

    def test_set_and_get(self):
        set_recent_errors(["error1", "error2"])
        assert get_recent_errors() == ["error1", "error2"]
        clear_recent_errors()

    def test_clear(self):
        set_recent_errors(["error1"])
        clear_recent_errors()
        assert get_recent_errors() == []

    def test_isolation(self):
        """Setting errors creates a copy."""
        original = ["error1"]
        set_recent_errors(original)
        original.append("error2")
        assert get_recent_errors() == ["error1"]
        clear_recent_errors()
