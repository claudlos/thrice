"""Tests for ``secret_scanner``."""
from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_MODULES = os.path.normpath(os.path.join(_HERE, "..", "modules"))
if _MODULES not in sys.path:
    sys.path.insert(0, _MODULES)

from secret_scanner import (  # noqa: E402
    scan_diff,
    scan_files,
    scan_text,
    shannon_entropy,
)

# AWS-key-id-shaped fixture, split across concatenation so GitHub push
# protection doesn't flag this test file as committing a real secret.
_AKIA_FIXTURE = "AKIA" + "ABCDEFGHIJKLMNOP"

# ---------------------------------------------------------------------------
# Pattern rules
# ---------------------------------------------------------------------------

class TestRules:

    @pytest.mark.parametrize(
        "text,rule",
        [
            (_AKIA_FIXTURE, "aws_access_key_id"),
            ('ghp_' + 'a' * 40, "github_token"),
            ('ghs_' + 'a' * 40, "github_app_token"),
            ("xoxb-12345-abcdefghij", "slack_token"),
            ("sk_live_" + "a" * 30, "stripe_live_key"),
            ("sk_test_" + "a" * 30, "stripe_test_key"),
            ("AIza" + "A" * 35, "google_api_key"),
            # real-looking but bogus Anthropic token (>= 80 trailing chars)
            ("sk-ant-api03-" + "A" * 90, "anthropic_key"),
            ("-----BEGIN PRIVATE KEY-----", "private_key_block"),
            ("-----BEGIN RSA PRIVATE KEY-----", "private_key_block"),
        ],
    )
    def test_rule_fires(self, text, rule):
        findings = scan_text(text)
        names = {f.rule for f in findings}
        assert rule in names, f"expected {rule}, got {names}"

    def test_jwt_detected(self):
        jwt = "eyJabc123.eyJxyz456.signaturetrailerpart"
        findings = scan_text(jwt)
        assert any(f.rule == "jwt" for f in findings)

    def test_password_assignment(self):
        findings = scan_text('password = "hunter22shhh"')
        assert any(f.rule == "generic_password_assignment" for f in findings)

    def test_basic_auth_url(self):
        # "example" is in the allowlist (placeholder), so use a non-placeholder domain.
        findings = scan_text("connect https://alice:hunter2@prod.internal/db")
        assert any(f.rule == "basic_auth_url" for f in findings)

    def test_no_findings_on_clean_text(self):
        findings = scan_text("# Just a normal comment\nx = 1\ny = 2\n")
        assert findings == []

    def test_match_is_masked(self):
        findings = scan_text(_AKIA_FIXTURE)
        assert findings
        assert "ABCDEFGHIJKLMNOP" not in findings[0].match
        assert "***" in findings[0].match


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------

class TestAllowlist:
    def test_example_placeholder_skipped(self):
        # The AWS docs example literal is universally allowlisted.
        findings = scan_text("AKIAIOSFODNN7EXAMPLE")
        assert all(f.rule != "aws_access_key_id" for f in findings)

    def test_redacted_skipped(self):
        findings = scan_text("ghp_REDACTEDREDACTEDREDACTEDREDACTEDaa")
        assert all(f.rule != "github_token" for f in findings)


# ---------------------------------------------------------------------------
# Entropy heuristic
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_shannon_entropy_bounds(self):
        assert shannon_entropy("") == 0.0
        assert shannon_entropy("aaaaa") == 0.0
        # uniform over 4 symbols => 2 bits/char
        assert abs(shannon_entropy("abcdabcdabcd") - 2.0) < 0.01

    def test_high_entropy_string_flagged(self):
        # Random-looking 40-char base64-ish, split across concat so GitHub
        # push protection doesn't mistake this fixture for a real AWS
        # secret access key.
        suspicious = "k9J2pQ7vR4mN8xLtV1yB" + "3zUoH6cDwGsAfEiXjK5R"
        findings = scan_text(f"token = '{suspicious}'")
        assert any(f.rule == "high_entropy_string" for f in findings)

    def test_short_string_not_flagged(self):
        findings = scan_text("short_token = 'abc123'")
        assert all(f.rule != "high_entropy_string" for f in findings)


# ---------------------------------------------------------------------------
# scan_diff
# ---------------------------------------------------------------------------

_DIFF_ADDS_SECRET = (
    "diff --git a/settings.py b/settings.py\n"
    "index 0000..1111\n"
    "--- a/settings.py\n"
    "+++ b/settings.py\n"
    "@@ -5,0 +5,2 @@\n"
    f"+AWS_KEY = '{_AKIA_FIXTURE}'\n"
    "+DEBUG = True\n"
)

_DIFF_REMOVES_SECRET = (
    "diff --git a/settings.py b/settings.py\n"
    "index 0000..1111\n"
    "--- a/settings.py\n"
    "+++ b/settings.py\n"
    "@@ -5,2 +5,0 @@\n"
    f"-AWS_KEY = '{_AKIA_FIXTURE}'\n"
    "-DEBUG = True\n"
)


class TestScanDiff:
    def test_added_secret_is_flagged(self):
        findings = scan_diff(_DIFF_ADDS_SECRET)
        aws = [f for f in findings if f.rule == "aws_access_key_id"]
        assert aws
        assert aws[0].file == "settings.py"

    def test_removed_secret_is_not_flagged(self):
        findings = scan_diff(_DIFF_REMOVES_SECRET)
        assert not findings


# ---------------------------------------------------------------------------
# scan_files
# ---------------------------------------------------------------------------

class TestScanFiles:
    def test_scans_multiple_files(self, tmp_path):
        (tmp_path / "a.py").write_text("x = 1\n")
        (tmp_path / "b.py").write_text("TOK = 'ghp_" + "a" * 40 + "'\n")
        findings = scan_files([str(tmp_path / "a.py"), str(tmp_path / "b.py")])
        files = {f.file for f in findings if f.rule == "github_token"}
        assert any(p.endswith("b.py") for p in files)

    def test_missing_file_silently_skipped(self, tmp_path):
        # Should not raise - returns findings for files that do exist only.
        findings = scan_files([str(tmp_path / "does_not_exist.py")])
        assert findings == []
