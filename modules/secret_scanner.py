"""Pre-commit secret scanner for Hermes Agent (Thrice).

Detects likely secrets (API keys, credentials, private keys, JWTs, high-
entropy strings) in a diff or file tree, so ``auto_commit`` can refuse to
ship them.

Design:

- **Pattern rules** (regex-based) cover the best-known secret formats
  with high precision.
- **Entropy heuristic** flags strings of 20+ chars with Shannon entropy
  above a threshold AND not already in an allowlist of benign shapes.
- **Allowlist** skips obvious-test-data patterns
  (``example``, ``placeholder``, sha-looking paths, etc.).

Usage::

    from secret_scanner import scan_diff, scan_text, Finding

    findings = scan_diff(subprocess.check_output(
        ["git", "diff", "--cached"], text=True
    ))
    if findings:
        for f in findings:
            print(f.format_short())
        sys.exit(1)

The entropy threshold and rule set are deliberately conservative; false
positives are annoying but false negatives (shipping a real key) are
expensive.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Pattern, Sequence

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Finding:
    """One flagged secret."""

    rule: str              # e.g. "aws_access_key_id"
    severity: str          # "high" | "medium" | "low"
    file: Optional[str]    # None when scanning raw text
    line: Optional[int]    # 1-based; None when unknown
    match: str             # the masked / truncated matched string
    reason: str            # human-readable description

    def format_short(self) -> str:
        loc = f"{self.file}:{self.line}" if self.file else "<input>"
        return f"{loc}  [{self.severity}] {self.rule}: {self.reason} ({self.match})"


# ---------------------------------------------------------------------------
# Pattern rules
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    name: str
    regex: Pattern[str]
    severity: str
    reason: str


def _r(pattern: str) -> Pattern[str]:
    return re.compile(pattern)


_RULES: Sequence[Rule] = (
    Rule("aws_access_key_id",
         _r(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
         "high",
         "AWS access key id"),
    Rule("aws_secret_access_key",
         _r(r"(?i)aws(?:.{0,20})?(?:secret|sk)(?:.{0,5})?[=:]\s*"
            r"['\"]?([A-Za-z0-9/+=]{40})['\"]?"),
         "high",
         "AWS secret access key"),
    Rule("github_token",
         _r(r"\b(?:gh[pousr])_[A-Za-z0-9]{36,255}\b"),
         "high",
         "GitHub personal / OAuth / server / user / refresh token"),
    Rule("github_app_token",
         _r(r"\b(?:ghs|ghu)_[A-Za-z0-9]{36,255}\b"),
         "high",
         "GitHub App token"),
    Rule("slack_token",
         _r(r"\bxox[abpr]-[A-Za-z0-9-]{10,}\b"),
         "high",
         "Slack API token"),
    Rule("stripe_live_key",
         _r(r"\bsk_live_[0-9a-zA-Z]{24,}\b"),
         "high",
         "Stripe secret (live) key"),
    Rule("stripe_test_key",
         _r(r"\bsk_test_[0-9a-zA-Z]{24,}\b"),
         "medium",
         "Stripe secret (test) key"),
    Rule("google_api_key",
         _r(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
         "high",
         "Google API key"),
    Rule("openai_key",
         _r(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
         "high",
         "OpenAI API key"),
    Rule("anthropic_key",
         _r(r"\bsk-ant-(?:api|oat)\d{2}-[A-Za-z0-9_\-]{80,}\b"),
         "high",
         "Anthropic API key"),
    Rule("private_key_block",
         _r(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY-----"),
         "high",
         "Private key PEM block"),
    Rule("jwt",
         _r(r"\beyJ[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]{10,}\b"),
         "medium",
         "JSON Web Token"),
    Rule("generic_password_assignment",
         _r(r"(?i)(?:password|passwd|pwd)\s*[=:]\s*['\"]"
            r"(?P<value>[^'\"\n]{6,})['\"]"),
         "medium",
         "Hard-coded password assignment"),
    Rule("basic_auth_url",
         _r(r"https?://[^\s:/]+:[^@\s]+@[^\s]+"),
         "medium",
         "HTTP URL with embedded credentials"),
)


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------

# Any substring match against the full matched string skips the finding.
# These are common test-fixture values or obvious placeholders.
_ALLOWLIST_SUBSTRINGS = (
    "example", "EXAMPLE", "placeholder", "REDACTED", "xxxxxxxx",
    "AKIAIOSFODNN7EXAMPLE", "dummy", "fake-", "test-token-",
)


def _is_allowlisted(value: str) -> bool:
    v = value.lower()
    return any(s.lower() in v for s in _ALLOWLIST_SUBSTRINGS)


# ---------------------------------------------------------------------------
# Entropy heuristic
# ---------------------------------------------------------------------------

_MIN_ENTROPY_LEN = 24
_ENTROPY_THRESHOLD = 4.3   # above ~4.3 bits/char is very uncommon for prose

# Candidate strings: quoted or assigned values that look base64-ish / hex-ish.
_ENTROPY_CANDIDATE = re.compile(
    r"(?P<value>[A-Za-z0-9+/=_\-]{%d,})" % _MIN_ENTROPY_LEN
)


def shannon_entropy(data: str) -> float:
    """Shannon entropy (bits/char) of ``data`` using its own alphabet."""
    if not data:
        return 0.0
    freq = {c: data.count(c) / len(data) for c in set(data)}
    return -sum(p * math.log2(p) for p in freq.values())


def _entropy_findings(text: str, file: Optional[str]) -> List[Finding]:
    out: List[Finding] = []
    for m in _ENTROPY_CANDIDATE.finditer(text):
        value = m.group("value")
        if _is_allowlisted(value):
            continue
        if shannon_entropy(value) < _ENTROPY_THRESHOLD:
            continue
        # Ignore pure-hex strings that look like git/file hashes - unless
        # they're very long.
        if re.fullmatch(r"[0-9a-f]+", value) and len(value) < 40:
            continue
        line = text.count("\n", 0, m.start()) + 1
        out.append(Finding(
            rule="high_entropy_string",
            severity="low",
            file=file,
            line=line,
            match=_mask(value),
            reason=f"high-entropy string (len={len(value)}, "
                   f"entropy={shannon_entropy(value):.2f})",
        ))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_text(text: str, file: Optional[str] = None) -> List[Finding]:
    """Scan arbitrary text for secrets.  Returns all findings in input order."""
    findings: List[Finding] = []
    for rule in _RULES:
        for m in rule.regex.finditer(text):
            matched = m.group(0)
            if _is_allowlisted(matched):
                continue
            line = text.count("\n", 0, m.start()) + 1
            findings.append(Finding(
                rule=rule.name,
                severity=rule.severity,
                file=file,
                line=line,
                match=_mask(matched),
                reason=rule.reason,
            ))
    findings.extend(_entropy_findings(text, file))
    return sorted(findings, key=_finding_sort_key)


def scan_diff(diff: str) -> List[Finding]:
    """Scan a unified diff.  Only added lines (``+``) are considered so we
    don't flag secrets that are being removed."""
    findings: List[Finding] = []
    current_file: Optional[str] = None
    new_line = 0
    in_hunk = False
    for raw_line in diff.splitlines():
        if raw_line.startswith("+++ b/"):
            current_file = raw_line[len("+++ b/"):].strip()
            in_hunk = False
            continue
        m = re.match(r"^@@ [^+]+ \+(\d+)", raw_line)
        if m:
            new_line = int(m.group(1))
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            added = raw_line[1:]
            for f in scan_text(added, current_file):
                findings.append(Finding(
                    rule=f.rule,
                    severity=f.severity,
                    file=f.file,
                    line=new_line,
                    match=f.match,
                    reason=f.reason,
                ))
            new_line += 1
        elif not raw_line.startswith("-"):
            new_line += 1
    return sorted(findings, key=_finding_sort_key)


def scan_files(paths: Iterable[str]) -> List[Finding]:
    """Scan a list of files on disk.  Non-UTF-8 files are read with ``errors='replace'``."""
    findings: List[Finding] = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            continue
        findings.extend(scan_text(text, file=path))
    return sorted(findings, key=_finding_sort_key)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mask(value: str) -> str:
    """Return a masked preview so the scanner output never leaks the secret."""
    if len(value) <= 8:
        return value[:2] + "***"
    return value[:4] + "***" + value[-2:]


_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def _finding_sort_key(f: Finding):
    return (
        _SEVERITY_ORDER.get(f.severity, 99),
        f.file or "",
        f.line or 0,
        f.rule,
    )


__all__ = [
    "Finding",
    "scan_text",
    "scan_diff",
    "scan_files",
    "shannon_entropy",
]
