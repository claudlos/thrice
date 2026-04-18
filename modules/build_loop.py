"""Build-fix loop for Hermes Agent (Thrice).

The counterpart to ``test_fix_loop.py`` and ``test_lint_loop.py``: wraps a
language-specific build/compile tool, parses its diagnostics into a
structured ``BuildError`` list, and returns a ranked set of problems for
the agent to address.

Supported toolchains (auto-detected from project files; override via
``BuildConfig.command``):

    - Rust             : ``cargo build --message-format=short``
    - Go               : ``go build ./...``
    - TypeScript       : ``tsc --noEmit --pretty false``
    - C / C++          : ``make -k`` (or ``gcc -fdiagnostics-format=json``)
    - Python (syntax)  : ``python -m compileall -q .``

Usage::

    from build_loop import BuildLoop, BuildConfig

    loop = BuildLoop(BuildConfig(project_dir="."))
    result = loop.run()
    if not result.ok:
        for err in result.errors[:10]:
            print(f"{err.file}:{err.line}:{err.column}: {err.message}")

Design notes:

- Parsers are **line-oriented regex** for simplicity and portability.
  The goal is "good enough for agent context", not IDE-grade.
- Every parser is independently testable via ``parse_<toolchain>()``.
- Output is deterministically ordered (file, line, column, code) so that
  ``change_impact`` and ``repo_map`` can reason about it stably.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple

logger = logging.getLogger(__name__)


Severity = Literal["error", "warning", "note"]
Toolchain = Literal["rust", "go", "typescript", "c", "python", "auto"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BuildError:
    """One diagnostic emitted by the build tool."""

    file: str
    line: int
    column: int
    severity: Severity
    message: str
    code: Optional[str] = None          # e.g. "E0308" (rustc) or "TS2322"
    hint: Optional[str] = None

    def format_short(self) -> str:
        c = f"[{self.code}] " if self.code else ""
        return f"{self.file}:{self.line}:{self.column}: {self.severity}: {c}{self.message}"


@dataclass
class BuildConfig:
    """Configuration for a build-loop run."""

    project_dir: str = "."
    toolchain: Toolchain = "auto"
    command: Optional[List[str]] = None   # overrides toolchain defaults
    timeout: float = 120.0
    max_errors: int = 50
    env: Optional[Dict[str, str]] = None  # extra env vars


@dataclass
class BuildResult:
    """Result of invoking the build."""

    toolchain: str
    command: List[str]
    ok: bool
    returncode: int
    errors: List[BuildError] = field(default_factory=list)
    warnings: List[BuildError] = field(default_factory=list)
    duration_s: float = 0.0
    stdout: str = ""
    stderr: str = ""
    error_summary: str = ""

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def first_errors(self, n: int = 5) -> List[BuildError]:
        """Return the first ``n`` errors in deterministic order."""
        return sorted(
            self.errors,
            key=lambda e: (e.file, e.line, e.column, e.code or ""),
        )[:n]


# ---------------------------------------------------------------------------
# Toolchain detection
# ---------------------------------------------------------------------------

def detect_toolchain(project_dir: str) -> Toolchain:
    """Auto-detect the toolchain from project files."""
    p = Path(project_dir)
    markers: List[Tuple[str, Toolchain]] = [
        ("Cargo.toml", "rust"),
        ("go.mod", "go"),
        ("tsconfig.json", "typescript"),
        ("CMakeLists.txt", "c"),
        ("Makefile", "c"),
        ("pyproject.toml", "python"),
        ("setup.py", "python"),
    ]
    for fname, tc in markers:
        if (p / fname).exists():
            return tc
    return "auto"


def default_command(toolchain: Toolchain) -> Optional[List[str]]:
    """Return the default build command for a toolchain."""
    return {
        "rust":       ["cargo", "build", "--message-format=short"],
        "go":         ["go", "build", "./..."],
        "typescript": ["npx", "--no", "tsc", "--noEmit", "--pretty", "false"],
        "c":          ["make", "-k"],
        "python":     ["python", "-m", "compileall", "-q", "."],
    }.get(toolchain)


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

# Each parser returns a list of BuildError; accepts the full stdout+stderr
# string produced by the tool.  Parsers must be pure and deterministic.

_RUST_RE = re.compile(
    r"""^
    (?P<severity>error|warning|note)
    (?:\[(?P<code>[A-Z]\d+)\])?
    :\s*(?P<message>.+?)\n
    \s*-->\s*(?P<file>[^:\n]+):(?P<line>\d+):(?P<col>\d+)
    """,
    re.VERBOSE | re.MULTILINE,
)


def parse_rust(output: str) -> List[BuildError]:
    """Parse `cargo build` / `rustc` short-format diagnostics."""
    out: List[BuildError] = []
    for m in _RUST_RE.finditer(output):
        out.append(BuildError(
            file=m.group("file"),
            line=int(m.group("line")),
            column=int(m.group("col")),
            severity=m.group("severity"),   # type: ignore[arg-type]
            message=m.group("message").strip(),
            code=m.group("code"),
        ))
    return out


# ``./main.go:10:5: cannot use x (type int) as type string``
_GO_RE = re.compile(
    r"^(?P<file>\.?\.?/?[^\s:][^:\n]*\.go):(?P<line>\d+):(?P<col>\d+):\s*(?P<message>.+)$",
    re.MULTILINE,
)


def parse_go(output: str) -> List[BuildError]:
    """Parse `go build` diagnostics."""
    out: List[BuildError] = []
    for m in _GO_RE.finditer(output):
        msg = m.group("message").strip()
        severity: Severity = "warning" if msg.lower().startswith("warning") else "error"
        out.append(BuildError(
            file=m.group("file"),
            line=int(m.group("line")),
            column=int(m.group("col")),
            severity=severity,
            message=msg,
        ))
    return out


# ``src/foo.ts(10,5): error TS2322: Type 'number' is not assignable to type 'string'.``
# ``src/foo.ts:10:5 - error TS2322: Type 'number' ...``  (pretty=false)
_TS_RE = re.compile(
    r"""^
    (?P<file>[^\s(][^(:\n]*?)
    (?:\((?P<line1>\d+),(?P<col1>\d+)\)|:(?P<line2>\d+):(?P<col2>\d+))
    \s*[:\-]\s*
    (?P<severity>error|warning)
    \s+(?P<code>TS\d+)
    :\s*(?P<message>.+?)$
    """,
    re.VERBOSE | re.MULTILINE,
)


def parse_typescript(output: str) -> List[BuildError]:
    """Parse `tsc --pretty false` diagnostics."""
    out: List[BuildError] = []
    for m in _TS_RE.finditer(output):
        line = int(m.group("line1") or m.group("line2"))
        col = int(m.group("col1") or m.group("col2"))
        out.append(BuildError(
            file=m.group("file"),
            line=line,
            column=col,
            severity=m.group("severity"),  # type: ignore[arg-type]
            message=m.group("message").strip(),
            code=m.group("code"),
        ))
    return out


# ``foo.c:10:5: error: expected ';' before ...``
_GCC_RE = re.compile(
    r"^(?P<file>[^\s:][^:\n]*):(?P<line>\d+):(?P<col>\d+):\s*"
    r"(?P<severity>error|warning|note):\s*(?P<message>.+)$",
    re.MULTILINE,
)


def parse_c(output: str) -> List[BuildError]:
    """Parse GCC/Clang diagnostics."""
    out: List[BuildError] = []
    for m in _GCC_RE.finditer(output):
        out.append(BuildError(
            file=m.group("file"),
            line=int(m.group("line")),
            column=int(m.group("col")),
            severity=m.group("severity"),  # type: ignore[arg-type]
            message=m.group("message").strip(),
        ))
    return out


# ``  File "foo.py", line 10\n    if x\n        ^\nSyntaxError: invalid syntax``
_PY_RE = re.compile(
    r'File "(?P<file>[^"]+)", line (?P<line>\d+)(?:.*?\n)+?'
    r"(?P<err>[A-Z][A-Za-z]+Error|SyntaxError|IndentationError): (?P<message>.+)",
    re.MULTILINE,
)


def parse_python(output: str) -> List[BuildError]:
    """Parse ``compileall``/``py_compile`` output for syntax errors."""
    out: List[BuildError] = []
    for m in _PY_RE.finditer(output):
        out.append(BuildError(
            file=m.group("file"),
            line=int(m.group("line")),
            column=0,
            severity="error",
            message=m.group("message").strip(),
            code=m.group("err"),
        ))
    return out


_PARSERS: Dict[str, Callable[[str], List[BuildError]]] = {
    "rust":       parse_rust,
    "go":         parse_go,
    "typescript": parse_typescript,
    "c":          parse_c,
    "python":     parse_python,
}


# ---------------------------------------------------------------------------
# Build loop
# ---------------------------------------------------------------------------

class BuildLoop:
    """Run a build and return structured errors for the agent."""

    def __init__(self, config: Optional[BuildConfig] = None):
        self.config = config or BuildConfig()

    # -- Public API --------------------------------------------------------

    def run(self) -> BuildResult:
        """Invoke the build and return a structured result."""
        tc = self._resolve_toolchain()
        cmd = self._resolve_command(tc)
        t0 = time.monotonic()
        proc = self._invoke(cmd)
        elapsed = time.monotonic() - t0
        stdout, stderr = proc.stdout or "", proc.stderr or ""
        combined = stdout + "\n" + stderr

        parser = _PARSERS.get(tc)
        diagnostics: List[BuildError] = parser(combined) if parser else []
        # Apply cap on errors to keep the context budget sane.
        errors = [d for d in diagnostics if d.severity == "error"][: self.config.max_errors]
        warnings = [d for d in diagnostics if d.severity == "warning"][: self.config.max_errors]

        summary = self._summary(errors, warnings, proc.returncode, tc)
        return BuildResult(
            toolchain=tc,
            command=cmd,
            ok=(proc.returncode == 0 and not errors),
            returncode=proc.returncode,
            errors=errors,
            warnings=warnings,
            duration_s=elapsed,
            stdout=stdout,
            stderr=stderr,
            error_summary=summary,
        )

    # -- Helpers ----------------------------------------------------------

    def _resolve_toolchain(self) -> str:
        tc = self.config.toolchain
        if tc == "auto":
            tc = detect_toolchain(self.config.project_dir)
            if tc == "auto":
                raise ValueError(
                    "Could not auto-detect toolchain.  Pass BuildConfig(toolchain=...) "
                    "or provide an explicit command."
                )
        return tc

    def _resolve_command(self, tc: str) -> List[str]:
        if self.config.command:
            return list(self.config.command)
        cmd = default_command(tc)  # type: ignore[arg-type]
        if cmd is None:
            raise ValueError(f"No default command for toolchain: {tc}")
        if shutil.which(cmd[0]) is None:
            raise FileNotFoundError(
                f"Toolchain binary not found on PATH: {cmd[0]}"
            )
        return cmd

    def _invoke(self, cmd: List[str]) -> subprocess.CompletedProcess:
        env = None
        if self.config.env:
            env = {**os.environ, **self.config.env}
        try:
            return subprocess.run(
                cmd,
                cwd=self.config.project_dir,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=124,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + f"\n[build_loop] timed out after {self.config.timeout:.0f}s",
            )

    def _summary(
        self,
        errors: List[BuildError],
        warnings: List[BuildError],
        rc: int,
        tc: str,
    ) -> str:
        if rc == 0 and not errors:
            return f"build OK ({tc}); {len(warnings)} warnings"
        lines = [f"build FAILED ({tc}): {len(errors)} errors, {len(warnings)} warnings, rc={rc}"]
        for e in sorted(errors, key=lambda x: (x.file, x.line))[:5]:
            lines.append("  " + e.format_short())
        if len(errors) > 5:
            lines.append(f"  ... and {len(errors) - 5} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def build(project_dir: str = ".", toolchain: Toolchain = "auto", **kwargs) -> BuildResult:
    """One-shot convenience wrapper."""
    cfg = BuildConfig(project_dir=project_dir, toolchain=toolchain, **kwargs)
    return BuildLoop(cfg).run()
