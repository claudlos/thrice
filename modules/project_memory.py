"""Project Memory — persistent per-project context for LLM system prompts.

Similar to Claude Code's CLAUDE.md or Cursor's .cursorrules.  Loads and merges
hierarchical markdown memory files from project directories, caches them with
mtime-based invalidation, and formats them for system prompt injection.

Memory files use markdown with ## sections:
  ## Conventions, ## Architecture, ## Commands, ## Testing, ## Notes

Searches multiple locations:
  - PROJECT_ROOT/.hermes/project.md  (primary)
  - PROJECT_ROOT/HERMES.md           (convenience alias)
  - SUBDIR/.hermes/project.md        (subdirectory overrides)

Uses only stdlib — no external dependencies.
"""

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT_MARKERS: Tuple[str, ...] = (
    ".git",
    "package.json",
    "pyproject.toml",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    "setup.py",
    "setup.cfg",
    ".hg",
)

MEMORY_FILENAME = "project.md"
HERMES_DIR = ".hermes"
HERMES_ROOT_FILE = "HERMES.md"

DEFAULT_SECTIONS: Tuple[str, ...] = (
    "Conventions",
    "Architecture",
    "Commands",
    "Testing",
    "Notes",
)

# ---------------------------------------------------------------------------
# Helpers — project detection
# ---------------------------------------------------------------------------


def find_project_root(start: str) -> Optional[str]:
    """Walk up from *start* looking for a project root marker.

    Returns the first directory that contains one of PROJECT_ROOT_MARKERS,
    or ``None`` if the filesystem root is reached without finding one.
    """
    current = os.path.realpath(start)
    while True:
        for marker in PROJECT_ROOT_MARKERS:
            candidate = os.path.join(current, marker)
            if os.path.exists(candidate):
                return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def detect_project_type(root: str) -> Dict[str, Optional[str]]:
    """Scan *root* and return a dict describing the project.

    Keys: language, test_framework, build_tool, package_manager.
    Values are best-effort strings or ``None``.
    """
    info: Dict[str, Optional[str]] = {
        "language": None,
        "test_framework": None,
        "build_tool": None,
        "package_manager": None,
    }

    root_path = Path(root)

    # --- Python ---
    if (root_path / "pyproject.toml").is_file() or (root_path / "setup.py").is_file():
        info["language"] = "python"
        info["package_manager"] = "pip"
        # Check for poetry / pdm / hatch
        pyproject = root_path / "pyproject.toml"
        if pyproject.is_file():
            try:
                text = pyproject.read_text(encoding="utf-8", errors="replace")
                if "tool.poetry" in text:
                    info["package_manager"] = "poetry"
                    info["build_tool"] = "poetry"
                elif "tool.pdm" in text:
                    info["package_manager"] = "pdm"
                    info["build_tool"] = "pdm"
                elif "tool.hatch" in text or "hatchling" in text:
                    info["build_tool"] = "hatch"
            except OSError:
                pass
        # Test frameworks
        if (root_path / "pytest.ini").is_file() or (root_path / "conftest.py").is_file():
            info["test_framework"] = "pytest"
        elif pyproject.is_file():
            try:
                text = pyproject.read_text(encoding="utf-8", errors="replace")
                if "pytest" in text:
                    info["test_framework"] = "pytest"
            except OSError:
                pass
        if info["test_framework"] is None:
            info["test_framework"] = "unittest"
        if info["build_tool"] is None:
            if (root_path / "Makefile").is_file():
                info["build_tool"] = "make"
            else:
                info["build_tool"] = "pip"

    # --- JavaScript / TypeScript ---
    elif (root_path / "package.json").is_file():
        info["language"] = "javascript"
        # Check for TS
        if (root_path / "tsconfig.json").is_file():
            info["language"] = "typescript"
        info["package_manager"] = "npm"
        pkg_file = root_path / "package.json"
        try:
            text = pkg_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            text = ""
        if (root_path / "yarn.lock").is_file():
            info["package_manager"] = "yarn"
        elif (root_path / "pnpm-lock.yaml").is_file():
            info["package_manager"] = "pnpm"
        elif (root_path / "bun.lockb").is_file():
            info["package_manager"] = "bun"
        # Test framework
        if "jest" in text:
            info["test_framework"] = "jest"
        elif "vitest" in text:
            info["test_framework"] = "vitest"
        elif "mocha" in text:
            info["test_framework"] = "mocha"
        # Build tool
        if "vite" in text:
            info["build_tool"] = "vite"
        elif "webpack" in text:
            info["build_tool"] = "webpack"
        elif "esbuild" in text:
            info["build_tool"] = "esbuild"
        else:
            info["build_tool"] = info["package_manager"]

    # --- Rust ---
    elif (root_path / "Cargo.toml").is_file():
        info["language"] = "rust"
        info["build_tool"] = "cargo"
        info["package_manager"] = "cargo"
        info["test_framework"] = "cargo test"

    # --- Go ---
    elif (root_path / "go.mod").is_file():
        info["language"] = "go"
        info["build_tool"] = "go"
        info["package_manager"] = "go modules"
        info["test_framework"] = "go test"

    return info


# ---------------------------------------------------------------------------
# Memory file parsing / serialization
# ---------------------------------------------------------------------------

_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def parse_memory_file(text: str) -> Dict[str, str]:
    """Parse markdown text into {section_title: body} dict.

    Sections are delimited by ``## Title`` headers.  Content before the first
    header is stored under the key ``"_preamble"``.
    """
    sections: Dict[str, str] = {}
    matches = list(_SECTION_RE.finditer(text))
    if not matches:
        stripped = text.strip()
        if stripped:
            sections["_preamble"] = stripped
        return sections

    # Content before first section
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections["_preamble"] = preamble

    for i, m in enumerate(matches):
        title = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections[title] = body

    return sections


def serialize_memory(sections: Dict[str, str]) -> str:
    """Convert a {section: body} dict back into markdown text."""
    parts: List[str] = []
    # Preamble first, if any
    if "_preamble" in sections:
        parts.append(sections["_preamble"])
        parts.append("")

    for title, body in sections.items():
        if title == "_preamble":
            continue
        parts.append(f"## {title}")
        parts.append("")
        if body:
            parts.append(body)
            parts.append("")

    return "\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------------
# ProjectMemory — single project's memory
# ---------------------------------------------------------------------------

@dataclass
class MemorySource:
    """Tracks a single memory file on disk."""
    path: str
    mtime: float = 0.0
    sections: Dict[str, str] = field(default_factory=dict)


_UNSET = object()


class ProjectMemory:
    """Loads, merges, and manages memory files for a single project.

    Parameters
    ----------
    project_root : str | None | _UNSET
        Explicit project root.  If not provided, auto-detected from *cwd*.
        Pass ``None`` explicitly to create an unbound instance.
    cwd : str | None
        Working directory for auto-detection.  Defaults to ``os.getcwd()``.
    """

    def __init__(
        self,
        project_root: Optional[str] = _UNSET,  # type: ignore[assignment]
        cwd: Optional[str] = None,
    ) -> None:
        if project_root is _UNSET:
            start = cwd or os.getcwd()
            project_root = find_project_root(start)
        self.project_root: Optional[str] = project_root
        self._sources: List[MemorySource] = []
        self._merged: Dict[str, str] = {}
        if self.project_root:
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_context(self, cwd: Optional[str] = None) -> str:
        """Return merged project memory formatted as markdown.

        If *cwd* is inside a subdirectory that has its own
        ``.hermes/project.md``, that content is merged on top of the
        root-level memory.
        """
        if not self.project_root:
            return ""

        merged = dict(self._merged)

        if cwd:
            subdir_sections = self._load_subdir_memory(cwd)
            for key, val in subdir_sections.items():
                if key in merged and val:
                    merged[key] = merged[key] + "\n\n" + val
                elif val:
                    merged[key] = val

        if not merged:
            return ""
        return serialize_memory(merged)

    def update(self, section: str, content: str) -> None:
        """Update (or create) a section in the primary memory file.

        Writes changes to ``PROJECT_ROOT/.hermes/project.md``.
        """
        if not self.project_root:
            raise RuntimeError("No project root detected — cannot update memory.")

        self._merged[section] = content

        # Write to primary file
        hermes_dir = os.path.join(self.project_root, HERMES_DIR)
        os.makedirs(hermes_dir, exist_ok=True)
        primary_path = os.path.join(hermes_dir, MEMORY_FILENAME)

        # Load existing file sections (to preserve sections we didn't change)
        existing: Dict[str, str] = {}
        if os.path.isfile(primary_path):
            try:
                text = Path(primary_path).read_text(encoding="utf-8", errors="replace")
                existing = parse_memory_file(text)
            except OSError:
                pass

        existing[section] = content
        Path(primary_path).write_text(serialize_memory(existing), encoding="utf-8")

        # Reload
        self._load()

    def list_sections(self) -> List[str]:
        """Return the titles of all sections in the merged memory."""
        return [k for k in self._merged if k != "_preamble"]

    @property
    def sections(self) -> Dict[str, str]:
        """Read-only view of merged sections."""
        return dict(self._merged)

    @property
    def is_loaded(self) -> bool:
        return bool(self._merged)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _memory_file_paths(self) -> List[str]:
        """Return candidate memory file paths at the project root level."""
        if not self.project_root:
            return []
        return [
            os.path.join(self.project_root, HERMES_DIR, MEMORY_FILENAME),
            os.path.join(self.project_root, HERMES_ROOT_FILE),
        ]

    def _load(self) -> None:
        """Load / reload memory from disk."""
        self._sources.clear()
        self._merged.clear()

        for path in self._memory_file_paths():
            if os.path.isfile(path):
                try:
                    text = Path(path).read_text(encoding="utf-8", errors="replace")
                    mtime = os.path.getmtime(path)
                    sections = parse_memory_file(text)
                    src = MemorySource(path=path, mtime=mtime, sections=sections)
                    self._sources.append(src)
                    # Merge: later files override / append
                    for key, val in sections.items():
                        if key in self._merged and val:
                            self._merged[key] = self._merged[key] + "\n\n" + val
                        else:
                            self._merged[key] = val
                except OSError as exc:
                    logger.warning("Failed to read memory file %s: %s", path, exc)

    def _load_subdir_memory(self, cwd: str) -> Dict[str, str]:
        """Load subdirectory-level memory if it exists and differs from root."""
        if not self.project_root:
            return {}

        real_cwd = os.path.realpath(cwd)
        real_root = os.path.realpath(self.project_root)

        if real_cwd == real_root:
            return {}

        # Walk from cwd up to (but not including) project root
        sections: Dict[str, str] = {}
        current = real_cwd
        while current != real_root and current.startswith(real_root):
            sub_mem = os.path.join(current, HERMES_DIR, MEMORY_FILENAME)
            if os.path.isfile(sub_mem):
                try:
                    text = Path(sub_mem).read_text(encoding="utf-8", errors="replace")
                    parsed = parse_memory_file(text)
                    # Deeper dirs override shallower — since we walk bottom-up,
                    # the first found wins.
                    for key, val in parsed.items():
                        if key not in sections:
                            sections[key] = val
                except OSError:
                    pass
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent

        return sections

    def _check_mtimes(self) -> bool:
        """Return True if any source file has changed since last load."""
        for src in self._sources:
            try:
                if os.path.getmtime(src.path) != src.mtime:
                    return True
            except OSError:
                return True
        return False


# ---------------------------------------------------------------------------
# ProjectMemoryManager — multi-project cache
# ---------------------------------------------------------------------------

class ProjectMemoryManager:
    """Caches ProjectMemory instances per project root with mtime invalidation.

    Thread-safe via a lock.
    """

    def __init__(self, cache_ttl: float = 30.0) -> None:
        self._lock = threading.Lock()
        self._cache: Dict[str, ProjectMemory] = {}
        self._cache_times: Dict[str, float] = {}
        self._ttl = cache_ttl

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_memory(self, cwd: Optional[str] = None) -> ProjectMemory:
        """Return a (possibly cached) ProjectMemory for the project at *cwd*."""
        cwd = cwd or os.getcwd()
        root = find_project_root(cwd)
        if root is None:
            return ProjectMemory(project_root=None)

        now = time.monotonic()

        with self._lock:
            cached = self._cache.get(root)
            cache_time = self._cache_times.get(root, 0.0)

            if cached is not None and (now - cache_time) < self._ttl:
                # Quick mtime check
                if not cached._check_mtimes():
                    return cached

            # Build / rebuild
            mem = ProjectMemory(project_root=root)
            self._cache[root] = mem
            self._cache_times[root] = now
            return mem

    def invalidate(self, root: Optional[str] = None) -> None:
        """Drop cache for *root*, or all caches if ``None``."""
        with self._lock:
            if root is None:
                self._cache.clear()
                self._cache_times.clear()
            else:
                self._cache.pop(root, None)
                self._cache_times.pop(root, None)

    def format_for_prompt(self, cwd: Optional[str] = None) -> str:
        """Return project memory formatted for system prompt injection.

        Wraps the markdown in delimiters for clarity in the prompt.
        """
        mem = self.get_memory(cwd)
        context = mem.get_context(cwd)
        if not context.strip():
            return ""

        lines = [
            "<project_memory>",
            context.rstrip(),
            "</project_memory>",
        ]
        return "\n".join(lines)

    def generate_initial_memory(self, project_root: str) -> str:
        """Scan *project_root* and generate initial memory markdown.

        Returns the markdown string (does NOT write to disk).
        """
        info = detect_project_type(project_root)
        root_path = Path(project_root)

        sections: Dict[str, str] = {}

        # -- Conventions --
        conv_lines: List[str] = []
        if info["language"]:
            conv_lines.append(f"- Language: {info['language']}")
        if info["package_manager"]:
            conv_lines.append(f"- Package manager: {info['package_manager']}")
        # Check for formatter/linter configs
        for name, tool in [
            (".prettierrc", "prettier"),
            (".prettierrc.json", "prettier"),
            (".eslintrc", "eslint"),
            (".eslintrc.json", "eslint"),
            ("ruff.toml", "ruff"),
            (".flake8", "flake8"),
            (".pylintrc", "pylint"),
            ("rustfmt.toml", "rustfmt"),
        ]:
            if (root_path / name).exists():
                conv_lines.append(f"- Formatter/linter: {tool}")
                break
        # Check pyproject for ruff
        pyproj = root_path / "pyproject.toml"
        if pyproj.is_file():
            try:
                text = pyproj.read_text(encoding="utf-8", errors="replace")
                if "tool.ruff" in text:
                    conv_lines.append("- Linter: ruff")
                if "tool.black" in text or "tool.blue" in text:
                    conv_lines.append("- Formatter: black")
            except OSError:
                pass
        sections["Conventions"] = "\n".join(conv_lines) if conv_lines else ""

        # -- Architecture --
        arch_lines: List[str] = []
        # Detect common directory structures
        for d in ["src", "lib", "app", "cmd", "pkg", "internal", "api", "core"]:
            if (root_path / d).is_dir():
                arch_lines.append(f"- `{d}/` directory present")
        sections["Architecture"] = "\n".join(arch_lines) if arch_lines else ""

        # -- Commands --
        cmd_lines: List[str] = []
        if info["build_tool"]:
            cmd_lines.append(f"- Build: `{info['build_tool']}`")
        if info["test_framework"]:
            if info["language"] == "python" and info["test_framework"] == "pytest":
                cmd_lines.append("- Test: `pytest`")
            elif info["language"] in ("javascript", "typescript"):
                pm = info["package_manager"] or "npm"
                cmd_lines.append(f"- Test: `{pm} test`")
            elif info["language"] == "rust":
                cmd_lines.append("- Test: `cargo test`")
            elif info["language"] == "go":
                cmd_lines.append("- Test: `go test ./...`")
            else:
                cmd_lines.append(f"- Test: {info['test_framework']}")
        sections["Commands"] = "\n".join(cmd_lines) if cmd_lines else ""

        # -- Testing --
        test_lines: List[str] = []
        if info["test_framework"]:
            test_lines.append(f"- Framework: {info['test_framework']}")
        for d in ["tests", "test", "__tests__", "spec"]:
            if (root_path / d).is_dir():
                test_lines.append(f"- Test directory: `{d}/`")
                break
        sections["Testing"] = "\n".join(test_lines) if test_lines else ""

        # -- Notes --
        sections["Notes"] = ""

        return serialize_memory(sections)

    def generate_and_save(self, project_root: str) -> str:
        """Generate initial memory and save to PROJECT_ROOT/.hermes/project.md.

        Returns the generated markdown.
        """
        content = self.generate_initial_memory(project_root)
        hermes_dir = os.path.join(project_root, HERMES_DIR)
        os.makedirs(hermes_dir, exist_ok=True)
        target = os.path.join(hermes_dir, MEMORY_FILENAME)
        Path(target).write_text(content, encoding="utf-8")
        self.invalidate(project_root)
        return content

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def cached_roots(self) -> List[str]:
        with self._lock:
            return list(self._cache.keys())
