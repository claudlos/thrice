"""
Smart Context Gathering — systematically gather relevant context before code changes.

Reads a target file, discovers its imports, reverse dependencies, test files,
and related files, then prioritizes and formats the gathered context for
inclusion in LLM prompts.

Uses only stdlib: ast, re, os, pathlib. No external dependencies.
"""

import os
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Language extensions
# ---------------------------------------------------------------------------

LANG_EXTENSIONS: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".rs": "rust",
}

# Directories to always skip
SKIP_DIRS: Set[str] = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", "venv", ".venv", "env",
    ".env", "dist", "build", ".tox", ".eggs", "egg-info",
    "target", "vendor",
}

# Approximate chars per token for estimation
CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ContextRequest:
    """A request to gather context for a code change."""
    target_file: str
    change_description: str
    project_root: str


@dataclass
class GatheredContext:
    """The result of context gathering — everything the LLM needs."""
    target_file_content: str = ""
    imports: List[str] = field(default_factory=list)
    imported_by: List[str] = field(default_factory=list)
    test_files: List[str] = field(default_factory=list)
    related_files: List[str] = field(default_factory=list)
    dependency_chain: List[str] = field(default_factory=list)
    summary: str = ""
    token_estimate: int = 0


# ---------------------------------------------------------------------------
# ImportExtractor — language-aware import extraction
# ---------------------------------------------------------------------------

# Python import patterns
_PY_IMPORT_RE = re.compile(
    r"^\s*import\s+([\w.]+)"
    r"|"
    r"^\s*from\s+([\w.]+)\s+import",
    re.MULTILINE,
)

# JS/TS import patterns
_JS_IMPORT_RE = re.compile(
    r"""import\s+(?:[\w{},\s*]+\s+from\s+)?['"]([^'"]+)['"]"""
    r"|"
    r"""require\s*\(\s*['"]([^'"]+)['"]\s*\)""",
    re.MULTILINE,
)

# Rust use patterns
_RUST_USE_RE = re.compile(
    r"^\s*use\s+([\w:]+(?:::\{[^}]+\})?)\s*;",
    re.MULTILINE,
)


class ImportExtractor:
    """Extracts import statements from source files across languages."""

    @staticmethod
    def extract_python_imports(content: str) -> List[str]:
        """Extract Python import module paths from source content."""
        imports = []
        for m in _PY_IMPORT_RE.finditer(content):
            module = m.group(1) or m.group(2)
            if module:
                imports.append(module)
        return imports

    @staticmethod
    def extract_js_imports(content: str) -> List[str]:
        """Extract JS/TS import paths from source content."""
        imports = []
        for m in _JS_IMPORT_RE.finditer(content):
            path = m.group(1) or m.group(2)
            if path:
                imports.append(path)
        return imports

    @staticmethod
    def extract_rust_imports(content: str) -> List[str]:
        """Extract Rust use paths from source content."""
        imports = []
        for m in _RUST_USE_RE.finditer(content):
            use_path = m.group(1)
            if use_path:
                imports.append(use_path)
        return imports

    @staticmethod
    def auto_detect_language(file_path: str) -> str:
        """Detect programming language from file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        return LANG_EXTENSIONS.get(ext, "unknown")

    def extract_imports(self, content: str, file_path: str) -> List[str]:
        """Auto-detect language and extract imports."""
        lang = self.auto_detect_language(file_path)
        if lang == "python":
            return self.extract_python_imports(content)
        elif lang in ("javascript", "typescript"):
            return self.extract_js_imports(content)
        elif lang == "rust":
            return self.extract_rust_imports(content)
        return []

    @staticmethod
    def resolve_import_to_file(import_path: str, project_root: str) -> Optional[str]:
        """Resolve a Python module import path to a file on disk.

        Tries: module/as/path.py, module/as/path/__init__.py
        Returns the resolved path relative to project_root, or None.
        """
        # Convert dotted module path to filesystem path
        parts = import_path.replace(".", os.sep)

        # Try as a .py file
        candidate = os.path.join(project_root, parts + ".py")
        if os.path.isfile(candidate):
            return os.path.relpath(candidate, project_root)

        # Try as a package (__init__.py)
        candidate = os.path.join(project_root, parts, "__init__.py")
        if os.path.isfile(candidate):
            return os.path.relpath(candidate, project_root)

        # Try JS/TS extensions
        for ext in (".js", ".ts", ".jsx", ".tsx"):
            candidate = os.path.join(project_root, parts + ext)
            if os.path.isfile(candidate):
                return os.path.relpath(candidate, project_root)

            # Try with index file
            candidate = os.path.join(project_root, parts, "index" + ext)
            if os.path.isfile(candidate):
                return os.path.relpath(candidate, project_root)

        return None


# ---------------------------------------------------------------------------
# TestFileFinder — find test files for a given source file
# ---------------------------------------------------------------------------

class TestFileFinder:
    """Finds test files associated with a source file."""
    __test__ = False


    # Patterns to try: (directory_pattern, filename_pattern)
    # {stem} is replaced with the source file's stem (name without extension)
    PYTHON_PATTERNS = [
        (".", "test_{stem}.py"),
        (".", "{stem}_test.py"),
        ("tests", "test_{stem}.py"),
        ("tests", "{stem}_test.py"),
        ("test", "test_{stem}.py"),
        ("test", "{stem}_test.py"),
    ]

    JS_PATTERNS = [
        ("__tests__", "{stem}.test.js"),
        ("__tests__", "{stem}.test.ts"),
        ("__tests__", "{stem}.test.jsx"),
        ("__tests__", "{stem}.test.tsx"),
        (".", "{stem}.test.js"),
        (".", "{stem}.test.ts"),
        (".", "{stem}.spec.js"),
        (".", "{stem}.spec.ts"),
        (".", "{stem}.spec.jsx"),
        (".", "{stem}.spec.tsx"),
    ]

    RUST_PATTERNS = [
        # Rust tests are typically inline, but sometimes in tests/ dir
        ("tests", "{stem}.rs"),
        ("tests", "test_{stem}.rs"),
    ]

    @classmethod
    def find_test_files(cls, source_file: str, project_root: str) -> List[str]:
        """Find test files for the given source file.

        Searches in the source file's directory and common test directories,
        both relative to the file and relative to the project root.
        Returns paths relative to project_root.
        """
        source_path = Path(source_file)
        if source_path.is_absolute():
            try:
                source_path = source_path.relative_to(project_root)
            except ValueError:
                return []

        stem = source_path.stem
        # Remove test_ prefix or _test suffix if the file is already a test
        if stem.startswith("test_"):
            stem = stem[5:]
        elif stem.endswith("_test"):
            stem = stem[:-5]

        lang = ImportExtractor.auto_detect_language(str(source_path))
        if lang == "python":
            patterns = cls.PYTHON_PATTERNS
        elif lang in ("javascript", "typescript"):
            patterns = cls.JS_PATTERNS
        elif lang == "rust":
            patterns = cls.RUST_PATTERNS
        else:
            patterns = cls.PYTHON_PATTERNS  # default fallback

        source_dir = source_path.parent
        found = []

        for dir_pattern, file_pattern in patterns:
            filename = file_pattern.format(stem=stem)

            # Search relative to source file's directory
            candidate = source_dir / dir_pattern / filename
            full = Path(project_root) / candidate
            if full.is_file():
                found.append(str(candidate))

            # Search relative to project root
            if dir_pattern != ".":
                candidate = Path(dir_pattern) / filename
                full = Path(project_root) / candidate
                if full.is_file() and str(candidate) not in found:
                    found.append(str(candidate))

        return found


# ---------------------------------------------------------------------------
# ContextPrioritizer — trim context to fit token budget
# ---------------------------------------------------------------------------

class ContextPrioritizer:
    """Prioritizes gathered context to fit within a token budget.

    Priority order (highest to lowest):
        1. target file content
        2. test files
        3. direct imports
        4. reverse dependencies (imported_by)
        5. related files
    """

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count from text length."""
        return max(1, len(text) // CHARS_PER_TOKEN)

    @classmethod
    def prioritize(cls, gathered: GatheredContext, max_tokens: int) -> GatheredContext:
        """Return a new GatheredContext trimmed to fit within max_tokens.

        Drops lowest-priority file lists first (related > imported_by > imports > tests).
        The target file content is never dropped.
        """
        result = GatheredContext(
            target_file_content=gathered.target_file_content,
            imports=list(gathered.imports),
            imported_by=list(gathered.imported_by),
            test_files=list(gathered.test_files),
            related_files=list(gathered.related_files),
            dependency_chain=list(gathered.dependency_chain),
            summary=gathered.summary,
            token_estimate=gathered.token_estimate,
        )

        current_tokens = cls._estimate_tokens(result.target_file_content)

        # Add token estimates for each list: approximate as path length
        list_fields = [
            ("test_files", result.test_files),
            ("imports", result.imports),
            ("imported_by", result.imported_by),
            ("related_files", result.related_files),
        ]

        # Estimate tokens for each category
        category_tokens: Dict[str, int] = {}
        for name, items in list_fields:
            tokens = sum(cls._estimate_tokens(p) for p in items) if items else 0
            category_tokens[name] = tokens
            current_tokens += tokens

        # If within budget, just update estimate and return
        if current_tokens <= max_tokens:
            result.token_estimate = current_tokens
            return result

        # Drop from lowest priority first
        drop_order = ["related_files", "imported_by", "imports", "test_files"]

        for field_name in drop_order:
            if current_tokens <= max_tokens:
                break
            field_list = getattr(result, field_name)
            tokens = category_tokens.get(field_name, 0)
            # Remove items from the end until we fit
            while field_list and current_tokens > max_tokens:
                removed = field_list.pop()
                removed_tokens = cls._estimate_tokens(removed)
                current_tokens -= removed_tokens

        result.token_estimate = current_tokens
        return result


# ---------------------------------------------------------------------------
# ContextGatherer — the main orchestrator
# ---------------------------------------------------------------------------

class ContextGatherer:
    """Gathers all relevant context for a code change."""

    def __init__(self):
        self.extractor = ImportExtractor()
        self.test_finder = TestFileFinder()
        self.prioritizer = ContextPrioritizer()

    def gather(self, request: ContextRequest) -> GatheredContext:
        """Run the full context discovery pipeline.

        Steps:
            1. Read the target file
            2. Extract imports from the target
            3. Find files that import the target (reverse deps)
            4. Find associated test files
            5. Find related files by name similarity and proximity
            6. Build a dependency chain
            7. Estimate tokens
        """
        ctx = GatheredContext()
        project_root = os.path.abspath(request.project_root)

        # Resolve the target file path
        target = request.target_file
        if not os.path.isabs(target):
            target = os.path.join(project_root, target)

        # Step 1: Read the target file
        ctx.target_file_content = self._read_file(target)

        # Step 2: Extract imports
        raw_imports = self.extractor.extract_imports(ctx.target_file_content, target)
        resolved_imports = []
        for imp in raw_imports:
            resolved = self.extractor.resolve_import_to_file(imp, project_root)
            if resolved:
                resolved_imports.append(resolved)
            else:
                resolved_imports.append(imp)
        ctx.imports = resolved_imports

        # Step 3: Find reverse dependencies
        target_rel = os.path.relpath(target, project_root)
        ctx.imported_by = self._find_reverse_deps(target_rel, project_root)

        # Step 4: Find test files
        ctx.test_files = self.test_finder.find_test_files(target, project_root)

        # Step 5: Find related files
        ctx.related_files = self._find_related_files(target_rel, project_root)

        # Step 6: Build dependency chain
        ctx.dependency_chain = self._build_dependency_chain(
            target_rel, project_root, max_depth=3
        )

        # Step 7: Build summary and estimate tokens
        ctx.summary = self._build_summary(request, ctx)
        ctx.token_estimate = self._estimate_total_tokens(ctx)

        return ctx

    def gather_for_edit(self, file_path: str, edit_description: str) -> GatheredContext:
        """Convenience wrapper: infer project root and gather context."""
        # Walk up to find project root (look for .git, pyproject.toml, package.json, Cargo.toml)
        project_root = self._find_project_root(file_path)
        request = ContextRequest(
            target_file=file_path,
            change_description=edit_description,
            project_root=project_root,
        )
        return self.gather(request)

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _read_file(path: str) -> str:
        """Read a file, returning empty string on failure."""
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except (OSError, IOError):
            return ""

    @staticmethod
    def _find_project_root(file_path: str) -> str:
        """Walk up from file_path to find a project root directory."""
        markers = {".git", "pyproject.toml", "setup.py", "package.json", "Cargo.toml"}
        current = Path(file_path).resolve().parent
        for _ in range(20):  # max depth
            for marker in markers:
                if (current / marker).exists():
                    return str(current)
            parent = current.parent
            if parent == current:
                break
            current = parent
        # Fallback: directory containing the file
        return str(Path(file_path).resolve().parent)

    def _iter_source_files(self, project_root: str):
        """Iterate over source files in the project."""
        for dirpath, dirnames, filenames in os.walk(project_root):
            dirnames[:] = [
                d for d in dirnames
                if d not in SKIP_DIRS and not d.startswith(".")
            ]
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in LANG_EXTENSIONS:
                    yield os.path.join(dirpath, fname)

    def _find_reverse_deps(self, target_rel: str, project_root: str) -> List[str]:
        """Find files that import the target file."""
        target_stem = Path(target_rel).stem
        target_module = target_rel.replace(os.sep, ".").replace("/", ".")
        if target_module.endswith(".py"):
            target_module = target_module[:-3]

        reverse_deps = []

        for fpath in self._iter_source_files(project_root):
            rel = os.path.relpath(fpath, project_root)
            if rel == target_rel:
                continue

            try:
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except (OSError, IOError):
                continue

            imports = self.extractor.extract_imports(content, fpath)

            for imp in imports:
                # Check if the import refers to our target
                if (imp == target_module
                    or imp == target_stem
                    or imp.endswith("." + target_stem)
                    or imp.split(".")[-1] == target_stem):
                    reverse_deps.append(rel)
                    break

        return reverse_deps

    def _find_related_files(
        self, target_rel: str, project_root: str, max_results: int = 5
    ) -> List[str]:
        """Find related files by name similarity and directory proximity."""
        target_path = Path(target_rel)
        target_stem = target_path.stem.lower()
        target_dir = target_path.parent

        candidates: List[Tuple[float, str]] = []

        for fpath in self._iter_source_files(project_root):
            rel = os.path.relpath(fpath, project_root)
            if rel == target_rel:
                continue

            rel_path = Path(rel)
            stem = rel_path.stem.lower()

            # Name similarity score (0.0 to 1.0)
            name_score = SequenceMatcher(None, target_stem, stem).ratio()

            # Directory proximity bonus
            try:
                common = Path(os.path.commonpath([str(target_dir), str(rel_path.parent)]))
                depth_diff = (
                    len(target_dir.parts) - len(common.parts)
                    + len(rel_path.parent.parts) - len(common.parts)
                )
            except ValueError:
                depth_diff = 10

            proximity_score = max(0.0, 1.0 - depth_diff * 0.2)

            combined = name_score * 0.6 + proximity_score * 0.4
            if combined > 0.3:  # threshold
                candidates.append((combined, rel))

        # Sort by score descending, return top results
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [c[1] for c in candidates[:max_results]]

    def _build_dependency_chain(
        self, target_rel: str, project_root: str, max_depth: int = 3
    ) -> List[str]:
        """Build a transitive dependency chain from the target file's imports."""
        visited: Set[str] = set()
        chain: List[str] = []

        def _walk(rel_path: str, depth: int):
            if depth > max_depth or rel_path in visited:
                return
            visited.add(rel_path)

            full_path = os.path.join(project_root, rel_path)
            content = self._read_file(full_path)
            if not content:
                return

            imports = self.extractor.extract_imports(content, full_path)
            for imp in imports:
                resolved = self.extractor.resolve_import_to_file(imp, project_root)
                if resolved and resolved not in visited:
                    chain.append(resolved)
                    _walk(resolved, depth + 1)

        _walk(target_rel, 0)
        return chain

    @staticmethod
    def _build_summary(request: ContextRequest, ctx: GatheredContext) -> str:
        """Build a human-readable summary of gathered context."""
        parts = [f"Context for: {request.target_file}"]
        parts.append(f"Change: {request.change_description}")
        parts.append(f"Target file: {len(ctx.target_file_content)} chars")

        if ctx.imports:
            parts.append(f"Imports: {len(ctx.imports)} modules")
        if ctx.imported_by:
            parts.append(f"Imported by: {len(ctx.imported_by)} files")
        if ctx.test_files:
            parts.append(f"Test files: {len(ctx.test_files)}")
        if ctx.related_files:
            parts.append(f"Related files: {len(ctx.related_files)}")
        if ctx.dependency_chain:
            parts.append(f"Dependency chain depth: {len(ctx.dependency_chain)}")

        return " | ".join(parts)

    @staticmethod
    def _estimate_total_tokens(ctx: GatheredContext) -> int:
        """Estimate total tokens for all gathered context."""
        total_chars = len(ctx.target_file_content)
        for lst in [ctx.imports, ctx.imported_by, ctx.test_files,
                     ctx.related_files, ctx.dependency_chain]:
            total_chars += sum(len(item) for item in lst)
        total_chars += len(ctx.summary)
        return max(1, total_chars // CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# format_for_prompt — format context for LLM consumption
# ---------------------------------------------------------------------------

def format_for_prompt(gathered: GatheredContext) -> str:
    """Format gathered context as a structured block for an LLM prompt.

    Returns a clear, sectioned text block that can be prepended to
    a code-change prompt.
    """
    sections = []

    sections.append("=" * 60)
    sections.append("GATHERED CONTEXT")
    sections.append("=" * 60)

    if gathered.summary:
        sections.append("")
        sections.append(f"Summary: {gathered.summary}")

    sections.append("")
    sections.append("-" * 40)
    sections.append("TARGET FILE")
    sections.append("-" * 40)
    if gathered.target_file_content:
        sections.append(gathered.target_file_content)
    else:
        sections.append("(file not found or empty)")

    if gathered.imports:
        sections.append("")
        sections.append("-" * 40)
        sections.append("IMPORTS (direct dependencies)")
        sections.append("-" * 40)
        for imp in gathered.imports:
            sections.append(f"  - {imp}")

    if gathered.imported_by:
        sections.append("")
        sections.append("-" * 40)
        sections.append("IMPORTED BY (reverse dependencies)")
        sections.append("-" * 40)
        for dep in gathered.imported_by:
            sections.append(f"  - {dep}")

    if gathered.test_files:
        sections.append("")
        sections.append("-" * 40)
        sections.append("TEST FILES")
        sections.append("-" * 40)
        for tf in gathered.test_files:
            sections.append(f"  - {tf}")

    if gathered.related_files:
        sections.append("")
        sections.append("-" * 40)
        sections.append("RELATED FILES")
        sections.append("-" * 40)
        for rf in gathered.related_files:
            sections.append(f"  - {rf}")

    if gathered.dependency_chain:
        sections.append("")
        sections.append("-" * 40)
        sections.append("DEPENDENCY CHAIN")
        sections.append("-" * 40)
        for dc in gathered.dependency_chain:
            sections.append(f"  - {dc}")

    sections.append("")
    sections.append(f"Estimated tokens: {gathered.token_estimate}")
    sections.append("=" * 60)

    return "\n".join(sections)
