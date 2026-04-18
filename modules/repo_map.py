"""
Repository Map — AST/regex-based codebase structure extraction with PageRank relevance.

Scans a project directory, extracts symbols (functions, classes, methods),
builds a dependency graph, ranks by simplified PageRank, and produces a
compact text map suitable for inclusion in LLM system prompts.

Uses only stdlib: ast for Python, regex for JS/TS/Go/Rust.
"""

import ast
import os
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Symbol / File data structures
# ---------------------------------------------------------------------------

@dataclass
class Symbol:
    """A code symbol (function, class, method, type, struct, etc.)."""
    name: str
    kind: str  # "function", "class", "method", "struct", "impl", "type"
    file_path: str
    line: int = 0
    signature: str = ""  # human-readable signature
    is_exported: bool = False
    references: int = 0  # number of times referenced elsewhere

    @property
    def fqn(self) -> str:
        """Fully-qualified name: file_path::name"""
        return f"{self.file_path}::{self.name}"


@dataclass
class FileInfo:
    """Parsed info about a single source file."""
    path: str
    language: str
    symbols: List[Symbol] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)  # imported names
    references: List[str] = field(default_factory=list)  # names referenced
    mtime: float = 0.0


# ---------------------------------------------------------------------------
# Language extensions
# ---------------------------------------------------------------------------

LANG_EXTENSIONS: Dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
}

# Directories to always skip
SKIP_DIRS: Set[str] = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", "venv", ".venv", "env",
    ".env", "dist", "build", ".tox", ".eggs", "egg-info",
    "target", "vendor",
}

# ---------------------------------------------------------------------------
# Python parser (stdlib ast)
# ---------------------------------------------------------------------------

def _parse_python(source: str, file_path: str) -> FileInfo:
    """Parse a Python file using the ast module."""
    info = FileInfo(path=file_path, language="python")
    try:
        tree = ast.parse(source, filename=file_path)
    except (SyntaxError, ValueError):
        # ValueError covers sources with null bytes (binary-like content);
        # SyntaxError covers ill-formed Python.
        return info

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
            # Determine if it's a method (inside a class) or top-level function
            # We check parent via a simple heuristic: top-level if col_offset == 0
            kind = "function" if node.col_offset == 0 else "method"
            args = _py_format_args(node.args)
            sig = f"{node.name}({args})"
            is_exported = not node.name.startswith("_")
            info.symbols.append(Symbol(
                name=node.name, kind=kind, file_path=file_path,
                line=node.lineno, signature=sig, is_exported=is_exported,
            ))

        elif isinstance(node, ast.ClassDef):
            bases = ", ".join(_py_name(b) for b in node.bases)
            sig = f"class {node.name}({bases})" if bases else f"class {node.name}"
            is_exported = not node.name.startswith("_")
            info.symbols.append(Symbol(
                name=node.name, kind="class", file_path=file_path,
                line=node.lineno, signature=sig, is_exported=is_exported,
            ))

        elif isinstance(node, ast.Import):
            for alias in node.names:
                info.imports.append(alias.asname or alias.name)

        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                info.imports.append(alias.asname or alias.name)

        elif isinstance(node, ast.Name):
            info.references.append(node.id)

        elif isinstance(node, ast.Attribute):
            info.references.append(node.attr)

    return info


def _py_format_args(args: ast.arguments) -> str:
    """Format function arguments compactly."""
    parts = []
    for a in args.args:
        name = a.arg
        if name == "self" or name == "cls":
            continue
        parts.append(name)
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return ", ".join(parts)


def _py_name(node: ast.expr) -> str:
    """Extract a name string from an AST expression node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_py_name(node.value)}.{node.attr}"
    return "..."


# ---------------------------------------------------------------------------
# JS / TS parser (regex)
# ---------------------------------------------------------------------------

_JS_PATTERNS = [
    # function declarations
    (re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)", re.MULTILINE),
     "function"),
    # class declarations
    (re.compile(r"^(?:export\s+)?class\s+(\w+)(?:\s+extends\s+(\w+))?", re.MULTILINE),
     "class"),
    # const arrow functions (exported or not)
    (re.compile(r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(?([^)]*)\)?\s*=>", re.MULTILINE),
     "function"),
    # method definitions inside classes: name(args) {
    (re.compile(r"^\s+(?:async\s+)?(\w+)\s*\(([^)]*)\)\s*\{", re.MULTILINE),
     "method"),
]

_JS_IMPORT_RE = re.compile(
    r"(?:import|require)\s*\(?['\"]([^'\"]+)['\"]|"
    r"import\s+\{([^}]+)\}\s+from",
    re.MULTILINE
)

_JS_EXPORT_RE = re.compile(r"^export\s+", re.MULTILINE)


def _parse_js_ts(source: str, file_path: str, language: str) -> FileInfo:
    """Parse JS/TS using regex patterns."""
    info = FileInfo(path=file_path, language=language)

    for pattern, kind in _JS_PATTERNS:
        for m in pattern.finditer(source):
            name = m.group(1)
            args = m.group(2) if m.lastindex >= 2 and m.group(2) else ""
            # Check line start for export keyword
            line_start = source.rfind("\n", 0, m.start()) + 1
            line_text = source[line_start:m.end()]
            is_exported = bool(_JS_EXPORT_RE.match(line_text))

            if kind == "class":
                sig = f"class {name}"
                if m.lastindex >= 2 and m.group(2):
                    sig += f" extends {m.group(2)}"
            else:
                sig = f"{name}({args})"

            line_no = source[:m.start()].count("\n") + 1
            info.symbols.append(Symbol(
                name=name, kind=kind, file_path=file_path,
                line=line_no, signature=sig, is_exported=is_exported,
            ))

    for m in _JS_IMPORT_RE.finditer(source):
        if m.group(1):
            info.imports.append(m.group(1).split("/")[-1])
        if m.group(2):
            for name in m.group(2).split(","):
                name = name.strip().split(" as ")[-1].strip()
                if name:
                    info.imports.append(name)

    # Collect identifier references (simple word boundary scan)
    for word in re.findall(r"\b([A-Za-z_]\w*)\b", source):
        info.references.append(word)

    return info


# ---------------------------------------------------------------------------
# Go parser (regex)
# ---------------------------------------------------------------------------

_GO_PATTERNS = [
    (re.compile(r"^func\s+(\w+)\s*\(([^)]*)\)", re.MULTILINE), "function"),
    (re.compile(r"^func\s+\([^)]+\)\s+(\w+)\s*\(([^)]*)\)", re.MULTILINE), "method"),
    (re.compile(r"^type\s+(\w+)\s+struct\b", re.MULTILINE), "struct"),
    (re.compile(r"^type\s+(\w+)\s+interface\b", re.MULTILINE), "type"),
]

_GO_IMPORT_RE = re.compile(r'"([^"]+)"')


def _parse_go(source: str, file_path: str) -> FileInfo:
    """Parse Go using regex patterns."""
    info = FileInfo(path=file_path, language="go")

    for pattern, kind in _GO_PATTERNS:
        for m in pattern.finditer(source):
            name = m.group(1)
            args = m.group(2) if m.lastindex >= 2 else ""
            is_exported = name[0].isupper() if name else False

            if kind in ("struct", "type"):
                sig = f"type {name} {kind}"
            else:
                sig = f"{name}({args})"

            line_no = source[:m.start()].count("\n") + 1
            info.symbols.append(Symbol(
                name=name, kind=kind, file_path=file_path,
                line=line_no, signature=sig, is_exported=is_exported,
            ))

    for m in _GO_IMPORT_RE.finditer(source):
        info.imports.append(m.group(1).split("/")[-1])

    for word in re.findall(r"\b([A-Za-z_]\w*)\b", source):
        info.references.append(word)

    return info


# ---------------------------------------------------------------------------
# Rust parser (regex)
# ---------------------------------------------------------------------------

_RUST_PATTERNS = [
    (re.compile(r"^\s*(?:pub\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)", re.MULTILINE), "function"),
    (re.compile(r"^(?:pub\s+)?struct\s+(\w+)", re.MULTILINE), "struct"),
    (re.compile(r"^(?:pub\s+)?enum\s+(\w+)", re.MULTILINE), "type"),
    (re.compile(r"^impl(?:<[^>]*>)?\s+(\w+)", re.MULTILINE), "impl"),
    (re.compile(r"^(?:pub\s+)?trait\s+(\w+)", re.MULTILINE), "type"),
]

_RUST_USE_RE = re.compile(r"use\s+[\w:]+::(\w+)", re.MULTILINE)


def _parse_rust(source: str, file_path: str) -> FileInfo:
    """Parse Rust using regex patterns."""
    info = FileInfo(path=file_path, language="rust")

    for pattern, kind in _RUST_PATTERNS:
        for m in pattern.finditer(source):
            name = m.group(1)
            args = m.group(2) if m.lastindex >= 2 else ""
            # Check for pub keyword
            line_start = source.rfind("\n", 0, m.start()) + 1
            line_text = source[line_start:m.end()]
            is_exported = line_text.lstrip().startswith("pub")

            if kind in ("struct", "type", "impl"):
                sig = f"{kind} {name}"
            else:
                # Trim self args for display
                args_clean = re.sub(r"&?\s*(?:mut\s+)?self\s*,?\s*", "", args).strip().rstrip(",")
                sig = f"{name}({args_clean})"

            line_no = source[:m.start()].count("\n") + 1
            info.symbols.append(Symbol(
                name=name, kind=kind, file_path=file_path,
                line=line_no, signature=sig, is_exported=is_exported,
            ))

    for m in _RUST_USE_RE.finditer(source):
        info.imports.append(m.group(1))

    for word in re.findall(r"\b([A-Za-z_]\w*)\b", source):
        info.references.append(word)

    return info


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def parse_file(file_path: str, source: Optional[str] = None) -> Optional[FileInfo]:
    """Parse a source file and return FileInfo, or None if unsupported."""
    ext = os.path.splitext(file_path)[1].lower()
    language = LANG_EXTENSIONS.get(ext)
    if not language:
        return None

    if source is None:
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                source = f.read()
        except (OSError, IOError):
            return None

    try:
        stat = os.stat(file_path)
        mtime = stat.st_mtime
    except OSError:
        mtime = 0.0

    if language == "python":
        info = _parse_python(source, file_path)
    elif language in ("javascript", "typescript"):
        info = _parse_js_ts(source, file_path, language)
    elif language == "go":
        info = _parse_go(source, file_path)
    elif language == "rust":
        info = _parse_rust(source, file_path)
    else:
        return None

    info.mtime = mtime
    return info


# ---------------------------------------------------------------------------
# RepoMap — the main class
# ---------------------------------------------------------------------------

class RepoMap:
    """Scans a project, extracts symbols, ranks by PageRank, produces a map."""

    def __init__(self, root_dir: str, exclude_dirs: Optional[Set[str]] = None):
        self.root_dir = os.path.abspath(root_dir)
        self.exclude_dirs = SKIP_DIRS | (exclude_dirs or set())
        self.files: Dict[str, FileInfo] = {}
        self.symbols: Dict[str, Symbol] = {}  # fqn -> Symbol
        self._ranks: Dict[str, float] = {}  # fqn -> rank score
        self._scanned = False

    # -- scanning ----------------------------------------------------------

    def scan(self) -> "RepoMap":
        """Full scan of the repository."""
        self.files.clear()
        self.symbols.clear()
        self._ranks.clear()

        for fpath in self._iter_files():
            self._scan_file(fpath)

        self._build_references()
        self._compute_ranks()
        self._scanned = True
        return self

    def refresh(self, changed_files: Optional[List[str]] = None) -> "RepoMap":
        """Incremental update. Re-scan only changed files (or all if none given)."""
        if not self._scanned:
            return self.scan()

        if changed_files is None:
            # Re-scan files whose mtime changed
            changed_files = []
            for fpath in self._iter_files():
                try:
                    mtime = os.stat(fpath).st_mtime
                except OSError:
                    continue
                old = self.files.get(fpath)
                if old is None or old.mtime < mtime:
                    changed_files.append(fpath)

        if not changed_files:
            return self

        # Remove old symbols from changed files
        for fpath in changed_files:
            old = self.files.pop(fpath, None)
            if old:
                for sym in old.symbols:
                    self.symbols.pop(sym.fqn, None)

        # Re-parse changed files
        for fpath in changed_files:
            self._scan_file(fpath)

        self._build_references()
        self._compute_ranks()
        return self

    def _iter_files(self):
        """Iterate over source files in the repo."""
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            # Prune excluded dirs in-place
            dirnames[:] = [
                d for d in dirnames
                if d not in self.exclude_dirs and not d.startswith(".")
            ]
            for fname in filenames:
                ext = os.path.splitext(fname)[1].lower()
                if ext in LANG_EXTENSIONS:
                    yield os.path.join(dirpath, fname)

    def _scan_file(self, fpath: str):
        """Parse a single file and register its symbols."""
        info = parse_file(fpath)
        if info is None:
            return
        self.files[fpath] = info
        for sym in info.symbols:
            self.symbols[sym.fqn] = sym

    # -- reference counting ------------------------------------------------

    def _build_references(self):
        """Count how many times each symbol name is referenced across files."""
        # Reset counts
        for sym in self.symbols.values():
            sym.references = 0

        # Build a name -> list of symbols map
        name_to_syms: Dict[str, List[Symbol]] = defaultdict(list)
        for sym in self.symbols.values():
            name_to_syms[sym.name].append(sym)

        # Count references
        for finfo in self.files.values():
            seen_in_file: Set[str] = set()
            for ref_name in finfo.references:
                if ref_name in seen_in_file:
                    continue
                seen_in_file.add(ref_name)
                if ref_name in name_to_syms:
                    for sym in name_to_syms[ref_name]:
                        # Don't count self-references from the same file
                        if sym.file_path != finfo.path:
                            sym.references += 1

    # -- PageRank-inspired ranking -----------------------------------------

    def _compute_ranks(self):
        """Simplified PageRank: references + export boost + recency boost."""
        if not self.symbols:
            return

        now = time.time()
        max_refs = max((s.references for s in self.symbols.values()), default=1) or 1

        for sym in self.symbols.values():
            # Base score from references (normalized 0-1)
            ref_score = sym.references / max_refs

            # Export boost: exported symbols are more important
            export_boost = 0.3 if sym.is_exported else 0.0

            # Recency boost: files modified in last hour get a boost
            finfo = self.files.get(sym.file_path)
            recency_boost = 0.0
            if finfo and finfo.mtime > 0:
                age_hours = (now - finfo.mtime) / 3600
                if age_hours < 1:
                    recency_boost = 0.2
                elif age_hours < 24:
                    recency_boost = 0.1

            # Kind boost: classes/structs slightly more important than functions
            kind_boost = 0.1 if sym.kind in ("class", "struct", "type", "impl") else 0.0

            self._ranks[sym.fqn] = ref_score + export_boost + recency_boost + kind_boost

    # -- map generation ----------------------------------------------------

    def generate_map(self, max_tokens: int = 2000) -> str:
        """Generate a compact text map of the repository's top symbols.

        Format:
            path/file.py: ClassName, function_name(args), other_func(x, y)
            path/other.py: SomeClass, helper(data)

        Fits within approximately max_tokens (estimated at ~4 chars/token).
        """
        if not self._scanned:
            self.scan()

        max_chars = max_tokens * 4  # rough token estimate

        # Sort symbols by rank (descending)
        ranked = sorted(self._ranks.items(), key=lambda x: x[1], reverse=True)

        # Group by file, preserving rank order
        file_symbols: Dict[str, List[Tuple[str, Symbol]]] = defaultdict(list)
        for fqn, _rank in ranked:
            sym = self.symbols.get(fqn)
            if sym:
                file_symbols[sym.file_path].append((fqn, sym))

        # Order files by their best symbol rank
        file_order = []
        file_best_rank: Dict[str, float] = {}
        for fpath, syms in file_symbols.items():
            best = max(self._ranks.get(fqn, 0) for fqn, _ in syms)
            file_best_rank[fpath] = best
            file_order.append(fpath)
        file_order.sort(key=lambda f: file_best_rank[f], reverse=True)

        # Build output
        lines = []
        total_chars = 0

        for fpath in file_order:
            rel_path = os.path.relpath(fpath, self.root_dir)
            syms = file_symbols[fpath]

            # Format symbol signatures
            sig_parts = []
            for _, sym in syms:
                sig_parts.append(sym.signature)

            line = f"{rel_path}: {', '.join(sig_parts)}"

            # Truncate if too long
            if total_chars + len(line) + 1 > max_chars:
                # Try to fit a truncated version
                remaining = max_chars - total_chars - len(rel_path) - 2
                if remaining > 20:
                    truncated_sigs = []
                    used = 0
                    for sig in sig_parts:
                        if used + len(sig) + 2 > remaining:
                            break
                        truncated_sigs.append(sig)
                        used += len(sig) + 2
                    if truncated_sigs:
                        line = f"{rel_path}: {', '.join(truncated_sigs)}"
                        lines.append(line)
                break

            lines.append(line)
            total_chars += len(line) + 1

        return "\n".join(lines)

    def get_symbols_for_file(self, file_path: str) -> List[Symbol]:
        """Get all symbols defined in a specific file."""
        abs_path = os.path.abspath(file_path)
        finfo = self.files.get(abs_path)
        if finfo:
            return finfo.symbols
        return []

    def get_top_symbols(self, n: int = 20) -> List[Tuple[Symbol, float]]:
        """Get top N symbols by rank."""
        ranked = sorted(self._ranks.items(), key=lambda x: x[1], reverse=True)[:n]
        result = []
        for fqn, rank in ranked:
            sym = self.symbols.get(fqn)
            if sym:
                result.append((sym, rank))
        return result


# ---------------------------------------------------------------------------
# RepoMapCache — thread-safe caching wrapper
# ---------------------------------------------------------------------------

class RepoMapCache:
    """Thread-safe cache for RepoMap with mtime-based invalidation."""

    def __init__(self, ttl_seconds: float = 60.0):
        self._lock = threading.Lock()
        self._cache: Dict[str, Tuple[RepoMap, float, str]] = {}
        # key: root_dir -> (repo_map, cached_at, cached_map_text)
        self.ttl = ttl_seconds

    def get_map(self, root_dir: str, max_tokens: int = 2000,
                force_refresh: bool = False) -> str:
        """Get the repo map, using cache if available and fresh."""
        root_dir = os.path.abspath(root_dir)

        with self._lock:
            if not force_refresh and root_dir in self._cache:
                repo_map, cached_at, cached_text = self._cache[root_dir]
                if time.time() - cached_at < self.ttl:
                    return cached_text

            # Need to rebuild
            if root_dir in self._cache:
                repo_map = self._cache[root_dir][0]
                repo_map.refresh()
            else:
                repo_map = RepoMap(root_dir).scan()

            text = repo_map.generate_map(max_tokens=max_tokens)
            self._cache[root_dir] = (repo_map, time.time(), text)
            return text

    def invalidate(self, root_dir: Optional[str] = None):
        """Invalidate cache for a directory (or all)."""
        with self._lock:
            if root_dir:
                self._cache.pop(os.path.abspath(root_dir), None)
            else:
                self._cache.clear()

    def refresh_files(self, root_dir: str, changed_files: List[str],
                      max_tokens: int = 2000) -> str:
        """Incrementally refresh specific files and return updated map."""
        root_dir = os.path.abspath(root_dir)
        with self._lock:
            if root_dir in self._cache:
                repo_map = self._cache[root_dir][0]
            else:
                repo_map = RepoMap(root_dir).scan()

            repo_map.refresh(changed_files=changed_files)
            text = repo_map.generate_map(max_tokens=max_tokens)
            self._cache[root_dir] = (repo_map, time.time(), text)
            return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Module-level cache singleton
_global_cache = RepoMapCache(ttl_seconds=120.0)


def get_repo_map(project_dir: str, max_tokens: int = 2000,
                 force_refresh: bool = False) -> str:
    """Get a compact repository map for inclusion in system prompts.

    This is the main entry point. Call from prompt_builder to include
    codebase structure in the system prompt.

    Args:
        project_dir: Root directory of the project
        max_tokens: Approximate token budget for the map
        force_refresh: Force a full rescan

    Returns:
        A compact text representation of the codebase structure,
        ranked by importance (PageRank-inspired).
    """
    return _global_cache.get_map(project_dir, max_tokens, force_refresh)


def refresh_repo_map(project_dir: str, changed_files: List[str],
                     max_tokens: int = 2000) -> str:
    """Incrementally update the repo map after file changes.

    Call this when you know specific files changed (e.g., after a save).
    """
    return _global_cache.refresh_files(project_dir, changed_files, max_tokens)


def invalidate_repo_map(project_dir: Optional[str] = None):
    """Invalidate cached repo map(s)."""
    _global_cache.invalidate(project_dir)
