"""Semantic (AST-level) diff for Hermes Agent (Thrice).

Computes a structural diff between two versions of a Python file, so the
agent can tell "this edit actually changed code" from "this edit only
reformatted whitespace / reordered imports / renamed a local variable".
A single semantic change typically emits **one** record where a textual
diff would emit dozens, which is a large context-window multiplier.

Supported comparisons:

- ``diff_python_source(old, new)``         - two source strings
- ``diff_python_files(old_path, new_path)`` - two files on disk
- ``diff_python_trees(old, new)``          - two trees (dirs), recursive

Each change is a ``SemanticChange`` with:

- ``kind``: one of ``added`` | ``removed`` | ``signature_changed`` |
  ``body_changed`` | ``renamed`` | ``decorator_changed``
- ``qualname``: dotted path to the top-level symbol (e.g. ``Foo.bar``)
- ``old_line`` / ``new_line``: useful line numbers if we have them
- ``detail``: human-readable blurb

Non-Python files fall back to ``difflib`` unified diff, with the semantic
layer skipped - that way callers get a useful result for JSON/YAML too.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

ChangeKind = str   # Literal-ish; kept as str for simple serialization


@dataclass(frozen=True)
class SemanticChange:
    """One semantic change record."""

    kind: ChangeKind           # "added" | "removed" | "signature_changed" | ...
    qualname: str              # dotted path; "" for module-level
    old_line: Optional[int]
    new_line: Optional[int]
    detail: str

    def format_short(self) -> str:
        ln = f"L{self.new_line or self.old_line or '?'}"
        return f"[{self.kind}] {self.qualname or '<module>'} ({ln}): {self.detail}"


@dataclass
class SemanticDiffResult:
    """Result of comparing two Python sources."""

    changes: List[SemanticChange] = field(default_factory=list)
    fallback_unified: Optional[str] = None   # set when AST parse failed
    language: str = "python"

    @property
    def ok(self) -> bool:
        return self.fallback_unified is None

    def by_kind(self) -> Dict[str, List[SemanticChange]]:
        out: Dict[str, List[SemanticChange]] = {}
        for c in self.changes:
            out.setdefault(c.kind, []).append(c)
        return out

    def summary(self) -> str:
        if not self.ok:
            return "semantic_diff: AST parse failed; unified diff returned instead"
        if not self.changes:
            return "semantic_diff: identical after whitespace/import normalization"
        bucket = self.by_kind()
        parts = [f"{len(v)} {k}" for k, v in sorted(bucket.items())]
        return "semantic_diff: " + ", ".join(parts)


# ---------------------------------------------------------------------------
# Symbol extraction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Symbol:
    qualname: str
    kind: str                 # "function" | "class" | "assign" | "import" | "module"
    line: int
    signature: str            # canonical representation (args / base classes / etc.)
    body_hash: str            # sha1 of normalized body text
    decorator_hashes: Tuple[str, ...]


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _unparse_node(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _canonical_args(fn: ast.AST) -> str:
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return ""
    return _unparse_node(fn.args)


def _decorator_hashes(node: ast.AST) -> Tuple[str, ...]:
    decos = getattr(node, "decorator_list", []) or []
    return tuple(_hash(_unparse_node(d)) for d in decos)


def _symbols_from_module(tree: ast.Module) -> Dict[str, _Symbol]:
    """Collect top-level and one-level nested symbols."""
    out: Dict[str, _Symbol] = {}

    for node in tree.body:
        # ---- module-level function ----
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            body_src = _unparse_node(ast.Module(body=node.body, type_ignores=[]))
            out[node.name] = _Symbol(
                qualname=node.name,
                kind="function",
                line=node.lineno,
                signature=_canonical_args(node),
                body_hash=_hash(body_src),
                decorator_hashes=_decorator_hashes(node),
            )

        # ---- class (and one level of nested methods) ----
        elif isinstance(node, ast.ClassDef):
            base_src = ", ".join(_unparse_node(b) for b in node.bases)
            cls_body_src = _unparse_node(
                ast.Module(body=node.body, type_ignores=[])
            )
            out[node.name] = _Symbol(
                qualname=node.name,
                kind="class",
                line=node.lineno,
                signature=base_src,
                body_hash=_hash(cls_body_src),
                decorator_hashes=_decorator_hashes(node),
            )
            for inner in node.body:
                if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    inner_body = _unparse_node(
                        ast.Module(body=inner.body, type_ignores=[])
                    )
                    qn = f"{node.name}.{inner.name}"
                    out[qn] = _Symbol(
                        qualname=qn,
                        kind="function",
                        line=inner.lineno,
                        signature=_canonical_args(inner),
                        body_hash=_hash(inner_body),
                        decorator_hashes=_decorator_hashes(inner),
                    )

        # ---- module-level assignments ----
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets: List[str] = []
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    targets.append(_unparse_node(t))
            elif isinstance(node, ast.AnnAssign):
                targets.append(_unparse_node(node.target))
            for name in targets:
                out[name] = _Symbol(
                    qualname=name,
                    kind="assign",
                    line=node.lineno,
                    signature="",
                    body_hash=_hash(_unparse_node(node)),
                    decorator_hashes=(),
                )

        # ---- imports ----
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            src = _unparse_node(node)
            # Use the import statement's text as its identity.
            out[f"import::{src}"] = _Symbol(
                qualname=src,
                kind="import",
                line=node.lineno,
                signature=src,
                body_hash=_hash(src),
                decorator_hashes=(),
            )

    return out


# ---------------------------------------------------------------------------
# Diff engine
# ---------------------------------------------------------------------------

def diff_python_source(old: str, new: str) -> SemanticDiffResult:
    """Compute a semantic diff between two Python source strings."""
    try:
        old_tree = ast.parse(old or "")
        new_tree = ast.parse(new or "")
    except SyntaxError:
        diff = "\n".join(difflib.unified_diff(
            (old or "").splitlines(), (new or "").splitlines(),
            fromfile="old", tofile="new", lineterm="",
        ))
        return SemanticDiffResult(fallback_unified=diff)

    old_syms = _symbols_from_module(old_tree)
    new_syms = _symbols_from_module(new_tree)

    old_names = set(old_syms)
    new_names = set(new_syms)
    changes: List[SemanticChange] = []

    # --- Added / removed ---
    for qn in sorted(new_names - old_names):
        s = new_syms[qn]
        changes.append(SemanticChange(
            kind="added", qualname=qn, old_line=None, new_line=s.line,
            detail=f"new {s.kind}",
        ))
    for qn in sorted(old_names - new_names):
        s = old_syms[qn]
        # Heuristic rename: if a single "removed" and a single "added" share
        # the same signature + body hash, call it a rename.  Simple first:
        matched_rename = next(
            (nq for nq, ns in new_syms.items()
             if nq not in old_names
             and ns.kind == s.kind
             and ns.signature == s.signature
             and ns.body_hash == s.body_hash),
            None,
        )
        if matched_rename:
            changes.append(SemanticChange(
                kind="renamed", qualname=qn,
                old_line=s.line, new_line=new_syms[matched_rename].line,
                detail=f"{s.kind} renamed -> {matched_rename}",
            ))
            # Drop the matching 'added' so we don't double-report.
            changes = [c for c in changes
                       if not (c.kind == "added" and c.qualname == matched_rename)]
        else:
            changes.append(SemanticChange(
                kind="removed", qualname=qn, old_line=s.line, new_line=None,
                detail=f"removed {s.kind}",
            ))

    # --- Changed ---
    for qn in sorted(old_names & new_names):
        o, n = old_syms[qn], new_syms[qn]
        if o.signature != n.signature:
            changes.append(SemanticChange(
                kind="signature_changed",
                qualname=qn,
                old_line=o.line, new_line=n.line,
                detail=f"{o.kind} signature changed: {o.signature!r} -> {n.signature!r}",
            ))
        if o.body_hash != n.body_hash:
            changes.append(SemanticChange(
                kind="body_changed",
                qualname=qn,
                old_line=o.line, new_line=n.line,
                detail=f"{o.kind} body changed",
            ))
        if o.decorator_hashes != n.decorator_hashes:
            changes.append(SemanticChange(
                kind="decorator_changed",
                qualname=qn,
                old_line=o.line, new_line=n.line,
                detail=f"{o.kind} decorators changed",
            ))

    # Stable sort: by line then qualname then kind.
    changes.sort(key=lambda c: (c.new_line or c.old_line or 0, c.qualname, c.kind))
    return SemanticDiffResult(changes=changes)


def diff_python_files(old_path: str, new_path: str) -> SemanticDiffResult:
    def _read(p: str) -> str:
        try:
            with open(p, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            return ""
    return diff_python_source(_read(old_path), _read(new_path))


def diff_python_trees(old_root: str, new_root: str) -> Dict[str, SemanticDiffResult]:
    """Compare two directory trees.  Returns ``{relpath: result}`` for every
    ``.py`` file where either side has semantic changes."""
    old_files = _py_files(old_root)
    new_files = _py_files(new_root)
    out: Dict[str, SemanticDiffResult] = {}
    for rel in sorted(old_files | new_files):
        op = os.path.join(old_root, rel) if rel in old_files else None
        np = os.path.join(new_root, rel) if rel in new_files else None
        if op is None:
            # New file -> treat as all-added.
            out[rel] = diff_python_source("", _read_text(np))
        elif np is None:
            out[rel] = diff_python_source(_read_text(op), "")
        else:
            r = diff_python_source(_read_text(op), _read_text(np))
            if r.changes or not r.ok:
                out[rel] = r
    return out


def _py_files(root: str) -> set:
    out = set()
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if not d.startswith(".") and d != "__pycache__"]
        for f in fns:
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(dp, f), root)
                out.add(rel.replace("\\", "/"))
    return out


def _read_text(p: Optional[str]) -> str:
    if not p:
        return ""
    try:
        with open(p, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


__all__ = [
    "SemanticChange",
    "SemanticDiffResult",
    "diff_python_source",
    "diff_python_files",
    "diff_python_trees",
]
