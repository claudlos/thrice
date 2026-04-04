"""
Change Impact Analysis — Dependency-aware analysis of code changes and their ripple effects.

Detects what changed between two versions of a file (function signatures, class
definitions, parameters, etc.), then scans a project to find all code that
depends on the changed interfaces. Produces human-readable reports and
LLM-optimized prompts for making the necessary updates.

Uses only stdlib: ast, re, os, pathlib. No external dependencies.
"""

import ast
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ChangeType(Enum):
    """Types of code changes that can be detected."""
    FUNCTION_RENAMED = "function_renamed"
    FUNCTION_SIGNATURE_CHANGED = "function_signature_changed"
    FUNCTION_REMOVED = "function_removed"
    CLASS_RENAMED = "class_renamed"
    CLASS_REMOVED = "class_removed"
    METHOD_CHANGED = "method_changed"
    PARAMETER_ADDED = "parameter_added"
    PARAMETER_REMOVED = "parameter_removed"
    PARAMETER_RENAMED = "parameter_renamed"
    RETURN_TYPE_CHANGED = "return_type_changed"
    IMPORT_CHANGED = "import_changed"
    CONSTANT_CHANGED = "constant_changed"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DetectedChange:
    """A single detected change between old and new versions of code."""
    change_type: ChangeType
    file_path: str
    symbol_name: str
    old_signature: str = ""
    new_signature: str = ""
    line_number: int = 0


@dataclass
class Usage:
    """A single usage of a symbol in a file."""
    line_number: int
    line_content: str
    usage_type: str  # "call", "import", "inheritance", "type_hint"


@dataclass
class ImpactedFile:
    """A file that is impacted by one or more detected changes."""
    file_path: str
    usages: List[Usage] = field(default_factory=list)
    risk_level: str = "low"  # "high", "medium", "low"
    suggested_update: str = ""


# ---------------------------------------------------------------------------
# Directories / files to skip during scanning
# ---------------------------------------------------------------------------

SKIP_DIRS: Set[str] = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__",
    ".mypy_cache", ".pytest_cache", "venv", ".venv", "env",
    ".env", "dist", "build", ".tox", ".eggs", "egg-info",
    "target", "vendor",
}


# ---------------------------------------------------------------------------
# Signature extraction helpers
# ---------------------------------------------------------------------------

def _extract_functions_ast(source: str) -> Dict[str, dict]:
    """Extract function/method signatures from Python source using AST.

    Returns dict: name -> {signature, line, params, return_annotation, is_method, class_name}
    """
    result = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return result

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            params = []
            for arg in node.args.args:
                param_name = arg.arg
                annotation = ""
                if arg.annotation:
                    annotation = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else ""
                params.append({"name": param_name, "annotation": annotation})

            # Check defaults
            defaults = node.args.defaults
            num_defaults = len(defaults)
            num_params = len(params)
            for i, d in enumerate(defaults):
                idx = num_params - num_defaults + i
                if idx >= 0:
                    default_val = ast.unparse(d) if hasattr(ast, 'unparse') else "..."
                    params[idx]["default"] = default_val

            # *args, **kwargs
            if node.args.vararg:
                params.append({"name": f"*{node.args.vararg.arg}", "annotation": ""})
            if node.args.kwarg:
                params.append({"name": f"**{node.args.kwarg.arg}", "annotation": ""})

            return_ann = ""
            if node.returns:
                return_ann = ast.unparse(node.returns) if hasattr(ast, 'unparse') else ""

            # Build signature string
            param_strs = []
            for p in params:
                s = p["name"]
                if p.get("annotation"):
                    s += f": {p['annotation']}"
                if p.get("default"):
                    s += f" = {p['default']}"
                param_strs.append(s)
            sig = f"def {node.name}({', '.join(param_strs)})"
            if return_ann:
                sig += f" -> {return_ann}"

            # Determine if it's a method (nested in a class)
            class_name = ""
            is_method = False
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef):
                    for child in ast.iter_child_nodes(parent):
                        if child is node:
                            is_method = True
                            class_name = parent.name
                            break

            key = f"{class_name}.{node.name}" if is_method else node.name
            result[key] = {
                "signature": sig,
                "line": node.lineno,
                "params": params,
                "return_annotation": return_ann,
                "is_method": is_method,
                "class_name": class_name,
                "name": node.name,
            }

    return result


def _extract_classes_ast(source: str) -> Dict[str, dict]:
    """Extract class definitions from Python source using AST.

    Returns dict: name -> {line, bases, signature}
    """
    result = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return result

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            bases = []
            for base in node.bases:
                bases.append(ast.unparse(base) if hasattr(ast, 'unparse') else "...")
            sig = f"class {node.name}"
            if bases:
                sig += f"({', '.join(bases)})"
            result[node.name] = {
                "line": node.lineno,
                "bases": bases,
                "signature": sig,
            }

    return result


def _extract_constants(source: str) -> Dict[str, dict]:
    """Extract top-level constant assignments (UPPER_CASE names)."""
    result = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return result

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    value = ast.unparse(node.value) if hasattr(ast, 'unparse') else "..."
                    result[target.id] = {
                        "line": node.lineno,
                        "value": value,
                    }

    return result


def _extract_functions_regex(source: str) -> Dict[str, dict]:
    """Fallback regex-based function extraction for non-parseable code."""
    result = {}
    pattern = re.compile(
        r'^(\s*)(async\s+)?def\s+(\w+)\s*\(([^)]*)\)\s*(?:->\s*([^\s:]+))?\s*:',
        re.MULTILINE,
    )
    for m in pattern.finditer(source):
        indent = m.group(1)
        name = m.group(3)
        params_str = m.group(4).strip()
        return_ann = m.group(5) or ""
        line = source[:m.start()].count('\n') + 1

        sig = f"def {name}({params_str})"
        if return_ann:
            sig += f" -> {return_ann}"

        params = []
        if params_str:
            for p in params_str.split(','):
                p = p.strip()
                if p:
                    pname = p.split(':')[0].split('=')[0].strip()
                    params.append({"name": pname, "annotation": ""})

        is_method = len(indent) > 0
        result[name] = {
            "signature": sig,
            "line": line,
            "params": params,
            "return_annotation": return_ann,
            "is_method": is_method,
            "class_name": "",
            "name": name,
        }

    return result


def _extract_classes_regex(source: str) -> Dict[str, dict]:
    """Fallback regex-based class extraction."""
    result = {}
    pattern = re.compile(r'^class\s+(\w+)\s*(?:\(([^)]*)\))?\s*:', re.MULTILINE)
    for m in pattern.finditer(source):
        name = m.group(1)
        bases_str = m.group(2) or ""
        line = source[:m.start()].count('\n') + 1
        bases = [b.strip() for b in bases_str.split(',') if b.strip()] if bases_str else []
        sig = f"class {name}"
        if bases:
            sig += f"({', '.join(bases)})"
        result[name] = {"line": line, "bases": bases, "signature": sig}

    return result


# ---------------------------------------------------------------------------
# ChangeDetector
# ---------------------------------------------------------------------------

class ChangeDetector:
    """Detects changes between two versions of a Python source file."""

    def detect_changes(
        self, old_content: str, new_content: str, file_path: str
    ) -> List[DetectedChange]:
        """Detect all changes between old and new content.

        Compares function signatures, class definitions, constants, etc.
        Returns a list of DetectedChange objects.
        """
        changes: List[DetectedChange] = []

        # Extract structures from both versions
        old_funcs = _extract_functions_ast(old_content) or _extract_functions_regex(old_content)
        new_funcs = _extract_functions_ast(new_content) or _extract_functions_regex(new_content)
        old_classes = _extract_classes_ast(old_content) or _extract_classes_regex(old_content)
        new_classes = _extract_classes_ast(new_content) or _extract_classes_regex(new_content)
        old_constants = _extract_constants(old_content)
        new_constants = _extract_constants(new_content)

        # Detect function changes
        changes.extend(self._detect_function_changes(old_funcs, new_funcs, file_path))

        # Detect class changes
        changes.extend(self._detect_class_changes(old_classes, new_classes, file_path))

        # Detect constant changes
        changes.extend(self._detect_constant_changes(old_constants, new_constants, file_path))

        return changes

    def _detect_function_changes(
        self,
        old_funcs: Dict[str, dict],
        new_funcs: Dict[str, dict],
        file_path: str,
    ) -> List[DetectedChange]:
        """Detect changes in function/method signatures."""
        changes: List[DetectedChange] = []
        old_names = set(old_funcs.keys())
        new_names = set(new_funcs.keys())

        # Removed functions
        for name in old_names - new_names:
            # Check if renamed (similar signature exists in new)
            renamed_to = self._find_rename(old_funcs[name], new_funcs, old_names)
            if renamed_to:
                info = old_funcs[name]
                new_info = new_funcs[renamed_to]
                ct = ChangeType.FUNCTION_RENAMED
                if info.get("is_method"):
                    ct = ChangeType.METHOD_CHANGED
                changes.append(DetectedChange(
                    change_type=ct,
                    file_path=file_path,
                    symbol_name=name,
                    old_signature=info["signature"],
                    new_signature=new_info["signature"],
                    line_number=info["line"],
                ))
            else:
                info = old_funcs[name]
                changes.append(DetectedChange(
                    change_type=ChangeType.FUNCTION_REMOVED,
                    file_path=file_path,
                    symbol_name=name,
                    old_signature=info["signature"],
                    new_signature="",
                    line_number=info["line"],
                ))

        # Changed functions (present in both)
        for name in old_names & new_names:
            old_info = old_funcs[name]
            new_info = new_funcs[name]

            if old_info["signature"] != new_info["signature"]:
                diffs = self.compare_signatures(old_info["signature"], new_info["signature"])
                change_type = self._classify_function_change(old_info, new_info, diffs)
                changes.append(DetectedChange(
                    change_type=change_type,
                    file_path=file_path,
                    symbol_name=name,
                    old_signature=old_info["signature"],
                    new_signature=new_info["signature"],
                    line_number=new_info["line"],
                ))

        return changes

    def _detect_class_changes(
        self,
        old_classes: Dict[str, dict],
        new_classes: Dict[str, dict],
        file_path: str,
    ) -> List[DetectedChange]:
        """Detect changes in class definitions."""
        changes: List[DetectedChange] = []
        old_names = set(old_classes.keys())
        new_names = set(new_classes.keys())

        # Removed classes
        for name in old_names - new_names:
            # Check if renamed
            renamed_to = None
            for new_name in new_names - old_names:
                if old_classes[name].get("bases") == new_classes[new_name].get("bases"):
                    renamed_to = new_name
                    break
            if renamed_to:
                changes.append(DetectedChange(
                    change_type=ChangeType.CLASS_RENAMED,
                    file_path=file_path,
                    symbol_name=name,
                    old_signature=old_classes[name]["signature"],
                    new_signature=new_classes[renamed_to]["signature"],
                    line_number=old_classes[name]["line"],
                ))
            else:
                changes.append(DetectedChange(
                    change_type=ChangeType.CLASS_REMOVED,
                    file_path=file_path,
                    symbol_name=name,
                    old_signature=old_classes[name]["signature"],
                    new_signature="",
                    line_number=old_classes[name]["line"],
                ))

        return changes

    def _detect_constant_changes(
        self,
        old_constants: Dict[str, dict],
        new_constants: Dict[str, dict],
        file_path: str,
    ) -> List[DetectedChange]:
        """Detect changes in top-level constants."""
        changes: List[DetectedChange] = []

        for name in old_constants:
            if name not in new_constants:
                changes.append(DetectedChange(
                    change_type=ChangeType.CONSTANT_CHANGED,
                    file_path=file_path,
                    symbol_name=name,
                    old_signature=f"{name} = {old_constants[name]['value']}",
                    new_signature="<removed>",
                    line_number=old_constants[name]["line"],
                ))
            elif old_constants[name]["value"] != new_constants[name]["value"]:
                changes.append(DetectedChange(
                    change_type=ChangeType.CONSTANT_CHANGED,
                    file_path=file_path,
                    symbol_name=name,
                    old_signature=f"{name} = {old_constants[name]['value']}",
                    new_signature=f"{name} = {new_constants[name]['value']}",
                    line_number=new_constants[name]["line"],
                ))

        return changes

    def _find_rename(
        self, old_info: dict, new_funcs: Dict[str, dict], old_names: Set[str]
    ) -> Optional[str]:
        """Check if a removed function was actually renamed.

        Heuristic: if a new function has the same parameters (ignoring name),
        it's likely a rename.
        """
        old_params = [p["name"] for p in old_info.get("params", []) if not p["name"].startswith("self")]
        for new_name, new_info in new_funcs.items():
            if new_name in old_names:
                continue  # Already exists in old
            new_params = [p["name"] for p in new_info.get("params", []) if not p["name"].startswith("self")]
            if old_params == new_params and len(old_params) > 0:
                return new_name
        return None

    def _classify_function_change(
        self, old_info: dict, new_info: dict, diffs: List[str]
    ) -> ChangeType:
        """Classify the type of function change based on differences."""
        old_params = {p["name"] for p in old_info.get("params", [])}
        new_params = {p["name"] for p in new_info.get("params", [])}

        if old_info.get("is_method"):
            return ChangeType.METHOD_CHANGED

        # Check return type change
        if old_info.get("return_annotation", "") != new_info.get("return_annotation", ""):
            if old_params == new_params:
                return ChangeType.RETURN_TYPE_CHANGED

        added = new_params - old_params
        removed = old_params - new_params

        if added and not removed:
            return ChangeType.PARAMETER_ADDED
        if removed and not added:
            return ChangeType.PARAMETER_REMOVED
        if added and removed and len(added) == len(removed):
            return ChangeType.PARAMETER_RENAMED

        return ChangeType.FUNCTION_SIGNATURE_CHANGED

    def compare_signatures(self, old_sig: str, new_sig: str) -> List[str]:
        """Compare two signature strings and return a list of human-readable differences."""
        diffs: List[str] = []

        if old_sig == new_sig:
            return diffs

        # Extract function names
        old_name_m = re.match(r'(?:async\s+)?def\s+(\w+)', old_sig)
        new_name_m = re.match(r'(?:async\s+)?def\s+(\w+)', new_sig)

        if old_name_m and new_name_m:
            if old_name_m.group(1) != new_name_m.group(1):
                diffs.append(f"name: {old_name_m.group(1)} -> {new_name_m.group(1)}")

        # Extract parameters
        old_params = self._extract_params_from_sig(old_sig)
        new_params = self._extract_params_from_sig(new_sig)

        old_param_names = [p.split(':')[0].split('=')[0].strip() for p in old_params]
        new_param_names = [p.split(':')[0].split('=')[0].strip() for p in new_params]

        for p in old_param_names:
            if p and p not in new_param_names:
                diffs.append(f"parameter removed: {p}")

        for p in new_param_names:
            if p and p not in old_param_names:
                diffs.append(f"parameter added: {p}")

        # Check type annotations changed
        for op in old_params:
            op_name = op.split(':')[0].split('=')[0].strip()
            for np_ in new_params:
                np_name = np_.split(':')[0].split('=')[0].strip()
                if op_name == np_name and op.strip() != np_.strip():
                    diffs.append(f"parameter changed: {op.strip()} -> {np_.strip()}")

        # Return type
        old_ret = self._extract_return_type(old_sig)
        new_ret = self._extract_return_type(new_sig)
        if old_ret != new_ret:
            diffs.append(f"return type: {old_ret or 'None'} -> {new_ret or 'None'}")

        if not diffs:
            diffs.append("signature changed (details unclear)")

        return diffs

    def _extract_params_from_sig(self, sig: str) -> List[str]:
        """Extract parameter list from a signature string."""
        m = re.search(r'\(([^)]*)\)', sig)
        if not m:
            return []
        params_str = m.group(1).strip()
        if not params_str:
            return []
        # Simple split by comma (doesn't handle nested generics perfectly)
        return [p.strip() for p in params_str.split(',')]

    def _extract_return_type(self, sig: str) -> str:
        """Extract return type annotation from a signature string."""
        m = re.search(r'->\s*(.+)$', sig)
        return m.group(1).strip() if m else ""


# ---------------------------------------------------------------------------
# ImpactAnalyzer
# ---------------------------------------------------------------------------

class ImpactAnalyzer:
    """Analyzes the impact of detected changes across a project."""

    def analyze_impact(
        self, changes: List[DetectedChange], project_root: str
    ) -> List[ImpactedFile]:
        """For each change, search the project for files that use the changed symbol.

        Returns a list of ImpactedFile objects with usage details and risk levels.
        """
        if not changes:
            return []

        # Collect all Python files in the project
        py_files = self._collect_python_files(project_root)

        # Build a map of symbol_name -> changes for efficient lookup
        symbol_changes: Dict[str, List[DetectedChange]] = {}
        for change in changes:
            # Use the base name (without class prefix) for searching
            base_name = change.symbol_name.split('.')[-1]
            symbol_changes.setdefault(base_name, []).append(change)

        impacted: Dict[str, ImpactedFile] = {}

        for py_file in py_files:
            # Skip the file that was changed itself
            rel_path = os.path.relpath(py_file, project_root)
            skip = False
            for change in changes:
                if change.file_path == py_file or change.file_path == rel_path:
                    skip = True
                    break
            if skip:
                continue

            try:
                content = Path(py_file).read_text(errors="replace")
            except (OSError, UnicodeDecodeError):
                continue

            lines = content.splitlines()

            for symbol_name, symbol_changes_list in symbol_changes.items():
                usages = self._find_usages(symbol_name, lines, py_file)
                if usages:
                    change = symbol_changes_list[0]
                    risk = self._classify_risk(usages)
                    suggestion = self.suggest_updates(change, usages[0])

                    if py_file in impacted:
                        impacted[py_file].usages.extend(usages)
                        # Upgrade risk level if needed
                        if self._risk_value(risk) > self._risk_value(impacted[py_file].risk_level):
                            impacted[py_file].risk_level = risk
                        if not impacted[py_file].suggested_update:
                            impacted[py_file].suggested_update = suggestion
                    else:
                        impacted[py_file] = ImpactedFile(
                            file_path=py_file,
                            usages=usages,
                            risk_level=risk,
                            suggested_update=suggestion,
                        )

        return list(impacted.values())

    def suggest_updates(self, change: DetectedChange, usage: Usage) -> str:
        """Generate a suggested update description for a specific usage."""
        ct = change.change_type
        name = change.symbol_name.split('.')[-1]

        if ct == ChangeType.FUNCTION_RENAMED:
            new_name = ""
            m = re.match(r'(?:async\s+)?def\s+(\w+)', change.new_signature)
            if m:
                new_name = m.group(1)
            return f"Rename '{name}' to '{new_name}' at line {usage.line_number}"

        if ct == ChangeType.FUNCTION_REMOVED:
            return f"Remove or replace usage of removed function '{name}' at line {usage.line_number}"

        if ct == ChangeType.CLASS_RENAMED:
            new_name = ""
            m = re.match(r'class\s+(\w+)', change.new_signature)
            if m:
                new_name = m.group(1)
            return f"Rename class '{name}' to '{new_name}' at line {usage.line_number}"

        if ct == ChangeType.CLASS_REMOVED:
            return f"Remove or replace usage of removed class '{name}' at line {usage.line_number}"

        if ct == ChangeType.PARAMETER_ADDED:
            return f"Update call to '{name}' — new parameter(s) added. New signature: {change.new_signature}"

        if ct == ChangeType.PARAMETER_REMOVED:
            return f"Update call to '{name}' — parameter(s) removed. New signature: {change.new_signature}"

        if ct == ChangeType.PARAMETER_RENAMED:
            return f"Update keyword arguments in calls to '{name}'. New signature: {change.new_signature}"

        if ct == ChangeType.RETURN_TYPE_CHANGED:
            return f"Check handling of return value from '{name}' — return type changed"

        if ct == ChangeType.METHOD_CHANGED:
            return f"Update calls to method '{name}'. New signature: {change.new_signature}"

        if ct == ChangeType.CONSTANT_CHANGED:
            return f"Constant '{name}' changed: {change.old_signature} -> {change.new_signature}"

        return f"Update usage of '{name}' to match new signature: {change.new_signature}"

    def _collect_python_files(self, project_root: str) -> List[str]:
        """Collect all Python files in the project, respecting skip dirs."""
        py_files: List[str] = []
        for dirpath, dirnames, filenames in os.walk(project_root):
            # Filter out skip directories in-place
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fname in filenames:
                if fname.endswith('.py'):
                    py_files.append(os.path.join(dirpath, fname))
        return py_files

    def _find_usages(
        self, symbol_name: str, lines: List[str], file_path: str
    ) -> List[Usage]:
        """Find all usages of a symbol name in a file's lines."""
        usages: List[Usage] = []
        # Build a pattern that matches the symbol as a whole word
        pattern = re.compile(r'\b' + re.escape(symbol_name) + r'\b')

        for i, line in enumerate(lines, 1):
            if pattern.search(line):
                usage_type = self._classify_usage(symbol_name, line)
                usages.append(Usage(
                    line_number=i,
                    line_content=line.rstrip(),
                    usage_type=usage_type,
                ))

        return usages

    def _classify_usage(self, symbol_name: str, line: str) -> str:
        """Classify how a symbol is used in a line of code."""
        stripped = line.strip()

        # Import
        if stripped.startswith(('import ', 'from ')) and symbol_name in stripped:
            return "import"

        # Inheritance: class Foo(SymbolName)
        if re.match(r'class\s+\w+\s*\(.*\b' + re.escape(symbol_name) + r'\b', stripped):
            return "inheritance"

        # Type hint: var: SymbolName or -> SymbolName
        if re.search(r':\s*' + re.escape(symbol_name) + r'\b', stripped) or \
           re.search(r'->\s*' + re.escape(symbol_name) + r'\b', stripped):
            return "type_hint"

        # Default: call / reference
        return "call"

    def _classify_risk(self, usages: List[Usage]) -> str:
        """Classify the risk level based on usage types."""
        types = {u.usage_type for u in usages}

        if "call" in types or "inheritance" in types:
            return "high"
        if "import" in types:
            return "medium"
        return "low"

    def _risk_value(self, risk: str) -> int:
        """Numeric risk for comparison."""
        return {"high": 3, "medium": 2, "low": 1}.get(risk, 0)


# ---------------------------------------------------------------------------
# ImpactReport
# ---------------------------------------------------------------------------

class ImpactReport:
    """Generates human-readable and LLM-optimized reports from impact analysis."""

    def __init__(
        self,
        changes: Optional[List[DetectedChange]] = None,
        impacts: Optional[List[ImpactedFile]] = None,
    ):
        self.changes: List[DetectedChange] = changes or []
        self.impacts: List[ImpactedFile] = impacts or []

    def generate(
        self,
        changes: Optional[List[DetectedChange]] = None,
        impacts: Optional[List[ImpactedFile]] = None,
    ) -> str:
        """Generate a human-readable impact report."""
        changes = changes or self.changes
        impacts = impacts or self.impacts

        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("CHANGE IMPACT ANALYSIS REPORT")
        lines.append("=" * 60)
        lines.append("")

        # Summary
        s = self.summary(changes, impacts)
        lines.append(f"Changes detected: {s['total_changes']}")
        lines.append(f"Files impacted:   {s['total_impacted']}")
        lines.append(f"  High risk:   {s['high']}")
        lines.append(f"  Medium risk: {s['medium']}")
        lines.append(f"  Low risk:    {s['low']}")
        lines.append("")

        # Changes section
        lines.append("-" * 40)
        lines.append("DETECTED CHANGES")
        lines.append("-" * 40)
        for i, change in enumerate(changes, 1):
            lines.append(f"\n{i}. [{change.change_type.value}] {change.symbol_name}")
            lines.append(f"   File: {change.file_path}:{change.line_number}")
            if change.old_signature:
                lines.append(f"   Old: {change.old_signature}")
            if change.new_signature:
                lines.append(f"   New: {change.new_signature}")

        lines.append("")

        # Impacted files section
        lines.append("-" * 40)
        lines.append("IMPACTED FILES")
        lines.append("-" * 40)

        # Sort by risk: high first
        sorted_impacts = sorted(
            impacts,
            key=lambda f: {"high": 0, "medium": 1, "low": 2}.get(f.risk_level, 3),
        )

        for impact in sorted_impacts:
            risk_marker = {"high": "!!!", "medium": "!!", "low": "!"}.get(impact.risk_level, "")
            lines.append(f"\n{risk_marker} {impact.file_path} [{impact.risk_level.upper()} RISK]")
            if impact.suggested_update:
                lines.append(f"   Suggestion: {impact.suggested_update}")
            for usage in impact.usages:
                lines.append(f"   L{usage.line_number} ({usage.usage_type}): {usage.line_content.strip()}")

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def generate_prompt(
        self,
        changes: Optional[List[DetectedChange]] = None,
        impacts: Optional[List[ImpactedFile]] = None,
    ) -> str:
        """Generate an LLM-optimized prompt for making the necessary updates."""
        changes = changes or self.changes
        impacts = impacts or self.impacts

        parts: List[str] = []
        parts.append("You need to update the following files to account for code changes.")
        parts.append("")
        parts.append("## Changes Made")
        parts.append("")

        for change in changes:
            parts.append(f"- **{change.change_type.value}**: `{change.symbol_name}` in `{change.file_path}`")
            if change.old_signature and change.new_signature:
                parts.append(f"  - Old: `{change.old_signature}`")
                parts.append(f"  - New: `{change.new_signature}`")
            elif change.old_signature:
                parts.append(f"  - Removed: `{change.old_signature}`")

        parts.append("")
        parts.append("## Files That Need Updating")
        parts.append("")

        sorted_impacts = sorted(
            impacts,
            key=lambda f: {"high": 0, "medium": 1, "low": 2}.get(f.risk_level, 3),
        )

        for impact in sorted_impacts:
            parts.append(f"### `{impact.file_path}` ({impact.risk_level} risk)")
            if impact.suggested_update:
                parts.append(f"Action: {impact.suggested_update}")
            parts.append("Usages to update:")
            for usage in impact.usages:
                parts.append(f"- Line {usage.line_number} ({usage.usage_type}): `{usage.line_content.strip()}`")
            parts.append("")

        parts.append("Please update each file to be compatible with the changes listed above.")
        parts.append("Preserve existing behavior where possible.")

        return "\n".join(parts)

    def summary(
        self,
        changes: Optional[List[DetectedChange]] = None,
        impacts: Optional[List[ImpactedFile]] = None,
    ) -> dict:
        """Return a summary dict with counts."""
        changes = changes or self.changes
        impacts = impacts or self.impacts

        high = sum(1 for f in impacts if f.risk_level == "high")
        medium = sum(1 for f in impacts if f.risk_level == "medium")
        low = sum(1 for f in impacts if f.risk_level == "low")

        return {
            "total_changes": len(changes),
            "total_impacted": len(impacts),
            "high": high,
            "medium": medium,
            "low": low,
        }


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def quick_impact_check(
    old_content: str,
    new_content: str,
    file_path: str,
    project_root: str,
) -> ImpactReport:
    """One-call convenience: detect changes and analyze their impact.

    Returns a fully populated ImpactReport.
    """
    detector = ChangeDetector()
    analyzer = ImpactAnalyzer()

    changes = detector.detect_changes(old_content, new_content, file_path)
    impacts = analyzer.analyze_impact(changes, project_root)

    return ImpactReport(changes=changes, impacts=impacts)
