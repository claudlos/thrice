"""
Formal Skill Verification — Validate skill definitions for correctness.

Provides schema validation, dependency resolution, and health checks for
Hermes skill files (SKILL.md with YAML frontmatter).

A skill file looks like:
    ---
    name: my-skill
    description: What the skill does
    dependencies:
      - other-skill
    commands:
      - git
      - docker
    paths:
      - ~/.config/myapp
    ---
    # Skill content (instructions for the agent)

Usage:
    from new_files.skill_verifier import SkillSchema, DependencyResolver, SkillHealthCheck

    schema = SkillSchema()
    errors = schema.validate_frontmatter(skill)
    resolver = DependencyResolver()
    dag = resolver.build_graph(skills)
    health = SkillHealthCheck()
    report = health.check_all(skills_dir)
"""

import logging
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ─── Data types ──────────────────────────────────────────────────────────────

@dataclass
class Skill:
    """Parsed skill definition."""
    name: str
    description: str = ""
    dependencies: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    paths: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    content: str = ""
    source_path: Optional[Path] = None

    @classmethod
    def from_frontmatter(cls, frontmatter: Dict[str, Any], content: str = "", source_path: Optional[Path] = None) -> "Skill":
        """Create a Skill from parsed YAML frontmatter."""
        return cls(
            name=frontmatter.get("name", ""),
            description=frontmatter.get("description", ""),
            dependencies=frontmatter.get("dependencies", []) or [],
            commands=frontmatter.get("commands", []) or [],
            paths=frontmatter.get("paths", []) or [],
            tags=frontmatter.get("tags", []) or [],
            content=content,
            source_path=source_path,
        )


@dataclass
class Cycle:
    """A dependency cycle."""
    nodes: List[str]

    def __str__(self) -> str:
        return " -> ".join(self.nodes + [self.nodes[0]]) if self.nodes else "(empty)"


@dataclass
class HealthReport:
    """Result of a comprehensive health check."""
    valid: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    missing_deps: List[Tuple[str, str]] = field(default_factory=list)  # (skill, missing_dep)
    cycles: List[Cycle] = field(default_factory=list)
    total_skills: int = 0

    @property
    def is_healthy(self) -> bool:
        return len(self.errors) == 0 and len(self.cycles) == 0

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    def summary(self) -> str:
        lines = [
            f"Skills checked: {self.total_skills}",
            f"Valid: {len(self.valid)}",
            f"Warnings: {len(self.warnings)}",
            f"Errors: {len(self.errors)}",
            f"Missing deps: {len(self.missing_deps)}",
            f"Cycles: {len(self.cycles)}",
            f"Healthy: {self.is_healthy}",
        ]
        return "\n".join(lines)


# ─── YAML frontmatter parser (minimal, no PyYAML dependency) ─────────────────

def parse_frontmatter(text: str) -> Tuple[Dict[str, Any], str]:
    """Parse YAML-like frontmatter from a skill file.

    Handles a simple subset of YAML sufficient for skill definitions:
    key: value, key: [list], and key:\\n  - item.

    Returns:
        (frontmatter_dict, remaining_content)
    """
    if not text.startswith("---"):
        return {}, text

    lines = text.split("\n")
    end_idx = -1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break

    if end_idx == -1:
        return {}, text

    fm_lines = lines[1:end_idx]
    content = "\n".join(lines[end_idx + 1:])
    result: Dict[str, Any] = {}

    current_key = None
    current_list: Optional[List[str]] = None

    for line in fm_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # List item under current key
        if stripped.startswith("- ") and current_key:
            if current_list is None:
                current_list = []
                result[current_key] = current_list
            current_list.append(stripped[2:].strip())
            continue

        # Key: value
        match = re.match(r'^(\w[\w-]*)\s*:\s*(.*)', line)
        if match:
            key = match.group(1)
            value = match.group(2).strip()

            # Save previous list
            current_key = key
            current_list = None

            if value:
                # Inline list: [a, b, c]
                if value.startswith("[") and value.endswith("]"):
                    items = [v.strip().strip("'\"") for v in value[1:-1].split(",") if v.strip()]
                    result[key] = items
                else:
                    result[key] = value.strip("'\"")
            else:
                # Value might be a list on following lines
                result[key] = ""

    return result, content


# ─── SkillSchema ─────────────────────────────────────────────────────────────

class SkillSchema:
    """Validate skill definitions against a schema.

    Checks:
      - Required frontmatter fields (name, description)
      - Field types and formats
      - Command availability via shutil.which
      - Path pattern validity
      - Dependency references
    """

    REQUIRED_FIELDS = {"name", "description"}
    OPTIONAL_FIELDS = {"dependencies", "commands", "paths", "tags", "version", "author"}
    VALID_NAME_PATTERN = re.compile(r'^[a-zA-Z][a-zA-Z0-9_-]*$')

    def validate_frontmatter(self, skill: Skill) -> List[str]:
        """Validate the frontmatter fields of a skill.

        Returns:
            List of error messages (empty if valid).
        """
        errors: List[str] = []

        # Required fields
        if not skill.name:
            errors.append("Missing required field: name")
        elif not self.VALID_NAME_PATTERN.match(skill.name):
            errors.append(
                f"Invalid skill name '{skill.name}': must start with a letter "
                f"and contain only letters, digits, hyphens, or underscores"
            )

        if not skill.description:
            errors.append("Missing required field: description")
        elif len(skill.description) < 5:
            errors.append("Description too short (minimum 5 characters)")

        # Type validation for list fields
        for field_name in ("dependencies", "commands", "paths", "tags"):
            value = getattr(skill, field_name, None)
            if value is not None and not isinstance(value, list):
                errors.append(f"Field '{field_name}' must be a list, got {type(value).__name__}")
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if not isinstance(item, str):
                        errors.append(
                            f"Field '{field_name}[{i}]' must be a string, "
                            f"got {type(item).__name__}"
                        )

        # Name length
        if skill.name and len(skill.name) > 64:
            errors.append(f"Skill name too long ({len(skill.name)} > 64 characters)")

        return errors

    def validate_commands(self, skill: Skill) -> List[str]:
        """Check that referenced commands exist on the system.

        Returns:
            List of error messages for missing commands.
        """
        errors: List[str] = []
        for cmd in skill.commands:
            if not shutil.which(cmd):
                errors.append(f"Command not found: '{cmd}'")
        return errors

    def validate_paths(self, skill: Skill) -> List[str]:
        """Check that path patterns are valid.

        Validates syntax, not existence (paths may be templates).

        Returns:
            List of error messages for invalid paths.
        """
        errors: List[str] = []
        for path_str in skill.paths:
            # Check for obviously invalid patterns
            if not path_str:
                errors.append("Empty path pattern")
                continue

            # Null bytes
            if "\0" in path_str:
                errors.append(f"Path contains null byte: '{path_str}'")
                continue

            # Try expanding ~ and environment variables
            try:
                expanded = os.path.expanduser(os.path.expandvars(path_str))
                # Check it's a plausible path
                Path(expanded)
            except (ValueError, OSError) as e:
                errors.append(f"Invalid path pattern '{path_str}': {e}")

        return errors

    def validate_dependencies(
        self, skill: Skill, all_skills: Dict[str, Skill]
    ) -> List[str]:
        """Check dependency references and detect circular dependencies.

        Args:
            skill: The skill to validate.
            all_skills: Dict mapping skill name -> Skill for all known skills.

        Returns:
            List of error messages.
        """
        errors: List[str] = []

        for dep in skill.dependencies:
            if dep not in all_skills:
                errors.append(f"Unknown dependency: '{dep}' (not found in skills)")

            # Self-reference
            if dep == skill.name:
                errors.append(f"Self-dependency: '{skill.name}' depends on itself")

        return errors

    def validate_all(
        self, skill: Skill, all_skills: Optional[Dict[str, Skill]] = None
    ) -> List[str]:
        """Run all validations on a skill.

        Returns:
            Combined list of all error messages.
        """
        errors = []
        errors.extend(self.validate_frontmatter(skill))
        errors.extend(self.validate_commands(skill))
        errors.extend(self.validate_paths(skill))
        if all_skills is not None:
            errors.extend(self.validate_dependencies(skill, all_skills))
        return errors


# ─── DependencyResolver ─────────────────────────────────────────────────────

class DependencyResolver:
    """Build and analyze dependency graphs for skills.

    Provides cycle detection and topological sorting for load ordering.
    """

    def __init__(self):
        self._graph: Dict[str, Set[str]] = {}
        self._reverse: Dict[str, Set[str]] = {}

    def build_graph(self, skills: Dict[str, Skill]) -> Dict[str, Set[str]]:
        """Build a dependency DAG from skill definitions.

        Args:
            skills: Dict mapping skill name -> Skill.

        Returns:
            Adjacency list: {skill_name: set of dependency names}.
        """
        self._graph = {}
        self._reverse = {}

        for name, skill in skills.items():
            if name not in self._graph:
                self._graph[name] = set()
            if name not in self._reverse:
                self._reverse[name] = set()

            for dep in skill.dependencies:
                self._graph[name].add(dep)
                if dep not in self._reverse:
                    self._reverse[dep] = set()
                self._reverse[dep].add(name)
                # Ensure dep is in graph even if not in skills dict
                if dep not in self._graph:
                    self._graph[dep] = set()

        return self._graph

    def detect_cycles(self) -> List[Cycle]:
        """Detect all cycles in the dependency graph.

        Uses DFS-based cycle detection.

        Returns:
            List of Cycle objects found.
        """
        cycles: List[Cycle] = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()
        path: List[str] = []

        def dfs(node: str) -> None:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self._graph.get(node, set()):
                if neighbor not in visited:
                    dfs(neighbor)
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycle_nodes = path[cycle_start:]
                    cycles.append(Cycle(nodes=list(cycle_nodes)))

            path.pop()
            rec_stack.discard(node)

        for node in self._graph:
            if node not in visited:
                dfs(node)

        return cycles

    def topological_sort(self) -> List[str]:
        """Return a valid load order (topological sort).

        Dependencies come before dependents.

        Returns:
            Ordered list of skill names.

        Raises:
            ValueError: If the graph contains cycles.
        """
        cycles = self.detect_cycles()
        if cycles:
            cycle_str = "; ".join(str(c) for c in cycles)
            raise ValueError(f"Cannot sort: graph has cycles: {cycle_str}")

        # Kahn's algorithm
        in_degree = {node: len(deps) for node, deps in self._graph.items()}

        queue = [node for node, deg in sorted(in_degree.items()) if deg == 0]
        result: List[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            # Reduce in-degree for nodes that depend on this one
            for dependent in self._reverse.get(node, set()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._graph):
            raise ValueError("Cannot produce complete topological sort (possible cycle)")

        return result

    def check_satisfiable(
        self, skill: Skill, installed_skills: Set[str]
    ) -> bool:
        """Check if a skill's dependencies can be satisfied.

        Args:
            skill: The skill to check.
            installed_skills: Set of currently installed skill names.

        Returns:
            True if all dependencies (recursive) are satisfied.
        """
        visited: Set[str] = set()
        stack = list(skill.dependencies)

        while stack:
            dep = stack.pop()
            if dep in visited:
                continue
            visited.add(dep)

            if dep not in installed_skills:
                return False

            # Check transitive dependencies
            for trans_dep in self._graph.get(dep, set()):
                if trans_dep not in visited:
                    stack.append(trans_dep)

        return True

    def get_install_order(self, skill_name: str) -> List[str]:
        """Get the order in which to install a skill and its dependencies.

        Returns:
            List starting with deepest dependencies, ending with skill_name.
        """
        visited: Set[str] = set()
        order: List[str] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            for dep in self._graph.get(name, set()):
                visit(dep)
            order.append(name)

        visit(skill_name)
        return order


# ─── SkillHealthCheck ────────────────────────────────────────────────────────

class SkillHealthCheck:
    """Comprehensive health check for a skills directory.

    Scans all SKILL.md files, validates each, checks dependencies,
    and produces a HealthReport.
    """

    def __init__(self):
        self._schema = SkillSchema()
        self._resolver = DependencyResolver()

    def _discover_skills(self, skills_dir: Path) -> Dict[str, Skill]:
        """Find and parse all SKILL.md files in a directory tree."""
        skills: Dict[str, Skill] = {}

        if not skills_dir.exists():
            return skills

        for skill_file in skills_dir.rglob("SKILL.md"):
            try:
                text = skill_file.read_text(encoding="utf-8")
                frontmatter, content = parse_frontmatter(text)
                skill = Skill.from_frontmatter(frontmatter, content, skill_file)

                if skill.name:
                    skills[skill.name] = skill
                else:
                    # Use directory name as fallback
                    dir_name = skill_file.parent.name
                    skill.name = dir_name
                    skills[dir_name] = skill
            except Exception as e:
                logger.warning(f"Failed to parse {skill_file}: {e}")

        return skills

    def check_all(self, skills_dir: Path) -> HealthReport:
        """Run a comprehensive health check on all skills.

        Args:
            skills_dir: Path to the skills directory.

        Returns:
            HealthReport with validation results.
        """
        report = HealthReport()
        skills = self._discover_skills(skills_dir)
        report.total_skills = len(skills)

        if not skills:
            report.warnings.append(f"No skills found in {skills_dir}")
            return report

        # Validate each skill
        for name, skill in skills.items():
            fm_errors = self._schema.validate_frontmatter(skill)
            cmd_errors = self._schema.validate_commands(skill)
            path_errors = self._schema.validate_paths(skill)
            dep_errors = self._schema.validate_dependencies(skill, skills)

            all_errors = fm_errors + dep_errors
            all_warnings = cmd_errors + path_errors

            if all_errors:
                for err in all_errors:
                    report.errors.append(f"{name}: {err}")
            elif all_warnings:
                for warn in all_warnings:
                    report.warnings.append(f"{name}: {warn}")
                report.valid.append(name)
            else:
                report.valid.append(name)

        # Check for missing dependencies
        all_names = set(skills.keys())
        for name, skill in skills.items():
            for dep in skill.dependencies:
                if dep not in all_names:
                    report.missing_deps.append((name, dep))

        # Check for dependency cycles
        self._resolver.build_graph(skills)
        report.cycles = self._resolver.detect_cycles()
        if report.cycles:
            for cycle in report.cycles:
                report.errors.append(f"Dependency cycle: {cycle}")

        return report
