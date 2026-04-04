"""Tests for Formal Skill Verification — SkillSchema, DependencyResolver, SkillHealthCheck."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

# Make new-files importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "new-files"))

from skill_verifier import (
    Skill,
    Cycle,
    HealthReport,
    SkillSchema,
    DependencyResolver,
    SkillHealthCheck,
    parse_frontmatter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def schema():
    return SkillSchema()


@pytest.fixture
def resolver():
    return DependencyResolver()


@pytest.fixture
def health_check():
    return SkillHealthCheck()


@pytest.fixture
def valid_skill():
    return Skill(
        name="deploy-aws",
        description="Deploy application to AWS infrastructure",
        dependencies=["docker-setup"],
        commands=["python"],  # python should exist
        paths=["~/.aws/config"],
        tags=["devops", "aws"],
    )


@pytest.fixture
def minimal_skill():
    return Skill(name="minimal", description="A minimal skill")


@pytest.fixture
def skill_set():
    """A set of skills with dependencies."""
    return {
        "base": Skill(name="base", description="Base skill"),
        "middle": Skill(name="middle", description="Middle skill", dependencies=["base"]),
        "top": Skill(name="top", description="Top skill", dependencies=["middle"]),
        "standalone": Skill(name="standalone", description="Standalone skill"),
    }


@pytest.fixture
def cyclic_skills():
    """Skills with a dependency cycle."""
    return {
        "a": Skill(name="a", description="Skill A", dependencies=["b"]),
        "b": Skill(name="b", description="Skill B", dependencies=["c"]),
        "c": Skill(name="c", description="Skill C", dependencies=["a"]),
    }


@pytest.fixture
def tmp_skills_dir(tmp_path):
    """Create a temporary skills directory with SKILL.md files."""
    # Valid skill
    s1 = tmp_path / "skills" / "deploy-aws"
    s1.mkdir(parents=True)
    (s1 / "SKILL.md").write_text(
        "---\nname: deploy-aws\ndescription: Deploy to AWS\n"
        "commands:\n  - python\n---\nDeploy instructions.\n"
    )

    # Another valid skill
    s2 = tmp_path / "skills" / "docker-setup"
    s2.mkdir(parents=True)
    (s2 / "SKILL.md").write_text(
        "---\nname: docker-setup\ndescription: Set up Docker\n---\nDocker steps.\n"
    )

    # Skill with dependency
    s3 = tmp_path / "skills" / "full-deploy"
    s3.mkdir(parents=True)
    (s3 / "SKILL.md").write_text(
        "---\nname: full-deploy\ndescription: Full deployment\n"
        "dependencies:\n  - deploy-aws\n  - docker-setup\n---\nFull deploy steps.\n"
    )

    return tmp_path / "skills"


# ---------------------------------------------------------------------------
# parse_frontmatter tests
# ---------------------------------------------------------------------------

class TestParseFrontmatter:
    def test_basic_parse(self):
        text = "---\nname: my-skill\ndescription: A skill\n---\nContent here\n"
        fm, content = parse_frontmatter(text)
        assert fm["name"] == "my-skill"
        assert fm["description"] == "A skill"
        assert "Content here" in content

    def test_list_values(self):
        text = "---\nname: test\ndependencies:\n  - dep1\n  - dep2\n---\n"
        fm, _ = parse_frontmatter(text)
        assert fm["dependencies"] == ["dep1", "dep2"]

    def test_inline_list(self):
        text = "---\nname: test\ntags: [devops, aws]\n---\n"
        fm, _ = parse_frontmatter(text)
        assert fm["tags"] == ["devops", "aws"]

    def test_no_frontmatter(self):
        text = "Just regular content\nNo frontmatter\n"
        fm, content = parse_frontmatter(text)
        assert fm == {}
        assert content == text

    def test_empty_value(self):
        text = "---\nname: test\ndescription:\n---\n"
        fm, _ = parse_frontmatter(text)
        assert fm["description"] == ""

    def test_unclosed_frontmatter(self):
        text = "---\nname: test\nno closing\n"
        fm, content = parse_frontmatter(text)
        assert fm == {}

    def test_comments_ignored(self):
        text = "---\nname: test\n# comment\ndescription: desc\n---\n"
        fm, _ = parse_frontmatter(text)
        assert "name" in fm
        assert "description" in fm

    def test_quoted_values(self):
        text = "---\nname: 'my-skill'\ndescription: \"A description\"\n---\n"
        fm, _ = parse_frontmatter(text)
        assert fm["name"] == "my-skill"
        assert fm["description"] == "A description"


# ---------------------------------------------------------------------------
# SkillSchema tests
# ---------------------------------------------------------------------------

class TestSkillSchema:
    def test_valid_skill_no_errors(self, schema, valid_skill):
        errors = schema.validate_frontmatter(valid_skill)
        assert errors == []

    def test_missing_name(self, schema):
        skill = Skill(name="", description="Valid description")
        errors = schema.validate_frontmatter(skill)
        assert any("name" in e.lower() for e in errors)

    def test_missing_description(self, schema):
        skill = Skill(name="valid-name", description="")
        errors = schema.validate_frontmatter(skill)
        assert any("description" in e.lower() for e in errors)

    def test_invalid_name_format(self, schema):
        skill = Skill(name="123-invalid", description="Valid desc")
        errors = schema.validate_frontmatter(skill)
        assert any("invalid" in e.lower() for e in errors)

    def test_name_with_spaces(self, schema):
        skill = Skill(name="has spaces", description="Valid desc")
        errors = schema.validate_frontmatter(skill)
        assert len(errors) > 0

    def test_short_description(self, schema):
        skill = Skill(name="test", description="abc")
        errors = schema.validate_frontmatter(skill)
        assert any("short" in e.lower() for e in errors)

    def test_long_name(self, schema):
        skill = Skill(name="a" * 65, description="Valid description")
        errors = schema.validate_frontmatter(skill)
        assert any("long" in e.lower() for e in errors)

    def test_validate_commands_existing(self, schema):
        skill = Skill(name="test", description="Test skill", commands=["python3"])
        errors = schema.validate_commands(skill)
        # python3 should exist on the system
        assert len(errors) == 0

    def test_validate_commands_missing(self, schema):
        skill = Skill(name="test", description="Test skill", commands=["nonexistent_cmd_xyz"])
        errors = schema.validate_commands(skill)
        assert len(errors) == 1
        assert "nonexistent_cmd_xyz" in errors[0]

    def test_validate_paths_valid(self, schema):
        skill = Skill(name="test", description="Test skill", paths=["~/.config/app", "/tmp/test"])
        errors = schema.validate_paths(skill)
        assert errors == []

    def test_validate_paths_empty(self, schema):
        skill = Skill(name="test", description="Test skill", paths=[""])
        errors = schema.validate_paths(skill)
        assert len(errors) == 1

    def test_validate_paths_null_byte(self, schema):
        skill = Skill(name="test", description="Test skill", paths=["/tmp/\0bad"])
        errors = schema.validate_paths(skill)
        assert len(errors) == 1
        assert "null" in errors[0].lower()

    def test_validate_dependencies_valid(self, schema, skill_set):
        errors = schema.validate_dependencies(skill_set["middle"], skill_set)
        assert errors == []

    def test_validate_dependencies_missing(self, schema, skill_set):
        skill = Skill(name="orphan", description="Has missing dep", dependencies=["nonexistent"])
        errors = schema.validate_dependencies(skill, skill_set)
        assert any("nonexistent" in e for e in errors)

    def test_validate_dependencies_self_reference(self, schema, skill_set):
        skill = Skill(name="self-ref", description="Depends on itself", dependencies=["self-ref"])
        errors = schema.validate_dependencies(skill, skill_set)
        assert any("self" in e.lower() for e in errors)

    def test_validate_all_combines_checks(self, schema):
        skill = Skill(
            name="", description="", commands=["nonexistent_xyz"],
            paths=["/tmp/\0bad"],
        )
        errors = schema.validate_all(skill)
        assert len(errors) >= 2  # At least name and description errors


# ---------------------------------------------------------------------------
# DependencyResolver tests
# ---------------------------------------------------------------------------

class TestDependencyResolver:
    def test_build_graph(self, resolver, skill_set):
        graph = resolver.build_graph(skill_set)
        assert "base" in graph
        assert "middle" in graph
        assert "base" in graph["middle"]  # middle depends on base

    def test_no_cycles_in_dag(self, resolver, skill_set):
        resolver.build_graph(skill_set)
        cycles = resolver.detect_cycles()
        assert cycles == []

    def test_detect_cycles(self, resolver, cyclic_skills):
        resolver.build_graph(cyclic_skills)
        cycles = resolver.detect_cycles()
        assert len(cycles) > 0
        # Verify cycle contains the expected nodes
        cycle_nodes = set()
        for c in cycles:
            cycle_nodes.update(c.nodes)
        assert "a" in cycle_nodes or "b" in cycle_nodes or "c" in cycle_nodes

    def test_topological_sort_valid(self, resolver, skill_set):
        resolver.build_graph(skill_set)
        order = resolver.topological_sort()
        assert len(order) == len(skill_set)

        # base must come before middle, middle before top
        base_idx = order.index("base")
        middle_idx = order.index("middle")
        top_idx = order.index("top")
        assert base_idx < middle_idx < top_idx

    def test_topological_sort_raises_on_cycles(self, resolver, cyclic_skills):
        resolver.build_graph(cyclic_skills)
        with pytest.raises(ValueError, match="cycle"):
            resolver.topological_sort()

    def test_check_satisfiable_all_present(self, resolver, skill_set):
        resolver.build_graph(skill_set)
        installed = {"base", "middle", "top", "standalone"}
        assert resolver.check_satisfiable(skill_set["top"], installed) is True

    def test_check_satisfiable_missing_dep(self, resolver, skill_set):
        resolver.build_graph(skill_set)
        installed = {"top"}  # Missing base and middle
        assert resolver.check_satisfiable(skill_set["top"], installed) is False

    def test_check_satisfiable_no_deps(self, resolver, skill_set):
        resolver.build_graph(skill_set)
        installed = set()
        assert resolver.check_satisfiable(skill_set["base"], installed) is True

    def test_get_install_order(self, resolver, skill_set):
        resolver.build_graph(skill_set)
        order = resolver.get_install_order("top")
        assert order[-1] == "top"
        assert "base" in order
        assert "middle" in order
        assert order.index("base") < order.index("middle")

    def test_empty_graph(self, resolver):
        graph = resolver.build_graph({})
        assert graph == {}
        assert resolver.detect_cycles() == []
        assert resolver.topological_sort() == []


# ---------------------------------------------------------------------------
# Cycle tests
# ---------------------------------------------------------------------------

class TestCycle:
    def test_str_representation(self):
        c = Cycle(nodes=["a", "b", "c"])
        assert str(c) == "a -> b -> c -> a"

    def test_empty_cycle(self):
        c = Cycle(nodes=[])
        assert str(c) == "(empty)"


# ---------------------------------------------------------------------------
# HealthReport tests
# ---------------------------------------------------------------------------

class TestHealthReport:
    def test_healthy_report(self):
        report = HealthReport(valid=["skill-a", "skill-b"], total_skills=2)
        assert report.is_healthy
        assert report.error_count == 0

    def test_unhealthy_with_errors(self):
        report = HealthReport(errors=["Something broke"], total_skills=1)
        assert not report.is_healthy

    def test_unhealthy_with_cycles(self):
        report = HealthReport(cycles=[Cycle(nodes=["a", "b"])], total_skills=2)
        assert not report.is_healthy

    def test_summary(self):
        report = HealthReport(
            valid=["a"], warnings=["w1"], errors=["e1"],
            missing_deps=[("x", "y")], total_skills=3,
        )
        summary = report.summary()
        assert "Skills checked: 3" in summary
        assert "Errors: 1" in summary


# ---------------------------------------------------------------------------
# SkillHealthCheck tests
# ---------------------------------------------------------------------------

class TestSkillHealthCheck:
    def test_check_all_valid_dir(self, health_check, tmp_skills_dir):
        report = health_check.check_all(tmp_skills_dir)
        assert report.total_skills == 3
        assert report.is_healthy

    def test_check_all_empty_dir(self, health_check, tmp_path):
        empty_dir = tmp_path / "empty_skills"
        empty_dir.mkdir()
        report = health_check.check_all(empty_dir)
        assert report.total_skills == 0
        assert len(report.warnings) > 0

    def test_check_all_nonexistent_dir(self, health_check, tmp_path):
        report = health_check.check_all(tmp_path / "nonexistent")
        assert report.total_skills == 0

    def test_check_all_detects_missing_deps(self, health_check, tmp_path):
        skill_dir = tmp_path / "skills" / "orphan"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: orphan\ndescription: Has missing dep\n"
            "dependencies:\n  - nonexistent\n---\nContent.\n"
        )
        report = health_check.check_all(tmp_path / "skills")
        assert len(report.missing_deps) > 0
        assert report.missing_deps[0] == ("orphan", "nonexistent")

    def test_check_all_detects_cycles(self, health_check, tmp_path):
        # Create cyclic skills
        for name, dep in [("skill-a", "skill-b"), ("skill-b", "skill-a")]:
            d = tmp_path / "skills" / name
            d.mkdir(parents=True)
            (d / "SKILL.md").write_text(
                f"---\nname: {name}\ndescription: Cyclic skill\n"
                f"dependencies:\n  - {dep}\n---\nContent.\n"
            )
        report = health_check.check_all(tmp_path / "skills")
        assert len(report.cycles) > 0
        assert not report.is_healthy

    def test_check_all_invalid_frontmatter(self, health_check, tmp_path):
        skill_dir = tmp_path / "skills" / "bad"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: 123bad\ndescription: ok desc\n---\n"
        )
        report = health_check.check_all(tmp_path / "skills")
        assert len(report.errors) > 0

    def test_discover_skills_fallback_name(self, health_check, tmp_path):
        skill_dir = tmp_path / "skills" / "my-skill"
        skill_dir.mkdir(parents=True)
        # No name in frontmatter
        (skill_dir / "SKILL.md").write_text(
            "---\ndescription: A skill without a name\n---\nContent.\n"
        )
        report = health_check.check_all(tmp_path / "skills")
        assert report.total_skills == 1
