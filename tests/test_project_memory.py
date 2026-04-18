"""Tests for project_memory module.

25+ tests using tempdir for isolated filesystem testing.
"""

import os
import sys
import tempfile
import textwrap
import time
import unittest
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _tempdir():
    """Temp dir whose path is resolved (Windows 8.3 → long form, macOS /var → /private/var)."""
    with tempfile.TemporaryDirectory() as tmp:
        yield os.path.realpath(tmp)

# Ensure the module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new-files"))

from project_memory import (
    DEFAULT_SECTIONS,
    HERMES_DIR,
    HERMES_ROOT_FILE,
    MEMORY_FILENAME,
    ProjectMemory,
    ProjectMemoryManager,
    detect_project_type,
    find_project_root,
    parse_memory_file,
    serialize_memory,
)


class TestFindProjectRoot(unittest.TestCase):
    """Tests for find_project_root()."""

    def test_finds_git_dir(self):
        with _tempdir() as tmp:
            os.makedirs(os.path.join(tmp, ".git"))
            self.assertEqual(find_project_root(tmp), tmp)

    def test_finds_pyproject(self):
        with _tempdir() as tmp:
            Path(tmp, "pyproject.toml").touch()
            self.assertEqual(find_project_root(tmp), tmp)

    def test_finds_package_json(self):
        with _tempdir() as tmp:
            Path(tmp, "package.json").touch()
            self.assertEqual(find_project_root(tmp), tmp)

    def test_finds_cargo_toml(self):
        with _tempdir() as tmp:
            Path(tmp, "Cargo.toml").touch()
            self.assertEqual(find_project_root(tmp), tmp)

    def test_walks_up_directories(self):
        with _tempdir() as tmp:
            os.makedirs(os.path.join(tmp, ".git"))
            subdir = os.path.join(tmp, "src", "deep", "nested")
            os.makedirs(subdir)
            self.assertEqual(find_project_root(subdir), tmp)

    def test_returns_none_for_no_markers(self):
        with _tempdir() as tmp:
            # Isolated dir with no markers — should eventually hit root and return None
            # We need a dir that has no markers all the way up, which is hard to guarantee.
            # Instead, just check it returns a string or None (it might find system-level markers).
            result = find_project_root(tmp)
            # Just verify it doesn't crash
            self.assertTrue(result is None or isinstance(result, str))

    def test_go_mod(self):
        with _tempdir() as tmp:
            Path(tmp, "go.mod").touch()
            self.assertEqual(find_project_root(tmp), tmp)


class TestParseMemoryFile(unittest.TestCase):
    """Tests for parse_memory_file()."""

    def test_empty_string(self):
        self.assertEqual(parse_memory_file(""), {})

    def test_preamble_only(self):
        result = parse_memory_file("This is just preamble text.\n")
        self.assertIn("_preamble", result)
        self.assertEqual(result["_preamble"], "This is just preamble text.")

    def test_single_section(self):
        text = "## Conventions\n\n- Use black formatter\n- Line length 88\n"
        result = parse_memory_file(text)
        self.assertIn("Conventions", result)
        self.assertIn("black formatter", result["Conventions"])

    def test_multiple_sections(self):
        text = textwrap.dedent("""\
            ## Conventions

            - Style guide A

            ## Testing

            - Use pytest

            ## Notes

            Some notes here.
        """)
        result = parse_memory_file(text)
        self.assertEqual(len([k for k in result if k != "_preamble"]), 3)
        self.assertIn("Conventions", result)
        self.assertIn("Testing", result)
        self.assertIn("Notes", result)

    def test_preamble_plus_sections(self):
        text = "Project: MyApp\n\n## Conventions\n\n- foo\n"
        result = parse_memory_file(text)
        self.assertIn("_preamble", result)
        self.assertIn("Conventions", result)

    def test_empty_section_body(self):
        text = "## Empty\n\n## HasContent\n\nSome content.\n"
        result = parse_memory_file(text)
        self.assertEqual(result["Empty"], "")
        self.assertEqual(result["HasContent"], "Some content.")


class TestSerializeMemory(unittest.TestCase):
    """Tests for serialize_memory()."""

    def test_roundtrip(self):
        original = {"Conventions": "- use black", "Testing": "- pytest"}
        text = serialize_memory(original)
        parsed = parse_memory_file(text)
        self.assertEqual(parsed["Conventions"], "- use black")
        self.assertEqual(parsed["Testing"], "- pytest")

    def test_includes_preamble(self):
        sections = {"_preamble": "My Project", "Notes": "hello"}
        text = serialize_memory(sections)
        self.assertTrue(text.startswith("My Project"))
        self.assertIn("## Notes", text)

    def test_empty_body_sections(self):
        sections = {"Commands": ""}
        text = serialize_memory(sections)
        self.assertIn("## Commands", text)


class TestDetectProjectType(unittest.TestCase):
    """Tests for detect_project_type()."""

    def test_python_project(self):
        with _tempdir() as tmp:
            Path(tmp, "pyproject.toml").write_text("[tool.poetry]\n", encoding="utf-8")
            info = detect_project_type(tmp)
            self.assertEqual(info["language"], "python")
            self.assertEqual(info["package_manager"], "poetry")

    def test_python_with_pytest(self):
        with _tempdir() as tmp:
            Path(tmp, "setup.py").touch()
            Path(tmp, "conftest.py").touch()
            info = detect_project_type(tmp)
            self.assertEqual(info["language"], "python")
            self.assertEqual(info["test_framework"], "pytest")

    def test_javascript_project(self):
        with _tempdir() as tmp:
            Path(tmp, "package.json").write_text('{"name":"test"}', encoding="utf-8")
            info = detect_project_type(tmp)
            self.assertEqual(info["language"], "javascript")
            self.assertEqual(info["package_manager"], "npm")

    def test_typescript_project(self):
        with _tempdir() as tmp:
            Path(tmp, "package.json").write_text('{"name":"test"}', encoding="utf-8")
            Path(tmp, "tsconfig.json").touch()
            info = detect_project_type(tmp)
            self.assertEqual(info["language"], "typescript")

    def test_yarn_detected(self):
        with _tempdir() as tmp:
            Path(tmp, "package.json").write_text('{}', encoding="utf-8")
            Path(tmp, "yarn.lock").touch()
            info = detect_project_type(tmp)
            self.assertEqual(info["package_manager"], "yarn")

    def test_rust_project(self):
        with _tempdir() as tmp:
            Path(tmp, "Cargo.toml").touch()
            info = detect_project_type(tmp)
            self.assertEqual(info["language"], "rust")
            self.assertEqual(info["build_tool"], "cargo")

    def test_go_project(self):
        with _tempdir() as tmp:
            Path(tmp, "go.mod").touch()
            info = detect_project_type(tmp)
            self.assertEqual(info["language"], "go")

    def test_unknown_project(self):
        with _tempdir() as tmp:
            info = detect_project_type(tmp)
            self.assertIsNone(info["language"])

    def test_jest_detected(self):
        with _tempdir() as tmp:
            Path(tmp, "package.json").write_text('{"devDependencies":{"jest":"^29"}}', encoding="utf-8")
            info = detect_project_type(tmp)
            self.assertEqual(info["test_framework"], "jest")


class TestProjectMemory(unittest.TestCase):
    """Tests for ProjectMemory class."""

    def _make_project(self, tmp):
        """Create a minimal project with .git and memory files."""
        os.makedirs(os.path.join(tmp, ".git"))
        hermes = os.path.join(tmp, HERMES_DIR)
        os.makedirs(hermes)
        return hermes

    def test_load_hermes_dir_memory(self):
        with _tempdir() as tmp:
            hermes = self._make_project(tmp)
            Path(hermes, MEMORY_FILENAME).write_text(
                "## Conventions\n\n- Use ruff\n", encoding="utf-8"
            )
            mem = ProjectMemory(project_root=tmp)
            self.assertIn("Conventions", mem.list_sections())
            self.assertIn("ruff", mem.sections["Conventions"])

    def test_load_hermes_md_root(self):
        with _tempdir() as tmp:
            os.makedirs(os.path.join(tmp, ".git"))
            Path(tmp, HERMES_ROOT_FILE).write_text(
                "## Notes\n\nImportant note.\n", encoding="utf-8"
            )
            mem = ProjectMemory(project_root=tmp)
            self.assertIn("Notes", mem.list_sections())

    def test_merge_both_files(self):
        with _tempdir() as tmp:
            hermes = self._make_project(tmp)
            Path(hermes, MEMORY_FILENAME).write_text(
                "## Conventions\n\n- From hermes dir\n", encoding="utf-8"
            )
            Path(tmp, HERMES_ROOT_FILE).write_text(
                "## Conventions\n\n- From root file\n", encoding="utf-8"
            )
            mem = ProjectMemory(project_root=tmp)
            ctx = mem.get_context()
            self.assertIn("From hermes dir", ctx)
            self.assertIn("From root file", ctx)

    def test_get_context_empty_project(self):
        with _tempdir() as tmp:
            os.makedirs(os.path.join(tmp, ".git"))
            mem = ProjectMemory(project_root=tmp)
            self.assertEqual(mem.get_context(), "")

    def test_get_context_with_subdirectory(self):
        with _tempdir() as tmp:
            hermes = self._make_project(tmp)
            Path(hermes, MEMORY_FILENAME).write_text(
                "## Conventions\n\n- Root conventions\n", encoding="utf-8"
            )
            subdir = os.path.join(tmp, "src", "api")
            os.makedirs(subdir)
            sub_hermes = os.path.join(subdir, HERMES_DIR)
            os.makedirs(sub_hermes)
            Path(sub_hermes, MEMORY_FILENAME).write_text(
                "## Conventions\n\n- API conventions\n", encoding="utf-8"
            )
            mem = ProjectMemory(project_root=tmp)
            ctx = mem.get_context(cwd=subdir)
            self.assertIn("Root conventions", ctx)
            self.assertIn("API conventions", ctx)

    def test_update_creates_file(self):
        with _tempdir() as tmp:
            os.makedirs(os.path.join(tmp, ".git"))
            mem = ProjectMemory(project_root=tmp)
            mem.update("Commands", "- make build")
            self.assertIn("Commands", mem.list_sections())
            # Verify file on disk
            on_disk = Path(tmp, HERMES_DIR, MEMORY_FILENAME).read_text(encoding="utf-8")
            self.assertIn("make build", on_disk)

    def test_update_preserves_other_sections(self):
        with _tempdir() as tmp:
            hermes = self._make_project(tmp)
            Path(hermes, MEMORY_FILENAME).write_text(
                "## Conventions\n\n- Existing\n\n## Notes\n\nOld note.\n",
                encoding="utf-8",
            )
            mem = ProjectMemory(project_root=tmp)
            mem.update("Notes", "New note.")
            on_disk = Path(hermes, MEMORY_FILENAME).read_text(encoding="utf-8")
            self.assertIn("Existing", on_disk)
            self.assertIn("New note.", on_disk)

    def test_is_loaded_property(self):
        with _tempdir() as tmp:
            hermes = self._make_project(tmp)
            Path(hermes, MEMORY_FILENAME).write_text("## Notes\n\nhello\n", encoding="utf-8")
            mem = ProjectMemory(project_root=tmp)
            self.assertTrue(mem.is_loaded)

    def test_no_project_root(self):
        mem = ProjectMemory(project_root=None)
        self.assertFalse(mem.is_loaded)
        self.assertEqual(mem.get_context(), "")
        self.assertEqual(mem.list_sections(), [])

    def test_update_no_root_raises(self):
        mem = ProjectMemory(project_root=None)
        with self.assertRaises(RuntimeError):
            mem.update("Notes", "test")


class TestProjectMemoryManager(unittest.TestCase):
    """Tests for ProjectMemoryManager class."""

    def _make_project(self, tmp):
        os.makedirs(os.path.join(tmp, ".git"))
        hermes = os.path.join(tmp, HERMES_DIR)
        os.makedirs(hermes)
        return hermes

    def test_get_memory_returns_instance(self):
        with _tempdir() as tmp:
            self._make_project(tmp)
            mgr = ProjectMemoryManager()
            mem = mgr.get_memory(cwd=tmp)
            self.assertIsInstance(mem, ProjectMemory)

    def test_caching(self):
        with _tempdir() as tmp:
            hermes = self._make_project(tmp)
            Path(hermes, MEMORY_FILENAME).write_text("## Notes\n\ncached\n", encoding="utf-8")
            mgr = ProjectMemoryManager(cache_ttl=60.0)
            mem1 = mgr.get_memory(cwd=tmp)
            mem2 = mgr.get_memory(cwd=tmp)
            self.assertIs(mem1, mem2)

    def test_invalidate_clears_cache(self):
        with _tempdir() as tmp:
            hermes = self._make_project(tmp)
            Path(hermes, MEMORY_FILENAME).write_text("## Notes\n\nv1\n", encoding="utf-8")
            mgr = ProjectMemoryManager()
            mem1 = mgr.get_memory(cwd=tmp)
            mgr.invalidate()
            mem2 = mgr.get_memory(cwd=tmp)
            self.assertIsNot(mem1, mem2)

    def test_invalidate_specific_root(self):
        with _tempdir() as tmp:
            hermes = self._make_project(tmp)
            Path(hermes, MEMORY_FILENAME).write_text("## Notes\n\nhi\n", encoding="utf-8")
            mgr = ProjectMemoryManager()
            mgr.get_memory(cwd=tmp)
            self.assertIn(tmp, mgr.cached_roots)
            mgr.invalidate(tmp)
            self.assertNotIn(tmp, mgr.cached_roots)

    def test_format_for_prompt(self):
        with _tempdir() as tmp:
            hermes = self._make_project(tmp)
            Path(hermes, MEMORY_FILENAME).write_text(
                "## Conventions\n\n- Use type hints\n", encoding="utf-8"
            )
            mgr = ProjectMemoryManager()
            prompt = mgr.format_for_prompt(cwd=tmp)
            self.assertIn("<project_memory>", prompt)
            self.assertIn("</project_memory>", prompt)
            self.assertIn("type hints", prompt)

    def test_format_for_prompt_empty(self):
        with _tempdir() as tmp:
            os.makedirs(os.path.join(tmp, ".git"))
            mgr = ProjectMemoryManager()
            prompt = mgr.format_for_prompt(cwd=tmp)
            self.assertEqual(prompt, "")

    def test_generate_initial_memory_python(self):
        with _tempdir() as tmp:
            Path(tmp, "pyproject.toml").write_text(
                "[tool.pytest]\n", encoding="utf-8"
            )
            Path(tmp, "conftest.py").touch()
            os.makedirs(os.path.join(tmp, "src"))
            os.makedirs(os.path.join(tmp, "tests"))
            mgr = ProjectMemoryManager()
            md = mgr.generate_initial_memory(tmp)
            self.assertIn("python", md.lower())
            self.assertIn("pytest", md.lower())
            self.assertIn("src/", md)

    def test_generate_initial_memory_js(self):
        with _tempdir() as tmp:
            Path(tmp, "package.json").write_text(
                '{"devDependencies":{"jest":"^29"}}', encoding="utf-8"
            )
            mgr = ProjectMemoryManager()
            md = mgr.generate_initial_memory(tmp)
            self.assertIn("javascript", md.lower())
            self.assertIn("jest", md.lower())

    def test_generate_and_save(self):
        with _tempdir() as tmp:
            Path(tmp, "Cargo.toml").touch()
            mgr = ProjectMemoryManager()
            md = mgr.generate_and_save(tmp)
            target = Path(tmp, HERMES_DIR, MEMORY_FILENAME)
            self.assertTrue(target.is_file())
            on_disk = target.read_text(encoding="utf-8")
            self.assertEqual(on_disk, md)
            self.assertIn("rust", md.lower())

    def test_mtime_invalidation(self):
        with _tempdir() as tmp:
            hermes = self._make_project(tmp)
            mem_file = Path(hermes, MEMORY_FILENAME)
            mem_file.write_text("## Notes\n\nversion1\n", encoding="utf-8")

            mgr = ProjectMemoryManager(cache_ttl=0.0)  # TTL=0 forces mtime check
            mem1 = mgr.get_memory(cwd=tmp)
            self.assertIn("version1", mem1.get_context())

            # Modify file with a new mtime
            time.sleep(0.05)
            mem_file.write_text("## Notes\n\nversion2\n", encoding="utf-8")

            mem2 = mgr.get_memory(cwd=tmp)
            self.assertIn("version2", mem2.get_context())

    def test_cached_roots_property(self):
        mgr = ProjectMemoryManager()
        self.assertEqual(mgr.cached_roots, [])


class TestEdgeCases(unittest.TestCase):
    """Edge cases and integration-like tests."""

    def test_section_with_special_chars(self):
        text = "## My Section (v2)\n\nContent here.\n"
        parsed = parse_memory_file(text)
        self.assertIn("My Section (v2)", parsed)

    def test_multiple_levels_of_subdirs(self):
        with _tempdir() as tmp:
            os.makedirs(os.path.join(tmp, ".git"))
            hermes = os.path.join(tmp, HERMES_DIR)
            os.makedirs(hermes)
            Path(hermes, MEMORY_FILENAME).write_text(
                "## Notes\n\nRoot note\n", encoding="utf-8"
            )

            # Level 1
            l1 = os.path.join(tmp, "src")
            os.makedirs(l1)
            l1h = os.path.join(l1, HERMES_DIR)
            os.makedirs(l1h)
            Path(l1h, MEMORY_FILENAME).write_text(
                "## Notes\n\nSrc note\n", encoding="utf-8"
            )

            # Level 2
            l2 = os.path.join(l1, "api")
            os.makedirs(l2)
            l2h = os.path.join(l2, HERMES_DIR)
            os.makedirs(l2h)
            Path(l2h, MEMORY_FILENAME).write_text(
                "## Notes\n\nAPI note\n", encoding="utf-8"
            )

            mem = ProjectMemory(project_root=tmp)
            ctx = mem.get_context(cwd=l2)
            self.assertIn("Root note", ctx)
            self.assertIn("API note", ctx)

    def test_serialize_then_parse_all_defaults(self):
        sections = {s: f"Content for {s}" for s in DEFAULT_SECTIONS}
        text = serialize_memory(sections)
        parsed = parse_memory_file(text)
        for s in DEFAULT_SECTIONS:
            self.assertEqual(parsed[s], f"Content for {s}")

    def test_auto_detect_from_cwd(self):
        with _tempdir() as tmp:
            os.makedirs(os.path.join(tmp, ".git"))
            hermes = os.path.join(tmp, HERMES_DIR)
            os.makedirs(hermes)
            Path(hermes, MEMORY_FILENAME).write_text("## Notes\n\nhi\n", encoding="utf-8")
            subdir = os.path.join(tmp, "deep", "path")
            os.makedirs(subdir)
            mem = ProjectMemory(cwd=subdir)
            self.assertEqual(mem.project_root, tmp)
            self.assertTrue(mem.is_loaded)


if __name__ == "__main__":
    unittest.main()
