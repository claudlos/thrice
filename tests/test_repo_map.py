"""Tests for repo_map module."""

import os
import sys
import tempfile
import textwrap
import time
import threading
import unittest

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "new-files"))

from repo_map import (
    RepoMap,
    RepoMapCache,
    parse_file,
    get_repo_map,
    invalidate_repo_map,
    Symbol,
    FileInfo,
    _parse_python,
    _parse_js_ts,
    _parse_go,
    _parse_rust,
)


class TestPythonParser(unittest.TestCase):
    """Test Python AST-based parsing."""

    def test_parse_function(self):
        source = textwrap.dedent("""\
            def hello(name, greeting="hi"):
                return f"{greeting} {name}"
        """)
        info = _parse_python(source, "test.py")
        self.assertEqual(len(info.symbols), 1)
        sym = info.symbols[0]
        self.assertEqual(sym.name, "hello")
        self.assertEqual(sym.kind, "function")
        self.assertIn("name", sym.signature)
        self.assertTrue(sym.is_exported)

    def test_parse_class(self):
        source = textwrap.dedent("""\
            class MyClass(BaseClass):
                def method(self, x):
                    pass

                def _private(self):
                    pass
        """)
        info = _parse_python(source, "test.py")
        names = {s.name for s in info.symbols}
        self.assertIn("MyClass", names)
        self.assertIn("method", names)
        self.assertIn("_private", names)

        cls = [s for s in info.symbols if s.name == "MyClass"][0]
        self.assertEqual(cls.kind, "class")
        self.assertIn("BaseClass", cls.signature)

        priv = [s for s in info.symbols if s.name == "_private"][0]
        self.assertFalse(priv.is_exported)

    def test_parse_imports(self):
        source = textwrap.dedent("""\
            import os
            from pathlib import Path
            from collections import defaultdict as dd
        """)
        info = _parse_python(source, "test.py")
        self.assertIn("os", info.imports)
        self.assertIn("Path", info.imports)
        self.assertIn("dd", info.imports)

    def test_parse_async_function(self):
        source = "async def fetch(url):\n    pass\n"
        info = _parse_python(source, "test.py")
        self.assertEqual(len(info.symbols), 1)
        self.assertEqual(info.symbols[0].name, "fetch")

    def test_syntax_error_handled(self):
        source = "def broken(\n"
        info = _parse_python(source, "test.py")
        self.assertEqual(len(info.symbols), 0)


class TestJSParser(unittest.TestCase):
    """Test JS/TS regex-based parsing."""

    def test_parse_function(self):
        source = "function processData(input, options) {\n}\n"
        info = _parse_js_ts(source, "test.js", "javascript")
        self.assertTrue(any(s.name == "processData" for s in info.symbols))

    def test_parse_exported_class(self):
        source = "export class Router extends BaseRouter {\n}\n"
        info = _parse_js_ts(source, "test.ts", "typescript")
        sym = [s for s in info.symbols if s.name == "Router"]
        self.assertTrue(len(sym) > 0)
        self.assertTrue(sym[0].is_exported)
        self.assertIn("extends", sym[0].signature)

    def test_parse_arrow_function(self):
        source = "export const handler = async (req, res) => {\n}\n"
        info = _parse_js_ts(source, "test.js", "javascript")
        sym = [s for s in info.symbols if s.name == "handler"]
        self.assertTrue(len(sym) > 0)

    def test_parse_imports(self):
        source = 'import { useState, useEffect } from "react";\n'
        info = _parse_js_ts(source, "test.js", "javascript")
        self.assertIn("useState", info.imports)
        self.assertIn("useEffect", info.imports)


class TestGoParser(unittest.TestCase):
    """Test Go regex-based parsing."""

    def test_parse_function(self):
        source = "func HandleRequest(w http.ResponseWriter, r *http.Request) {\n}\n"
        info = _parse_go(source, "main.go")
        sym = [s for s in info.symbols if s.name == "HandleRequest"]
        self.assertTrue(len(sym) > 0)
        self.assertTrue(sym[0].is_exported)  # uppercase = exported in Go

    def test_parse_struct(self):
        source = "type Config struct {\n\tHost string\n}\n"
        info = _parse_go(source, "config.go")
        sym = [s for s in info.symbols if s.name == "Config"]
        self.assertTrue(len(sym) > 0)
        self.assertEqual(sym[0].kind, "struct")

    def test_parse_method(self):
        source = "func (s *Server) Start(port int) error {\n}\n"
        info = _parse_go(source, "server.go")
        sym = [s for s in info.symbols if s.name == "Start"]
        self.assertTrue(len(sym) > 0)
        self.assertEqual(sym[0].kind, "method")

    def test_unexported(self):
        source = "func helper() {\n}\n"
        info = _parse_go(source, "util.go")
        sym = [s for s in info.symbols if s.name == "helper"]
        self.assertTrue(len(sym) > 0)
        self.assertFalse(sym[0].is_exported)


class TestRustParser(unittest.TestCase):
    """Test Rust regex-based parsing."""

    def test_parse_function(self):
        source = "pub fn process(data: &[u8]) -> Result<()> {\n}\n"
        info = _parse_rust(source, "lib.rs")
        sym = [s for s in info.symbols if s.name == "process"]
        self.assertTrue(len(sym) > 0)
        self.assertTrue(sym[0].is_exported)

    def test_parse_struct(self):
        source = "pub struct Config {\n    host: String,\n}\n"
        info = _parse_rust(source, "config.rs")
        sym = [s for s in info.symbols if s.name == "Config"]
        self.assertTrue(len(sym) > 0)
        self.assertEqual(sym[0].kind, "struct")

    def test_parse_impl(self):
        source = "impl Config {\n    fn new() -> Self { }\n}\n"
        info = _parse_rust(source, "config.rs")
        names = {s.name for s in info.symbols}
        self.assertIn("Config", names)
        self.assertIn("new", names)

    def test_private_function(self):
        source = "fn internal_helper(x: i32) -> i32 {\n}\n"
        info = _parse_rust(source, "lib.rs")
        sym = [s for s in info.symbols if s.name == "internal_helper"]
        self.assertTrue(len(sym) > 0)
        self.assertFalse(sym[0].is_exported)


class TestRepoMap(unittest.TestCase):
    """Test the full RepoMap with a temp directory."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create sample Python files
        self._write("models.py", textwrap.dedent("""\
            class User:
                def __init__(self, name, email):
                    self.name = name
                    self.email = email

                def validate(self):
                    return bool(self.email)

            class Post:
                def __init__(self, title, author):
                    self.title = title
                    self.author = author

                def publish(self):
                    if self.author.validate():
                        return True
                    return False
        """))

        self._write("views.py", textwrap.dedent("""\
            from models import User, Post

            def get_user(user_id):
                return User("test", "test@example.com")

            def create_post(title, user_id):
                user = get_user(user_id)
                post = Post(title, user)
                post.publish()
                return post

            def list_posts():
                return []
        """))

        self._write("utils.py", textwrap.dedent("""\
            def format_date(dt):
                return dt.strftime("%Y-%m-%d")

            def _internal_helper():
                pass
        """))

        # Create a subdirectory
        os.makedirs(os.path.join(self.tmpdir, "lib"), exist_ok=True)
        self._write("lib/helpers.py", textwrap.dedent("""\
            from models import User

            def find_user(name):
                return User(name, f"{name}@example.com")
        """))

    def _write(self, relpath, content):
        fpath = os.path.join(self.tmpdir, relpath)
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        with open(fpath, "w") as f:
            f.write(content)

    def test_scan_finds_files(self):
        rm = RepoMap(self.tmpdir).scan()
        self.assertEqual(len(rm.files), 4)

    def test_scan_finds_symbols(self):
        rm = RepoMap(self.tmpdir).scan()
        names = {s.name for s in rm.symbols.values()}
        self.assertIn("User", names)
        self.assertIn("Post", names)
        self.assertIn("get_user", names)
        self.assertIn("create_post", names)
        self.assertIn("format_date", names)
        self.assertIn("find_user", names)

    def test_references_counted(self):
        rm = RepoMap(self.tmpdir).scan()
        # User is referenced in views.py and lib/helpers.py
        user_syms = [s for s in rm.symbols.values() if s.name == "User"]
        self.assertTrue(len(user_syms) > 0)
        user = user_syms[0]
        self.assertGreater(user.references, 0)

    def test_generate_map(self):
        rm = RepoMap(self.tmpdir).scan()
        map_text = rm.generate_map(max_tokens=2000)
        self.assertIn("User", map_text)
        self.assertIn("Post", map_text)
        self.assertIn(".py:", map_text)
        # Should be multi-line
        lines = map_text.strip().split("\n")
        self.assertGreater(len(lines), 1)

    def test_generate_map_token_limit(self):
        rm = RepoMap(self.tmpdir).scan()
        # Very small limit
        map_text = rm.generate_map(max_tokens=50)
        self.assertLess(len(map_text), 250)

    def test_ranking_exported_higher(self):
        rm = RepoMap(self.tmpdir).scan()
        top = rm.get_top_symbols(5)
        top_names = [s.name for s, _ in top]
        # _internal_helper should NOT be in top 5
        self.assertNotIn("_internal_helper", top_names)

    def test_refresh_incremental(self):
        rm = RepoMap(self.tmpdir).scan()
        old_count = len(rm.symbols)

        # Add a new file
        self._write("new_module.py", "def brand_new_func(x):\n    return x\n")
        rm.refresh(changed_files=[os.path.join(self.tmpdir, "new_module.py")])

        self.assertGreater(len(rm.symbols), old_count)
        names = {s.name for s in rm.symbols.values()}
        self.assertIn("brand_new_func", names)

    def test_refresh_detects_mtime(self):
        rm = RepoMap(self.tmpdir).scan()

        # Modify a file (ensure mtime changes)
        time.sleep(0.05)
        self._write("utils.py", textwrap.dedent("""\
            def format_date(dt):
                return dt.strftime("%Y-%m-%d")

            def new_util():
                pass
        """))

        rm.refresh()  # auto-detect changes
        names = {s.name for s in rm.symbols.values()}
        self.assertIn("new_util", names)


class TestRepoMapCache(unittest.TestCase):
    """Test thread-safe caching."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        with open(os.path.join(self.tmpdir, "main.py"), "w") as f:
            f.write("def main():\n    pass\n")

    def test_cache_returns_same(self):
        cache = RepoMapCache(ttl_seconds=60)
        map1 = cache.get_map(self.tmpdir)
        map2 = cache.get_map(self.tmpdir)
        self.assertEqual(map1, map2)

    def test_cache_invalidation(self):
        cache = RepoMapCache(ttl_seconds=60)
        map1 = cache.get_map(self.tmpdir)
        cache.invalidate(self.tmpdir)
        # After invalidation, should rebuild
        map2 = cache.get_map(self.tmpdir)
        self.assertEqual(map1, map2)  # same content, just rebuilt

    def test_force_refresh(self):
        cache = RepoMapCache(ttl_seconds=60)
        map1 = cache.get_map(self.tmpdir)
        map2 = cache.get_map(self.tmpdir, force_refresh=True)
        self.assertEqual(map1, map2)

    def test_thread_safety(self):
        cache = RepoMapCache(ttl_seconds=60)
        results = []
        errors = []

        def worker():
            try:
                result = cache.get_map(self.tmpdir)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        self.assertEqual(len(errors), 0, f"Errors: {errors}")
        self.assertEqual(len(results), 10)
        # All should return the same map
        self.assertTrue(all(r == results[0] for r in results))


class TestPublicAPI(unittest.TestCase):
    """Test the public get_repo_map function."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        with open(os.path.join(self.tmpdir, "app.py"), "w") as f:
            f.write(textwrap.dedent("""\
                class Application:
                    def run(self):
                        pass

                def create_app():
                    return Application()
            """))
        invalidate_repo_map()  # clear global cache

    def test_get_repo_map(self):
        result = get_repo_map(self.tmpdir)
        self.assertIn("Application", result)
        self.assertIn("create_app", result)

    def test_get_repo_map_token_limit(self):
        result = get_repo_map(self.tmpdir, max_tokens=50)
        self.assertLess(len(result), 300)


class TestParseFile(unittest.TestCase):
    """Test the parse_file dispatcher."""

    def test_unsupported_extension(self):
        result = parse_file("readme.md")
        self.assertIsNone(result)

    def test_parse_real_file(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("def example(): pass\n")
            f.flush()
            result = parse_file(f.name)
        os.unlink(f.name)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.symbols), 1)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases."""

    def test_empty_directory(self):
        tmpdir = tempfile.mkdtemp()
        rm = RepoMap(tmpdir).scan()
        self.assertEqual(len(rm.files), 0)
        self.assertEqual(rm.generate_map(), "")

    def test_empty_file(self):
        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, "empty.py"), "w") as f:
            f.write("")
        rm = RepoMap(tmpdir).scan()
        self.assertEqual(len(rm.files), 1)
        self.assertEqual(len(rm.symbols), 0)

    def test_binary_like_content(self):
        """Files with encoding issues should not crash."""
        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, "weird.py"), "wb") as f:
            f.write(b"\x00\x01\x02def broken\xff\xfe\n")
        rm = RepoMap(tmpdir).scan()
        # Should not crash, may or may not find symbols
        self.assertIsNotNone(rm)

    def test_skips_excluded_dirs(self):
        tmpdir = tempfile.mkdtemp()
        node_modules = os.path.join(tmpdir, "node_modules")
        os.makedirs(node_modules)
        with open(os.path.join(node_modules, "lib.js"), "w") as f:
            f.write("function internal() {}\n")
        with open(os.path.join(tmpdir, "app.js"), "w") as f:
            f.write("function main() {}\n")
        rm = RepoMap(tmpdir).scan()
        paths = list(rm.files.keys())
        self.assertEqual(len(paths), 1)
        self.assertIn("app.js", paths[0])


if __name__ == "__main__":
    unittest.main()
