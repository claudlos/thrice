"""Regression tests for installer/update/uninstall safety."""

from __future__ import annotations

import json
import re
from pathlib import Path

import install
import uninstall
import update


def _fake_hermes(tmp_path: Path) -> Path:
    hermes = tmp_path / "hermes-agent"
    hermes.mkdir()
    (hermes / "run_agent.py").write_text("", encoding="utf-8")
    return hermes


def _py_modules_from_pyproject(root: Path) -> set[str]:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"py-modules\s*=\s*\[(.*?)\]", text, re.S)
    assert match, "pyproject.toml must declare py-modules"
    return set(re.findall(r'"([^"]+)"', match.group(1)))


def test_pyproject_lists_every_top_level_module():
    root = Path(__file__).resolve().parents[1]
    top_level = {p.stem for p in (root / "modules").glob("*.py")}

    declared = _py_modules_from_pyproject(root)

    assert top_level <= declared


def test_uninstall_restores_overwritten_modules(tmp_path):
    hermes = _fake_hermes(tmp_path)
    original = "ORIGINAL_VALUE = 1\n"
    (hermes / "adaptive_compression.py").write_text(original, encoding="utf-8")

    assert install.install(hermes, dry_run=False, skip_patches=True) is True
    manifest = json.loads((hermes / install.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["module_backups"]["adaptive_compression.py"].startswith("module__")
    assert (hermes / "adaptive_compression.py").read_text(encoding="utf-8") != original

    uninstall.uninstall(hermes, dry_run=False)

    assert (hermes / "adaptive_compression.py").read_text(encoding="utf-8") == original
    assert not (hermes / "auto_commit.py").exists()


def test_update_manifest_records_only_copied_modules(tmp_path, monkeypatch):
    hermes = _fake_hermes(tmp_path)
    monkeypatch.setattr(update, "PATCH_MAP", {})

    assert update.update(hermes, dry_run=False, modules_only=False) is True

    manifest = json.loads((hermes / update.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert set(manifest["modules"]) == set(update.STANDALONE_MODULES)
    assert "agent/__init__.py" not in manifest["modules"]
    assert "skill_index_cache.py" not in manifest["modules"]


def test_install_dry_run_reports_required_patch_failures(tmp_path):
    hermes = _fake_hermes(tmp_path)

    assert install.install(hermes, dry_run=True, skip_patches=False) is False
    assert not (hermes / install.MANIFEST_NAME).exists()
