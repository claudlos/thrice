#!/usr/bin/env python3
"""
Hermes Improvements Installer
Installs 38 standalone modules and patches 15 existing files.
All changes are reversible via uninstall.py.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
MODULES_DIR = SCRIPT_DIR / "modules"
PATCHES_DIR = SCRIPT_DIR / "patches"
MANIFEST_NAME = ".hermes-improvements-manifest.json"

# Map of patch files to their target paths in hermes-agent
PATCH_MAP = {
    "agent__context_compressor.py.patch": "agent/context_compressor.py",
    "agent__prompt_builder.py.patch": "agent/prompt_builder.py",
    "cron____init__.py.patch": "cron/__init__.py",
    "cron__jobs.py.patch": "cron/jobs.py",
    "cron__scheduler.py.patch": "cron/scheduler.py",
    "gateway__session.py.patch": "gateway/session.py",
    "gateway__stream_consumer.py.patch": "gateway/stream_consumer.py",
    "run_agent.py.patch": "run_agent.py",
    "tests__tools__test_delegate.py.patch": "tests/tools/test_delegate.py",
    "tools__cronjob_tools.py.patch": "tools/cronjob_tools.py",
    "tools__delegate_tool.py.patch": "tools/delegate_tool.py",
    "tools__file_tools.py.patch": "tools/file_tools.py",
    "tools__fuzzy_match.py.patch": "tools/fuzzy_match.py",
    "tools__process_registry.py.patch": "tools/process_registry.py",
    "tools__skills_tool.py.patch": "tools/skills_tool.py",
}

# Standalone modules to copy (relative to modules/)
STANDALONE_MODULES = [
    "adaptive_compression.py",
    "agent_loop_shadow.py",
    "agent_loop_state_machine.py",
    "auto_commit.py",
    "comment_strip_matcher.py",
    "consensus_protocol.py",
    "context_mentions.py",
    "context_optimizer.py",
    "debugging_guidance.py",
    "enforcement.py",
    "hermes_invariants.py",
    "invariants_unified.py",
    "prompt_algebra.py",
    "repo_map.py",
    "self_improving_tests.py",
    "self_reflection.py",
    "skill_verifier.py",
    "smart_truncation.py",
    "state_machine.py",
    "structured_errors.py",
    "test_lint_loop.py",
    "token_budgeting.py",
    "tool_alias_map.py",
    "tool_chain_analysis.py",
    "tool_selection_model.py",
    "verified_messages.py",
    # Wave 2 modules (plan.md #2-#12)
    "reproduce_first.py",
    "project_memory.py",
    "conversation_checkpoint.py",
    "edit_format.py",
    "error_recovery.py",
    "tool_call_batcher.py",
    "test_fix_loop.py",
    "context_gatherer.py",
    "change_impact.py",
    "agent_loop_components.py",
    "silent_error_audit.py",
    # SM-1 concrete state machine (paired with specs/tla/CronJob.tla)
    "cron_state_machine.py",
    # Wave 3: build & regression tooling
    "build_loop.py",
    "regression_bisector.py",
    # Wave 4: coding-quality gates
    "conventional_commit.py",
    "secret_scanner.py",
    "doctest_runner.py",
    "diff_preview.py",
    "semantic_diff.py",
    "trace_capture.py",
    "cost_estimator.py",
    "lsp_bridge.py",
    # Nested modules
    "agent/skill_index_cache.py",
]


def find_hermes_agent() -> Path:
    """Auto-detect hermes-agent directory.

    Checked locations, in priority order:

    1. ``$HERMES_DIR`` environment variable (explicit override).
    2. ``<cwd>/hermes-agent`` (common dev layout: thrice and hermes-agent
       side-by-side in a workspace).
    3. ``~/.hermes/hermes-agent`` (upstream-recommended install path).
    4. ``~/.config/hermes/hermes-agent`` (XDG-style Linux install).
    5. ``~/AppData/Local/hermes-agent`` (Windows install under ``LOCALAPPDATA``).
    6. ``~/AppData/Local/Programs/hermes/hermes-agent`` (alt Windows install).

    Any path with a ``run_agent.py`` at the top level is accepted.
    """
    env_override = os.environ.get("HERMES_DIR")
    candidates: list[Path] = []
    if env_override:
        candidates.append(Path(env_override))
    candidates += [
        Path.cwd() / "hermes-agent",
        Path.home() / ".hermes" / "hermes-agent",
        Path.home() / ".config" / "hermes" / "hermes-agent",
        Path.home() / "AppData" / "Local" / "hermes-agent",
        Path.home() / "AppData" / "Local" / "Programs" / "hermes" / "hermes-agent",
    ]
    for c in candidates:
        if (c / "run_agent.py").exists():
            return c
    return None


def backup_file(target: Path, backup_dir: Path) -> Path:
    """Backup a file before modifying it."""
    if not target.exists():
        return None
    rel = target.name
    backup = backup_dir / rel
    # Handle nested paths
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(target, backup)
    return backup


def apply_patch(hermes_dir: Path, patch_file: Path, target_rel: str, dry_run: bool = False) -> bool:
    """Apply a git-format patch to a target file."""
    target = hermes_dir / target_rel
    if not target.exists():
        print(f"  SKIP {target_rel} (file not found)")
        return False

    if dry_run:
        # Check if patch applies cleanly
        result = subprocess.run(
            ["git", "apply", "--check", str(patch_file)],
            capture_output=True, text=True, cwd=hermes_dir
        )
        if result.returncode == 0:
            print(f"  OK   {target_rel} (would apply cleanly)")
            return True
        else:
            # Check if already applied (reverse check)
            rev = subprocess.run(
                ["git", "apply", "--check", "--reverse", str(patch_file)],
                capture_output=True, text=True, cwd=hermes_dir
            )
            if rev.returncode == 0:
                print(f"  SKIP {target_rel} (already applied)")
                return True
            print(f"  WARN {target_rel} (may need manual resolution)")
            return False

    result = subprocess.run(
        ["git", "apply", str(patch_file)],
        capture_output=True, text=True, cwd=hermes_dir
    )
    if result.returncode == 0:
        print(f"  OK   {target_rel}")
        return True

    # Try with --3way for fuzzy matching
    result = subprocess.run(
        ["git", "apply", "--3way", str(patch_file)],
        capture_output=True, text=True, cwd=hermes_dir
    )
    if result.returncode == 0:
        print(f"  OK   {target_rel} (3-way merge)")
        return True

    # Try with increased fuzz
    result = subprocess.run(
        ["git", "apply", "--reject", str(patch_file)],
        capture_output=True, text=True, cwd=hermes_dir
    )
    if result.returncode == 0:
        print(f"  WARN {target_rel} (applied with rejects — check .rej files)")
        return True

    print(f"  FAIL {target_rel} — {result.stderr.strip()}")
    return False


def copy_module(src: Path, dst: Path, dry_run: bool = False, backup_dir: Path = None) -> bool:
    """Copy a standalone module, backing up existing file first."""
    if dry_run:
        exists = "overwrite" if dst.exists() else "new"
        print(f"  OK   {dst.name} ({exists})")
        return True

    dst.parent.mkdir(parents=True, exist_ok=True)
    # Backup existing file before overwriting
    if dst.exists() and backup_dir is not None:
        backup_path = backup_dir / f"module__{dst.name}"
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(dst, backup_path)
    shutil.copy2(src, dst)
    print(f"  OK   {dst.name}")
    return True


def syntax_check(hermes_dir: Path, files: list) -> int:
    """Run py_compile on files, return error count."""
    errors = 0
    for f in files:
        path = hermes_dir / f
        if path.exists():
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", str(path)],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"  SYNTAX ERROR: {f}")
                print(f"    {result.stderr.strip()}")
                errors += 1
    return errors


def install(hermes_dir: Path, dry_run: bool = False, skip_patches: bool = False):
    """Main install routine."""
    manifest = {
        "installed_at": datetime.now().isoformat(),
        "hermes_dir": str(hermes_dir),
        "modules": [],
        "patches": [],
        "backups_dir": None,
    }

    # Create backup directory
    backup_dir = hermes_dir / ".hermes-improvements-backup"
    if not dry_run:
        backup_dir.mkdir(exist_ok=True)
        manifest["backups_dir"] = str(backup_dir)

    # Phase 1: Copy standalone modules
    print("\n=== Phase 1: Installing standalone modules ===\n")
    modules_ok = 0
    for mod_rel in STANDALONE_MODULES:
        src = MODULES_DIR / mod_rel
        dst = hermes_dir / mod_rel
        if not src.exists():
            print(f"  MISS {mod_rel} (not in installer)")
            continue
        if copy_module(src, dst, dry_run, backup_dir=backup_dir):
            modules_ok += 1
            manifest["modules"].append(mod_rel)

    print(f"\n  {modules_ok}/{len(STANDALONE_MODULES)} modules installed")

    # Phase 2: Apply patches to existing files
    if not skip_patches:
        print("\n=== Phase 2: Patching existing files ===\n")
        patches_ok = 0
        patches_fail = 0
        patched_files = []  # Track successfully patched files for rollback
        for patch_name, target_rel in PATCH_MAP.items():
            patch_file = PATCHES_DIR / patch_name
            if not patch_file.exists():
                print(f"  MISS {patch_name}")
                continue

            # Backup before patching
            if not dry_run:
                target = hermes_dir / target_rel
                if target.exists():
                    backup_rel = target_rel.replace("/", "__")
                    backup = backup_dir / backup_rel
                    shutil.copy2(target, backup)

            if apply_patch(hermes_dir, patch_file, target_rel, dry_run):
                patches_ok += 1
                manifest["patches"].append(target_rel)
                patched_files.append(target_rel)
            else:
                patches_fail += 1
                # Rollback: restore all already-patched files from backups
                if not dry_run and patched_files:
                    print(f"\n  Rolling back {len(patched_files)} already-patched files...")
                    for prev_rel in patched_files:
                        prev_backup_rel = prev_rel.replace("/", "__")
                        prev_backup = backup_dir / prev_backup_rel
                        prev_target = hermes_dir / prev_rel
                        if prev_backup.exists():
                            shutil.copy2(prev_backup, prev_target)
                            print(f"  ROLLBACK {prev_rel}")
                    manifest["patches"] = []
                    patched_files = []
                break  # Stop applying further patches

        print(f"\n  {patches_ok}/{len(PATCH_MAP)} patches applied, {patches_fail} failed")
    else:
        print("\n=== Phase 2: SKIPPED (--skip-patches) ===")

    # Phase 3: Syntax validation
    print("\n=== Phase 3: Syntax validation ===\n")
    all_files = manifest["modules"] + manifest["patches"]
    if dry_run:
        print("  (skipped in dry-run mode)")
    else:
        errors = syntax_check(hermes_dir, all_files)
        if errors == 0:
            print(f"  All {len(all_files)} files pass syntax check")
        else:
            print(f"  {errors} syntax errors found!")

    # Save manifest
    if not dry_run:
        manifest_path = hermes_dir / MANIFEST_NAME
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest saved to {manifest_path}")

    # Summary
    print("\n" + "=" * 50)
    if dry_run:
        print("DRY RUN COMPLETE — no changes were made")
    else:
        print("INSTALLATION COMPLETE")
        print(f"  Modules: {modules_ok}")
        print(f"  Patches: {len(manifest['patches'])}")
        print(f"  Backups: {backup_dir}")
    print()
    print("Env-gated features (off by default):")
    print("  export HERMES_SM_SHADOW=true       # shadow state machine")
    print("  export HERMES_AUTO_COMMIT=true      # auto git commits")
    print("  export HERMES_SELF_TEST=true        # self-improving tests")
    print("  export HERMES_PREDICT_BUDGET=true   # token budget prediction")
    print()
    print("To uninstall: python3 uninstall.py")
    print("To re-apply after Hermes update: python3 update.py")


def main():
    parser = argparse.ArgumentParser(description="Install Hermes Improvements")
    parser.add_argument("--hermes-dir", type=Path, help="Path to hermes-agent directory")
    parser.add_argument("--dry-run", action="store_true", help="Check what would be done without making changes")
    parser.add_argument("--skip-patches", action="store_true", help="Only install standalone modules, skip patching existing files")
    parser.add_argument("--modules-only", action="store_true", help="Alias for --skip-patches")
    args = parser.parse_args()

    hermes_dir = args.hermes_dir or find_hermes_agent()
    if not hermes_dir:
        print("ERROR: Could not find hermes-agent directory.")
        print("Use --hermes-dir /path/to/hermes-agent")
        sys.exit(1)

    if not (hermes_dir / "run_agent.py").exists():
        print(f"ERROR: {hermes_dir} doesn't look like a hermes-agent directory")
        sys.exit(1)

    skip_patches = args.skip_patches or args.modules_only

    print(f"Hermes Improvements Installer")
    print(f"Target: {hermes_dir}")
    if args.dry_run:
        print("Mode: DRY RUN")
    print()

    # Check for existing installation
    manifest_path = hermes_dir / MANIFEST_NAME
    if manifest_path.exists():
        print("WARNING: Previous installation detected.")
        print("Run uninstall.py first, or use update.py to re-apply.")
        if not args.dry_run:
            response = input("Continue anyway? [y/N] ")
            if response.lower() != "y":
                sys.exit(0)

    install(hermes_dir, dry_run=args.dry_run, skip_patches=skip_patches)


if __name__ == "__main__":
    main()
