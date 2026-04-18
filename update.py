#!/usr/bin/env python3
"""
Hermes Improvements Updater
Re-applies improvements after a Hermes update (git pull).

Strategy:
1. Standalone modules: just re-copy (they don't conflict)
2. Patched files: attempt to re-apply patches on the new code
3. If patches fail, report which files need manual attention

Usage:
    python3 update.py              # re-apply everything
    python3 update.py --dry-run    # check what would happen
    python3 update.py --modules-only  # just re-copy modules (safe)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from install import PATCH_MAP, STANDALONE_MODULES

SCRIPT_DIR = Path(__file__).parent.resolve()
MANIFEST_NAME = ".hermes-improvements-manifest.json"


def find_hermes_agent() -> Path:
    """Find hermes-agent directory - matches install.py::find_hermes_agent."""
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


def update(hermes_dir: Path, dry_run: bool = False, modules_only: bool = False):
    modules_dir = SCRIPT_DIR / "modules"
    patches_dir = SCRIPT_DIR / "patches"

    # Phase 1: Re-copy standalone modules (always safe)
    print("\n=== Phase 1: Re-copying standalone modules ===\n")
    copied = 0
    for mod_rel_str in sorted(STANDALONE_MODULES):
        mod_path = modules_dir / mod_rel_str
        if not mod_path.exists():
            print(f"  SKIP {mod_rel_str} (not found in modules/)")
            continue
        rel = Path(mod_rel_str)
        dst = hermes_dir / rel
        if dry_run:
            status = "overwrite" if dst.exists() else "new"
            print(f"  OK   {rel} ({status})")
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mod_path, dst)
            print(f"  OK   {rel}")
        copied += 1
    print(f"\n  {copied} modules copied")

    if modules_only:
        print("\n=== Phase 2: SKIPPED (--modules-only) ===")
        print("\nDone. Standalone modules updated.")
        print("Patched files were NOT re-applied — they may need manual attention")
        print("if Hermes updated those files.")
        return

    # Phase 2: Re-apply patches
    print("\n=== Phase 2: Re-applying patches ===\n")

    # First, backup current files
    backup_dir = hermes_dir / ".hermes-improvements-backup"
    if not dry_run:
        backup_dir.mkdir(exist_ok=True)

    ok = 0
    fail = 0
    failed_files = []

    for patch_file in sorted(patches_dir.glob("*.patch")):
        # Look up target path from PATCH_MAP instead of deriving it
        target_rel = PATCH_MAP.get(patch_file.name)
        if target_rel is None:
            print(f"  SKIP {patch_file.name} (not in PATCH_MAP)")
            continue
        target = hermes_dir / target_rel

        if not target.exists():
            print(f"  SKIP {target_rel} (not found)")
            continue

        # Backup
        if not dry_run:
            backup_name = target_rel.replace("/", "__")
            shutil.copy2(target, backup_dir / backup_name)

        if dry_run:
            result = subprocess.run(
                ["git", "apply", "--check", str(patch_file)],
                capture_output=True, text=True, cwd=hermes_dir
            )
            if result.returncode == 0:
                print(f"  OK   {target_rel}")
                ok += 1
            else:
                # Already applied?
                print(f"  WARN {target_rel} (may already be applied or needs manual merge)")
                fail += 1
                failed_files.append(target_rel)
        else:
            # Try normal apply
            result = subprocess.run(
                ["git", "apply", str(patch_file)],
                capture_output=True, text=True, cwd=hermes_dir
            )
            if result.returncode == 0:
                print(f"  OK   {target_rel}")
                ok += 1
            else:
                # Try 3-way merge
                result = subprocess.run(
                    ["git", "apply", "--3way", str(patch_file)],
                    capture_output=True, text=True, cwd=hermes_dir
                )
                if result.returncode == 0:
                    print(f"  OK   {target_rel} (3-way merge)")
                    ok += 1
                else:
                    print(f"  FAIL {target_rel}")
                    fail += 1
                    failed_files.append(target_rel)

    print(f"\n  {ok} patches applied, {fail} failed")

    if failed_files:
        print("\n  Files needing manual attention:")
        for f in failed_files:
            print(f"    - {f}")
        print()
        print("  These patches may already be applied, or the upstream file")
        print("  changed too much. Compare against the integration branch:")
        print(f"    cd {hermes_dir}")
        print("    git diff hermes-improvements-integration -- <file>")

    # Update manifest
    if not dry_run:
        from datetime import datetime
        manifest = {
            "installed_at": datetime.now().isoformat(),
            "hermes_dir": str(hermes_dir),
            "modules": [str(p.relative_to(modules_dir)) for p in modules_dir.rglob("*.py")],
            "patches": [f for f in [pf.stem.replace("__", "/") for pf in patches_dir.glob("*.patch")] if f not in failed_files],
            "backups_dir": str(backup_dir),
            "failed_patches": failed_files,
        }
        manifest_path = hermes_dir / MANIFEST_NAME
        with open(manifest_path, "w") as fp:
            json.dump(manifest, fp, indent=2)

    print("\n" + "=" * 50)
    if dry_run:
        print("DRY RUN — no changes made")
    else:
        print("UPDATE COMPLETE")


def main():
    parser = argparse.ArgumentParser(description="Re-apply Hermes Improvements after update")
    parser.add_argument("--hermes-dir", type=Path, help="Path to hermes-agent directory")
    parser.add_argument("--dry-run", action="store_true", help="Check what would happen")
    parser.add_argument("--modules-only", action="store_true", help="Only re-copy modules, skip patches")
    args = parser.parse_args()

    hermes_dir = args.hermes_dir or find_hermes_agent()
    if not hermes_dir:
        print("ERROR: Could not find hermes-agent directory.")
        sys.exit(1)

    print("Hermes Improvements Updater")
    print(f"Target: {hermes_dir}")
    if args.dry_run:
        print("Mode: DRY RUN")

    update(hermes_dir, dry_run=args.dry_run, modules_only=args.modules_only)


if __name__ == "__main__":
    main()
