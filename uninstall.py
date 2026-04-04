#!/usr/bin/env python3
"""
Hermes Improvements Uninstaller
Removes standalone modules and restores patched files from backups.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

MANIFEST_NAME = ".hermes-improvements-manifest.json"


def find_hermes_agent() -> Path:
    candidates = [
        Path.home() / ".hermes" / "hermes-agent",
        Path.home() / ".config" / "hermes" / "hermes-agent",
    ]
    for c in candidates:
        if (c / "run_agent.py").exists():
            return c
    return None


def uninstall(hermes_dir: Path, dry_run: bool = False):
    manifest_path = hermes_dir / MANIFEST_NAME
    if not manifest_path.exists():
        print("ERROR: No installation manifest found.")
        print("Hermes improvements may not be installed, or were installed manually.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    backup_dir = Path(manifest["backups_dir"]) if manifest.get("backups_dir") else None

    # Phase 1: Remove standalone modules
    print("\n=== Phase 1: Removing standalone modules ===\n")
    removed = 0
    for mod_rel in manifest.get("modules", []):
        target = hermes_dir / mod_rel
        if target.exists():
            if dry_run:
                print(f"  DEL  {mod_rel}")
            else:
                target.unlink()
                print(f"  DEL  {mod_rel}")
            removed += 1
        else:
            print(f"  SKIP {mod_rel} (already gone)")

    print(f"\n  {removed} modules removed")

    # Phase 2: Restore patched files from backups
    print("\n=== Phase 2: Restoring patched files ===\n")
    restored = 0
    if backup_dir and backup_dir.exists():
        for target_rel in manifest.get("patches", []):
            backup_name = target_rel.replace("/", "__")
            backup = backup_dir / backup_name
            target = hermes_dir / target_rel
            if backup.exists():
                if dry_run:
                    print(f"  RESTORE {target_rel}")
                else:
                    shutil.copy2(backup, target)
                    print(f"  RESTORE {target_rel}")
                restored += 1
            else:
                print(f"  SKIP {target_rel} (no backup found)")
                if not dry_run:
                    print(f"    You may need to: cd {hermes_dir} && git checkout -- {target_rel}")
    else:
        print("  No backup directory found.")
        print("  To restore patched files manually:")
        print(f"    cd {hermes_dir}")
        for target_rel in manifest.get("patches", []):
            print(f"    git checkout main -- {target_rel}")

    print(f"\n  {restored} files restored")

    # Cleanup
    if not dry_run:
        manifest_path.unlink()
        if backup_dir and backup_dir.exists():
            # Validate backup_dir is under hermes_dir before rmtree
            try:
                backup_dir.resolve().relative_to(hermes_dir.resolve())
            except ValueError:
                print(f"\nWARNING: Backup directory {backup_dir} is not under {hermes_dir}, skipping removal")
            else:
                shutil.rmtree(backup_dir)
                print(f"\nBackup directory removed: {backup_dir}")
        print("\nManifest removed.")

    print("\n" + "=" * 50)
    if dry_run:
        print("DRY RUN — no changes made")
    else:
        print("UNINSTALL COMPLETE")
        print("Hermes is back to its original state.")


def main():
    parser = argparse.ArgumentParser(description="Uninstall Hermes Improvements")
    parser.add_argument("--hermes-dir", type=Path, help="Path to hermes-agent directory")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    hermes_dir = args.hermes_dir or find_hermes_agent()
    if not hermes_dir:
        print("ERROR: Could not find hermes-agent directory.")
        sys.exit(1)

    print(f"Hermes Improvements Uninstaller")
    print(f"Target: {hermes_dir}")
    if args.dry_run:
        print("Mode: DRY RUN")

    uninstall(hermes_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
